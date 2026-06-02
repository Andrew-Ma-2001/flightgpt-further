"""Hierarchical Localization Module (HLM) orchestrator.

Wires the two-phase coarse-to-fine pipeline:

    load 4K -> 2K downsample -> Glance (VLM) -> back-project to 4K
            -> dynamic crop -> Focus (VLM) -> inverse-map to 4K -> final point

The module depends only on a VLM "agent" exposing ``act_on_image(image_path,
prompt) -> str`` (see ``navgym.agents.CityNavAgent.GPTAgent``). All geometry is
delegated to ``coordinate_mapper`` and ``cropper``; all parsing to ``parsers``;
all prompts to ``prompt_templates``. The final point is always returned in the
global (4K) pixel frame so it can be fed directly to ``compute_pose`` in
``eval.py`` WITHOUT any further scaling.

Robustness / fallbacks:
* If ``enable_fine`` is False, only the Glance phase runs and its back-projected
  point is returned.
* If the Glance phase fails to produce a usable point (``[0, 0]`` sentinel), the
  Focus phase is skipped and the sentinel is propagated (downstream treats it as
  "no prediction").
* If the Focus phase fails to parse a point, the pipeline falls back to the
  back-projected coarse point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from navgym.hlm import coordinate_mapper as cm
from navgym.hlm import cropper as cp
from navgym.hlm.parsers import (
    parse_location,
    parse_bbox,
    location_parse_failed,
    bbox_parse_failed,
)
from navgym.hlm.prompt_templates import get_glance_prompt, get_fine_prompt

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class HLMConfig:
    """Configuration for the Hierarchical Localization Module.

    The crop-related fields share names with ``cropper.CropParams`` so a config
    instance can be passed straight to the cropper.
    """

    coarse_max_size: int = 2048          # longest side of the Glance (2K) image
    # --- cropper params (duck-compatible with cropper.CropParams) ---
    adaptive_crop: bool = True
    fixed_crop_size: int = 1024
    crop_alpha: float = 3.0
    crop_min: int = 512
    crop_max: int = 2048
    # --- focus phase ---
    model_input_size: Optional[int] = None   # if set, resize crop (longest side) before VLM
    enable_fine: bool = True                  # global kill-switch for the Focus phase
    # --- io ---
    work_dir: Optional[str] = None            # where intermediate images are written
    jpeg_quality: int = 95

    def crop_params(self) -> cp.CropParams:
        return cp.CropParams(
            adaptive_crop=self.adaptive_crop,
            fixed_crop_size=self.fixed_crop_size,
            crop_alpha=self.crop_alpha,
            crop_min=self.crop_min,
            crop_max=self.crop_max,
        )


@dataclass
class HLMResult:
    """Structured output of one HLM forward pass (all points in 4K unless noted)."""

    coarse_xy_4k: Point                      # Glance point back-projected to 4K
    landmark_bbox_4k: Optional[BBox]         # Glance landmark bbox in 4K (or None)
    crop_offset: Point                       # (o_x, o_y) crop top-left in 4K
    crop_size: Point                         # (W_crop, H_crop) actual crop size
    fine_xy_4k: Point                        # refined point in 4K
    final_xy: Point                          # alias of fine_xy_4k (downstream name)
    coarse_xy_2k: Point                      # raw Glance point in 2K frame
    scale_4k_to_2k: Tuple[float, float]      # (s_x, s_y)
    used_fine: bool                          # True if Focus phase contributed
    raw_coarse_response: str
    raw_fine_response: str
    coarse_image_path: Optional[str] = None
    crop_image_path: Optional[str] = None


class HierarchicalLocalizationModule:
    """Coarse-to-fine localization wrapper around a single VLM agent."""

    def __init__(self, agent, config: Optional[HLMConfig] = None):
        """
        Args:
            agent: object exposing ``act_on_image(image_path, prompt) -> str``
                and (optionally) ``target_description``.
            config: an :class:`HLMConfig`. Defaults are used if omitted.
        """
        if not hasattr(agent, "act_on_image"):
            raise TypeError("agent must provide an `act_on_image(image_path, prompt)` method")
        self.agent = agent
        self.config = config or HLMConfig()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def forward(self, image_4k_path: str, instruction: str, start_px: List[int]) -> HLMResult:
        """Run the full HLM pipeline on one map image.

        Args:
            image_4k_path: path to the full-resolution rendered map.
            instruction: target description (falls back to ``agent.target_description``).
            start_px: UAV start pixel ``[x, y]`` in the 4K frame.

        Returns:
            :class:`HLMResult` with ``final_xy`` in the 4K frame.
        """
        if instruction is None:
            instruction = getattr(self.agent, "target_description", "")

        image_4k = cv2.imread(image_4k_path)
        if image_4k is None:
            raise FileNotFoundError(f"could not read image: {image_4k_path}")
        h4, w4 = image_4k.shape[0], image_4k.shape[1]

        # ---- S1: 2K downsampling (aspect-preserving, never upscale) ----
        coarse_img, scale = self._downsample(image_4k, self.config.coarse_max_size)
        w2, h2 = coarse_img.shape[1], coarse_img.shape[0]
        coarse_path = self._write(image_4k_path, "_hlm_coarse", coarse_img)

        # ---- S2: Glance localization ----
        start_px_2k = cm.global_to_coarse((start_px[0], start_px[1]), scale)
        glance_prompt = get_glance_prompt(instruction, list(start_px_2k))
        raw_coarse = self.agent.act_on_image(coarse_path, glance_prompt)

        coarse_xy_2k = parse_location(raw_coarse)
        bbox_2k = parse_bbox(raw_coarse, "landmark_bbox")

        # ---- S3: back-project coarse point + bbox to 4K ----
        if location_parse_failed(coarse_xy_2k):
            # No usable Glance point: propagate the sentinel, skip Focus.
            coarse_xy_4k = (0, 0)
            return HLMResult(
                coarse_xy_4k=coarse_xy_4k, landmark_bbox_4k=None,
                crop_offset=(0, 0), crop_size=(0, 0),
                fine_xy_4k=coarse_xy_4k, final_xy=coarse_xy_4k,
                coarse_xy_2k=tuple(coarse_xy_2k), scale_4k_to_2k=scale,
                used_fine=False, raw_coarse_response=raw_coarse, raw_fine_response="",
                coarse_image_path=coarse_path,
            )

        coarse_xy_2k = cm.clamp_point(coarse_xy_2k, w2, h2)
        coarse_xy_4k = cm.coarse_to_global(coarse_xy_2k, scale)
        coarse_xy_4k = cm.clamp_point(coarse_xy_4k, w4, h4)

        bbox_4k: Optional[BBox] = None
        if not bbox_parse_failed(bbox_2k):
            bbox_4k = cm.coarse_bbox_to_global(tuple(bbox_2k), scale)

        # ---- early exit if Focus disabled ----
        if not self.config.enable_fine:
            return HLMResult(
                coarse_xy_4k=coarse_xy_4k, landmark_bbox_4k=bbox_4k,
                crop_offset=(0, 0), crop_size=(0, 0),
                fine_xy_4k=coarse_xy_4k, final_xy=coarse_xy_4k,
                coarse_xy_2k=tuple(coarse_xy_2k), scale_4k_to_2k=scale,
                used_fine=False, raw_coarse_response=raw_coarse, raw_fine_response="",
                coarse_image_path=coarse_path,
            )

        # ---- S4: dynamic crop from the 4K image ----
        crop_img, offset, crop_size = cp.crop(
            image_4k, coarse_xy_4k, bbox_4k, self.config.crop_params()
        )

        # optional resize of the crop before the model (aspect-preserving)
        crop_sent, resize_scale = self._maybe_resize_crop(crop_img)
        crop_path = self._write(image_4k_path, "_hlm_crop", crop_sent)
        sent_w, sent_h = crop_sent.shape[1], crop_sent.shape[0]

        # ---- S5: Focus localization ----
        fine_prompt = get_fine_prompt(instruction, sent_w, sent_h)
        raw_fine = self.agent.act_on_image(crop_path, fine_prompt)
        fine_xy_crop = parse_location(raw_fine)

        # ---- S6: inverse-map Focus point to 4K (fallback to coarse) ----
        if location_parse_failed(fine_xy_crop):
            fine_xy_4k = coarse_xy_4k
            used_fine = False
        else:
            fine_xy_crop = cm.clamp_point(fine_xy_crop, sent_w, sent_h)
            fine_xy_4k = cm.crop_to_global(fine_xy_crop, offset, resize_scale)
            fine_xy_4k = cm.clamp_point(fine_xy_4k, w4, h4)
            used_fine = True

        return HLMResult(
            coarse_xy_4k=coarse_xy_4k, landmark_bbox_4k=bbox_4k,
            crop_offset=tuple(offset), crop_size=tuple(crop_size),
            fine_xy_4k=fine_xy_4k, final_xy=fine_xy_4k,
            coarse_xy_2k=tuple(coarse_xy_2k), scale_4k_to_2k=scale,
            used_fine=used_fine, raw_coarse_response=raw_coarse, raw_fine_response=raw_fine,
            coarse_image_path=coarse_path, crop_image_path=crop_path,
        )

    # convenience alias
    __call__ = forward

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _downsample(image: np.ndarray, max_size: int) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Aspect-preserving downscale so the longest side == ``max_size``.

        Never upscales: if the image is already <= ``max_size`` on its longest
        side it is returned unchanged with scale ``(1.0, 1.0)``.
        """
        h, w = image.shape[0], image.shape[1]
        longest = max(w, h)
        f = min(1.0, max_size / float(longest))
        if f >= 1.0:
            return image, (1.0, 1.0)
        new_w = max(1, int(round(w * f)))
        new_h = max(1, int(round(h * f)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, cm.compute_scale((w, h), (new_w, new_h))

    def _maybe_resize_crop(self, crop_img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resize the crop to ``model_input_size`` (longest side), if configured.

        Returns ``(crop_to_send, (r_x, r_y))`` where ``r = sent / crop``. When no
        resize happens, ``r = (1.0, 1.0)``.
        """
        target = self.config.model_input_size
        if not target:
            return crop_img, (1.0, 1.0)
        h, w = crop_img.shape[0], crop_img.shape[1]
        longest = max(w, h)
        if longest == target:
            return crop_img, (1.0, 1.0)
        f = target / float(longest)
        new_w = max(1, int(round(w * f)))
        new_h = max(1, int(round(h * f)))
        interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(crop_img, (new_w, new_h), interpolation=interp)
        return resized, (new_w / float(w), new_h / float(h))

    def _write(self, base_path: str, suffix: str, image: np.ndarray) -> str:
        """Write an intermediate image next to ``base_path`` (or in ``work_dir``)."""
        stem, ext = os.path.splitext(os.path.basename(base_path))
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        out_dir = self.config.work_dir or os.path.dirname(os.path.abspath(base_path))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{stem}{suffix}{ext}")
        params = [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)] if ext != ".png" else []
        cv2.imwrite(out_path, image, params)
        return out_path
