"""Dynamic region cropping for the HLM Focus phase.

Given the back-projected coarse point in the global (4K) frame, this module
extracts a high-resolution window from the original map. The window size is
either fixed or adaptively derived from the predicted landmark bounding box
(larger landmarks tolerate larger windows). The window is then slid inside the
image bounds so it never falls off the edge, which preserves resolution (we do
not pad-and-shrink).

See ``navgym.hlm.coordinate_mapper`` for the inverse mapping that turns a
Focus-phase point (in the crop frame) back into the global frame using the
``offset`` this module returns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]
Size = Tuple[int, int]


@dataclass
class CropParams:
    """Parameters controlling the dynamic cropper.

    These attribute names are a subset of ``HLMConfig`` (Task 6), so an
    ``HLMConfig`` instance can be passed anywhere a ``CropParams`` is expected
    (duck typing).

    Attributes:
        adaptive_crop: if False, always use ``fixed_crop_size``.
        fixed_crop_size: square crop side used when adaptive is off or no bbox.
        crop_alpha: multiplier on the landmark bbox diagonal -> crop side.
        crop_min: lower clamp on the adaptive crop side (px).
        crop_max: upper clamp on the adaptive crop side (px).
    """

    adaptive_crop: bool = True
    fixed_crop_size: int = 1024
    crop_alpha: float = 3.0
    crop_min: int = 512
    crop_max: int = 2048


def compute_crop_size(bbox_4k: Optional[BBox], params: CropParams) -> Size:
    """Choose the crop window size ``(W_crop, H_crop)`` (square).

    Fixed mode (or missing bbox) returns ``(fixed_crop_size, fixed_crop_size)``.
    Adaptive mode returns ``clip(ceil(alpha * diag), crop_min, crop_max)`` where
    ``diag`` is the landmark bbox diagonal. A degenerate bbox (``diag == 0``)
    yields ``crop_min``.
    """
    if not params.adaptive_crop or bbox_4k is None:
        s = int(params.fixed_crop_size)
        return (s, s)

    x1, y1, x2, y2 = bbox_4k
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    diag = float(np.hypot(w, h))
    size = int(np.ceil(params.crop_alpha * diag))
    size = int(np.clip(size, params.crop_min, params.crop_max))
    return (size, size)


def crop_window(image_4k: np.ndarray, center: Point, size: Size) -> Tuple[np.ndarray, Point]:
    """Crop a ``size`` window from ``image_4k`` centered at ``center``.

    The requested size is first clamped to the image dimensions. The window is
    then slid fully inside the image so the returned crop always has exactly the
    (clamped) requested size.

    Args:
        image_4k: ``(H, W, C)`` or ``(H, W)`` array (the original map).
        center: ``(cx, cy)`` crop center in the global frame.
        size: ``(W_crop, H_crop)`` requested window size.

    Returns:
        ``(crop, offset)`` where ``offset = (o_x, o_y)`` is the crop top-left in
        the global frame.
    """
    if image_4k.ndim < 2:
        raise ValueError(f"image_4k must be at least 2D, got ndim={image_4k.ndim}")
    h4, w4 = image_4k.shape[0], image_4k.shape[1]
    if w4 <= 0 or h4 <= 0:
        raise ValueError(f"image has invalid shape {image_4k.shape}")

    wc = int(min(max(size[0], 1), w4))
    hc = int(min(max(size[1], 1), h4))

    # Clamp the center inside the image before deriving the window origin.
    cx = int(min(max(center[0], 0), w4 - 1))
    cy = int(min(max(center[1], 0), h4 - 1))

    ox = int(min(max(cx - wc // 2, 0), w4 - wc))
    oy = int(min(max(cy - hc // 2, 0), h4 - hc))

    crop = image_4k[oy:oy + hc, ox:ox + wc].copy()
    return crop, (ox, oy)


def crop(
    image_4k: np.ndarray,
    center_4k: Point,
    bbox_4k: Optional[BBox],
    params: CropParams,
) -> Tuple[np.ndarray, Point, Size]:
    """Full dynamic crop: choose size, then extract the clamped window.

    Returns:
        ``(crop_image, offset, size)`` where ``size`` is the *actual* crop size
        ``(crop.shape[1], crop.shape[0])`` after image-bound clamping.
    """
    requested = compute_crop_size(bbox_4k, params)
    crop_image, offset = crop_window(image_4k, center_4k, requested)
    actual = (crop_image.shape[1], crop_image.shape[0])
    return crop_image, offset, actual
