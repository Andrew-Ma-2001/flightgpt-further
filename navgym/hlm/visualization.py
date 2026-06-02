"""Debugging visualizations for the HLM pipeline.

``visualize_hlm`` renders, on a copy of the original 4K map, the four debug
layers:

* A - global Glance point (yellow)
* B - dynamic crop region (cyan rectangle)
* C - refined Focus point (green) with a Glance->Focus arrow
* D - ground-truth point (red) with a Focus->GT line and the pixel-NE label

Colors follow OpenCV BGR ordering and the conventions already used by
``eval.py:visualize_prediction``.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2

Point = Tuple[int, int]

_YELLOW = (0, 255, 255)   # Glance
_CYAN = (255, 255, 0)     # crop region
_GREEN = (0, 255, 0)      # Focus / refined
_RED = (0, 0, 255)        # ground truth


def _pt(p) -> Point:
    return (int(p[0]), int(p[1]))


def visualize_hlm(
    image_4k_path: str,
    result,
    gt_xy_4k: Optional[Point] = None,
    ne_px: Optional[float] = None,
    save_path: str = "hlm_debug.jpg",
    radius: int = 25,
) -> str:
    """Draw the HLM debug layers and save the annotated image.

    Args:
        image_4k_path: path to the original full-resolution map.
        result: an :class:`navgym.hlm.HLMResult`.
        gt_xy_4k: optional ground-truth point ``(x, y)`` in the 4K frame.
        ne_px: optional pixel-space navigation error to annotate.
        save_path: output image path (parent dirs are created).
        radius: marker radius in pixels.

    Returns:
        The ``save_path`` written.
    """
    img = cv2.imread(image_4k_path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {image_4k_path}")

    # Layer B: crop region (only if a crop was actually taken)
    ox, oy = _pt(result.crop_offset)
    cw, ch = _pt(result.crop_size)
    if cw > 0 and ch > 0:
        cv2.rectangle(img, (ox, oy), (ox + cw, oy + ch), _CYAN, 4)
        cv2.putText(img, "crop", (ox, max(oy - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _CYAN, 2)

    coarse = _pt(result.coarse_xy_4k)
    fine = _pt(result.fine_xy_4k)

    # Layer C: Glance -> Focus arrow (only meaningful if Focus ran)
    if getattr(result, "used_fine", False) and coarse != fine:
        cv2.arrowedLine(img, coarse, fine, _GREEN, 3, tipLength=0.05)

    # Layer A: Glance point
    cv2.circle(img, coarse, radius, _YELLOW, -1)
    cv2.putText(img, "glance", (coarse[0] + radius, coarse[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, _YELLOW, 2)

    # Layer C: Focus point
    cv2.circle(img, fine, radius, _GREEN, -1)
    cv2.putText(img, "fine", (fine[0] + radius, fine[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, _GREEN, 2)

    # Layer D: ground truth + error line/label
    if gt_xy_4k is not None:
        gt = _pt(gt_xy_4k)
        cv2.circle(img, gt, radius, _RED, -1)
        cv2.putText(img, "gt", (gt[0] + radius, gt[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, _RED, 2)
        cv2.line(img, fine, gt, _RED, 2)
        if ne_px is not None:
            cv2.putText(img, f"NE_px={ne_px:.1f}", (ox, max(oy - 44, 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, _RED, 3)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, img)
    return save_path
