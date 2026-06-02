"""Coordinate transformations for the Hierarchical Localization Module (HLM).

Three pixel frames are involved (all use x = column / right, y = row / down,
origin at top-left, integer pixels):

* Global (4K) frame ``G``     - size ``(W4, H4)``. Ground truth and the final
  answer live here.
* Coarse (2K) frame ``C``      - size ``(W2, H2)``. An aspect-preserving
  downscale of ``G`` with scale ``(s_x, s_y) = (W2 / W4, H2 / H4)``.
* Local crop frame ``L``       - a window of size ``(W_crop, H_crop)`` taken
  from ``G`` at top-left offset ``(o_x, o_y)``, optionally resized to the model
  input size ``(W_in, H_in)`` with ``(r_x, r_y) = (W_in / W_crop, H_in / H_crop)``.
  When the crop is not resized, ``(r_x, r_y) = (1.0, 1.0)``.

Forward maps (G -> other)::

    T_{G->C}(x, y) = (s_x * x, s_y * y)
    T_{G->L}(x, y) = (r_x * (x - o_x), r_y * (y - o_y))

Inverse maps (other -> G) - what the pipeline actually consumes::

    T_{C->G}(x_c, y_c) = (x_c / s_x, y_c / s_y)
    T_{L->G}(x_l, y_l) = (o_x + x_l / r_x, o_y + y_l / r_y)

All functions are pure and stateless. Points are returned as integer ``(x, y)``
tuples (rounded), scales are floats. The transforms are mutual inverses up to
integer rounding (±1 px), which the unit tests assert.
"""

from __future__ import annotations

from typing import Tuple

Point = Tuple[int, int]
Scale = Tuple[float, float]
BBox = Tuple[int, int, int, int]


def _round(value: float) -> int:
    """Round-half-away-from-zero is not needed; use banker-free int rounding."""
    return int(round(value))


def clamp_point(xy: Tuple[float, float], width: int, height: int) -> Point:
    """Clamp a point into the valid pixel range ``[0, width-1] x [0, height-1]``."""
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got ({width}, {height})")
    x = min(max(_round(xy[0]), 0), width - 1)
    y = min(max(_round(xy[1]), 0), height - 1)
    return (x, y)


def compute_scale(src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> Scale:
    """Return ``(s_x, s_y)`` mapping ``src_size`` -> ``dst_size``.

    Args:
        src_size: ``(width, height)`` of the source frame (e.g. 4K).
        dst_size: ``(width, height)`` of the destination frame (e.g. 2K).
    """
    sw, sh = src_size
    dw, dh = dst_size
    if sw <= 0 or sh <= 0:
        raise ValueError(f"src_size must be positive, got {src_size}")
    return (dw / float(sw), dh / float(sh))


# --------------------------------------------------------------------------- #
# Coarse (2K) <-> Global (4K)
# --------------------------------------------------------------------------- #
def global_to_coarse(xy_4k: Tuple[float, float], scale: Scale) -> Point:
    """Forward map G -> C: ``(s_x * x, s_y * y)``."""
    sx, sy = scale
    return (_round(xy_4k[0] * sx), _round(xy_4k[1] * sy))


def coarse_to_global(xy_2k: Tuple[float, float], scale: Scale) -> Point:
    """Inverse map C -> G: ``(x_c / s_x, y_c / s_y)``.

    This back-projects a Glance-phase point into the full-resolution frame to
    obtain the crop center.
    """
    sx, sy = scale
    if sx == 0 or sy == 0:
        raise ValueError(f"scale components must be non-zero, got {scale}")
    return (_round(xy_2k[0] / sx), _round(xy_2k[1] / sy))


# --------------------------------------------------------------------------- #
# Local crop <-> Global (4K)
# --------------------------------------------------------------------------- #
def global_to_crop(
    xy_4k: Tuple[float, float],
    offset: Point,
    crop_resize_scale: Scale = (1.0, 1.0),
) -> Point:
    """Forward map G -> L: ``(r_x * (x - o_x), r_y * (y - o_y))``.

    Args:
        xy_4k: point in the global frame.
        offset: ``(o_x, o_y)`` top-left of the crop window in the global frame.
        crop_resize_scale: ``(r_x, r_y) = (W_in / W_crop, H_in / H_crop)``.
            Use ``(1.0, 1.0)`` when the crop is fed to the model unresized.
    """
    ox, oy = offset
    rx, ry = crop_resize_scale
    return (_round((xy_4k[0] - ox) * rx), _round((xy_4k[1] - oy) * ry))


def crop_to_global(
    xy_crop: Tuple[float, float],
    offset: Point,
    crop_resize_scale: Scale = (1.0, 1.0),
) -> Point:
    """Inverse map L -> G: ``(o_x + x_l / r_x, o_y + y_l / r_y)``.

    This is the final HLM coordinate recovery: a Focus-phase point expressed in
    the (possibly resized) crop frame is mapped back to the global 4K frame.
    """
    ox, oy = offset
    rx, ry = crop_resize_scale
    if rx == 0 or ry == 0:
        raise ValueError(f"crop_resize_scale components must be non-zero, got {crop_resize_scale}")
    return (_round(ox + xy_crop[0] / rx), _round(oy + xy_crop[1] / ry))


def crop_bbox_to_global(
    bbox_crop: BBox,
    offset: Point,
    crop_resize_scale: Scale = (1.0, 1.0),
) -> BBox:
    """Map a ``[x1, y1, x2, y2]`` bbox from the crop frame to the global frame."""
    x1, y1 = crop_to_global((bbox_crop[0], bbox_crop[1]), offset, crop_resize_scale)
    x2, y2 = crop_to_global((bbox_crop[2], bbox_crop[3]), offset, crop_resize_scale)
    return (x1, y1, x2, y2)


def coarse_bbox_to_global(bbox_2k: BBox, scale: Scale) -> BBox:
    """Map a ``[x1, y1, x2, y2]`` bbox from the coarse (2K) frame to global (4K)."""
    x1, y1 = coarse_to_global((bbox_2k[0], bbox_2k[1]), scale)
    x2, y2 = coarse_to_global((bbox_2k[2], bbox_2k[3]), scale)
    return (x1, y1, x2, y2)
