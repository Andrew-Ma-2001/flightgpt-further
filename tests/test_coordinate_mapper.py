"""Tests for navgym.hlm.coordinate_mapper.

Runs under pytest *or* as a plain script (``python tests/test_coordinate_mapper.py``)
so it has no third-party dependencies beyond the standard library.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from navgym.hlm.coordinate_mapper import (  # noqa: E402
    clamp_point,
    compute_scale,
    coarse_to_global,
    global_to_coarse,
    crop_to_global,
    global_to_crop,
    crop_bbox_to_global,
    coarse_bbox_to_global,
)


# --------------------------------------------------------------------------- #
# compute_scale
# --------------------------------------------------------------------------- #
def test_compute_scale_basic():
    sx, sy = compute_scale((4000, 3000), (2048, 1536))
    assert abs(sx - 0.512) < 1e-9
    assert abs(sy - 0.512) < 1e-9


def test_compute_scale_square():
    assert compute_scale((4000, 4000), (2048, 2048)) == (0.512, 0.512)


# --------------------------------------------------------------------------- #
# Coarse <-> Global round trips
# --------------------------------------------------------------------------- #
def test_coarse_global_round_trip():
    scale = compute_scale((4000, 3000), (2048, 1536))
    for p in [(0, 0), (4000 - 1, 3000 - 1), (2000, 1500), (123, 2876)]:
        back = coarse_to_global(global_to_coarse(p, scale), scale)
        assert abs(back[0] - p[0]) <= 1
        assert abs(back[1] - p[1]) <= 1


def test_global_to_coarse_values():
    scale = compute_scale((4000, 3000), (2048, 1536))
    assert global_to_coarse((2000, 1500), scale) == (1024, 768)


def test_coarse_to_global_values():
    scale = compute_scale((4000, 3000), (2048, 1536))
    assert coarse_to_global((1024, 768), scale) == (2000, 1500)


def test_coarse_to_global_zero_scale_raises():
    try:
        coarse_to_global((10, 10), (0.0, 0.5))
    except ValueError:
        return
    raise AssertionError("expected ValueError for zero scale")


# --------------------------------------------------------------------------- #
# Crop <-> Global round trips
# --------------------------------------------------------------------------- #
def test_crop_global_round_trip_no_resize():
    offset = (500, 600)
    for p in [(500, 600), (1000, 1200), (1499, 1599)]:
        crop_p = global_to_crop(p, offset, (1.0, 1.0))
        back = crop_to_global(crop_p, offset, (1.0, 1.0))
        assert back == p


def test_crop_global_round_trip_with_resize():
    # crop window 1024x1024 from 4K resized to model input 512x512 -> r = 0.5
    offset = (800, 400)
    rscale = (512 / 1024.0, 512 / 1024.0)
    for p in [(800, 400), (1300, 900), (1823, 1423)]:
        crop_p = global_to_crop(p, offset, rscale)
        back = crop_to_global(crop_p, offset, rscale)
        assert abs(back[0] - p[0]) <= 2  # rounding on a 0.5 scale
        assert abs(back[1] - p[1]) <= 2


def test_crop_to_global_known_value():
    # local point (256, 256) in a 512-input crop (resized from 1024) at offset (800,400)
    rscale = (0.5, 0.5)
    assert crop_to_global((256, 256), (800, 400), rscale) == (800 + 512, 400 + 512)


def test_crop_offset_only():
    assert crop_to_global((10, 20), (100, 200)) == (110, 220)


def test_crop_to_global_zero_resize_raises():
    try:
        crop_to_global((10, 10), (0, 0), (0.0, 1.0))
    except ValueError:
        return
    raise AssertionError("expected ValueError for zero resize scale")


# --------------------------------------------------------------------------- #
# bbox helpers
# --------------------------------------------------------------------------- #
def test_crop_bbox_to_global():
    bbox = crop_bbox_to_global((10, 20, 30, 40), (100, 200), (1.0, 1.0))
    assert bbox == (110, 220, 130, 240)


def test_coarse_bbox_to_global():
    scale = compute_scale((4000, 4000), (2000, 2000))  # s = 0.5
    bbox = coarse_bbox_to_global((100, 100, 200, 200), scale)
    assert bbox == (200, 200, 400, 400)


# --------------------------------------------------------------------------- #
# clamp_point
# --------------------------------------------------------------------------- #
def test_clamp_inside():
    assert clamp_point((50.4, 60.6), 100, 100) == (50, 61)


def test_clamp_negative():
    assert clamp_point((-5, -10), 100, 100) == (0, 0)


def test_clamp_overflow():
    assert clamp_point((150, 200), 100, 80) == (99, 79)


def test_clamp_invalid_dims_raises():
    try:
        clamp_point((1, 1), 0, 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-positive dims")


# --------------------------------------------------------------------------- #
# full HLM composition: coarse point -> crop center -> back to 4K
# --------------------------------------------------------------------------- #
def test_full_inverse_composition():
    # 4K = 4000x3000, 2K = 2048x1536
    scale = compute_scale((4000, 3000), (2048, 1536))
    # model predicts coarse point (1024, 768) in 2K
    center_4k = coarse_to_global((1024, 768), scale)
    assert center_4k == (2000, 1500)
    # crop 1024x1024 centered, clamped well inside the image
    crop_w = crop_h = 1024
    ox = max(0, min(center_4k[0] - crop_w // 2, 4000 - crop_w))
    oy = max(0, min(center_4k[1] - crop_h // 2, 3000 - crop_h))
    offset = (ox, oy)
    # model predicts fine point at crop center (no resize)
    fine_4k = crop_to_global((crop_w // 2, crop_h // 2), offset, (1.0, 1.0))
    # fine point should land back at the original coarse center
    assert fine_4k == center_4k


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in funcs:
        fn()
        passed += 1
        print(f"  PASS {fn.__name__}")
    print(f"\n{passed}/{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
