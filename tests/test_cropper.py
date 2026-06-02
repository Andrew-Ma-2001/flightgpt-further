"""Tests for navgym.hlm.cropper. Runs under pytest or as a plain script."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402

from navgym.hlm.cropper import (  # noqa: E402
    CropParams,
    compute_crop_size,
    crop_window,
    crop,
)
from navgym.hlm.coordinate_mapper import crop_to_global  # noqa: E402


def _img(w, h, c=3):
    # encode global x,y into pixel values so we can verify the crop content
    img = np.zeros((h, w, c), dtype=np.uint8)
    return img


# --------------------------------------------------------------------------- #
# compute_crop_size
# --------------------------------------------------------------------------- #
def test_fixed_size():
    p = CropParams(adaptive_crop=False, fixed_crop_size=1024)
    assert compute_crop_size((0, 0, 500, 500), p) == (1024, 1024)


def test_fixed_when_no_bbox():
    p = CropParams(adaptive_crop=True, fixed_crop_size=800)
    assert compute_crop_size(None, p) == (800, 800)


def test_adaptive_size():
    p = CropParams(adaptive_crop=True, crop_alpha=3.0, crop_min=512, crop_max=2048)
    # bbox 300x400 -> diag 500 -> 3*500 = 1500
    assert compute_crop_size((0, 0, 300, 400), p) == (1500, 1500)


def test_adaptive_clamped_min():
    p = CropParams(adaptive_crop=True, crop_alpha=3.0, crop_min=512, crop_max=2048)
    # tiny bbox -> below min -> clamp to 512
    assert compute_crop_size((10, 10, 20, 20), p) == (512, 512)


def test_adaptive_clamped_max():
    p = CropParams(adaptive_crop=True, crop_alpha=3.0, crop_min=512, crop_max=2048)
    # large bbox diag -> above max -> clamp to 2048
    assert compute_crop_size((0, 0, 2000, 2000), p) == (2048, 2048)


def test_adaptive_degenerate_bbox():
    p = CropParams(adaptive_crop=True, crop_min=512, crop_max=2048)
    assert compute_crop_size((100, 100, 100, 100), p) == (512, 512)


# --------------------------------------------------------------------------- #
# crop_window: centering, boundary sliding, clamping
# --------------------------------------------------------------------------- #
def test_crop_centered_interior():
    img = _img(4000, 3000)
    crop_img, offset = crop_window(img, (2000, 1500), (1024, 1024))
    assert crop_img.shape == (1024, 1024, 3)
    assert offset == (2000 - 512, 1500 - 512)


def test_crop_slides_at_left_top_corner():
    img = _img(4000, 3000)
    crop_img, offset = crop_window(img, (10, 10), (1024, 1024))
    assert crop_img.shape == (1024, 1024, 3)
    assert offset == (0, 0)  # slid fully inside


def test_crop_slides_at_right_bottom_corner():
    img = _img(4000, 3000)
    crop_img, offset = crop_window(img, (3999, 2999), (1024, 1024))
    assert crop_img.shape == (1024, 1024, 3)
    assert offset == (4000 - 1024, 3000 - 1024)


def test_crop_size_larger_than_image():
    img = _img(800, 600)
    crop_img, offset = crop_window(img, (400, 300), (1024, 1024))
    assert crop_img.shape == (600, 800, 3)  # clamped to image
    assert offset == (0, 0)


def test_crop_center_out_of_bounds_clamped():
    img = _img(4000, 3000)
    crop_img, offset = crop_window(img, (-100, 99999), (512, 512))
    assert crop_img.shape == (512, 512, 3)
    assert offset == (0, 3000 - 512)


def test_crop_grayscale_2d():
    img = np.zeros((600, 800), dtype=np.uint8)
    crop_img, offset = crop_window(img, (400, 300), (256, 256))
    assert crop_img.shape == (256, 256)


def test_crop_returns_copy_not_view():
    img = _img(1000, 1000)
    crop_img, _ = crop_window(img, (500, 500), (100, 100))
    crop_img[0, 0, 0] = 200
    assert img[450, 450, 0] == 0  # original untouched


# --------------------------------------------------------------------------- #
# crop() full path + content correctness
# --------------------------------------------------------------------------- #
def test_crop_full_returns_actual_size():
    img = _img(800, 600)
    p = CropParams(adaptive_crop=False, fixed_crop_size=1024)
    crop_img, offset, size = crop(img, (400, 300), None, p)
    assert size == (800, 600)
    assert offset == (0, 0)


def test_crop_content_matches_global_coords():
    # Build an image whose pixel encodes (x % 256) in channel 0 and (y % 256) in
    # channel 1, then verify a crop pixel maps back to the right global coord.
    w, h = 2000, 1500
    xs = (np.arange(w) % 256).astype(np.uint8)
    ys = (np.arange(h) % 256).astype(np.uint8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = np.broadcast_to(xs, (h, w))
    img[:, :, 1] = np.broadcast_to(ys[:, None], (h, w))

    p = CropParams(adaptive_crop=False, fixed_crop_size=512)
    crop_img, offset, size = crop(img, (1000, 750), None, p)

    # pick a local point, map to global, compare encoded value
    local = (100, 200)
    gx, gy = crop_to_global(local, offset, (1.0, 1.0))
    assert crop_img[local[1], local[0], 0] == gx % 256
    assert crop_img[local[1], local[0], 1] == gy % 256


def test_crop_invalid_image_raises():
    try:
        crop_window(np.zeros((5,), dtype=np.uint8), (0, 0), (2, 2))
    except ValueError:
        return
    raise AssertionError("expected ValueError for 1D image")


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(funcs)}/{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
