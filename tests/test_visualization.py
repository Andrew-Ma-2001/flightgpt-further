"""Tests for navgym.hlm.visualization. Runs under pytest or as a plain script."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from navgym.hlm import HLMResult  # noqa: E402
from navgym.hlm.visualization import visualize_hlm  # noqa: E402


def _img(path, w=2000, h=1500):
    cv2.imwrite(path, np.zeros((h, w, 3), dtype=np.uint8))
    return path


def _result(used_fine=True, bbox=(900, 700, 1100, 800)):
    return HLMResult(
        coarse_xy_4k=(1000, 750),
        landmark_bbox_4k=bbox,
        crop_offset=(744, 494),
        crop_size=(512, 512),
        fine_xy_4k=(1010, 760),
        final_xy=(1010, 760),
        coarse_xy_2k=(512, 384),
        scale_4k_to_2k=(0.512, 0.512),
        used_fine=used_fine,
        raw_coarse_response="", raw_fine_response="",
    )


def test_viz_writes_file_with_gt():
    with tempfile.TemporaryDirectory() as tmp:
        img = _img(os.path.join(tmp, "m.jpg"))
        out = os.path.join(tmp, "viz", "out.jpg")
        path = visualize_hlm(img, _result(), gt_xy_4k=(1050, 800), ne_px=64.0, save_path=out)
        assert os.path.exists(path)
        rendered = cv2.imread(path)
        assert rendered is not None and rendered.shape == (1500, 2000, 3)


def test_viz_without_gt():
    with tempfile.TemporaryDirectory() as tmp:
        img = _img(os.path.join(tmp, "m.jpg"))
        out = os.path.join(tmp, "out.jpg")
        visualize_hlm(img, _result(), save_path=out)
        assert os.path.exists(out)


def test_viz_no_crop_region():
    # coarse-only result (no fine): crop_size (0,0) must not draw a rectangle
    with tempfile.TemporaryDirectory() as tmp:
        img = _img(os.path.join(tmp, "m.jpg"))
        res = _result(used_fine=False)
        res.crop_size = (0, 0)
        out = os.path.join(tmp, "out.jpg")
        visualize_hlm(img, res, gt_xy_4k=(1050, 800), ne_px=10.0, save_path=out)
        assert os.path.exists(out)


def test_viz_missing_image_raises():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            visualize_hlm(os.path.join(tmp, "nope.jpg"), _result(),
                          save_path=os.path.join(tmp, "o.jpg"))
        except FileNotFoundError:
            return
    raise AssertionError("expected FileNotFoundError for missing image")


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(funcs)}/{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
