"""Tests for navgym.hlm.diagnostics. Runs under pytest or as a plain script."""

import os
import sys
import json
import math
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from navgym.hlm.diagnostics import (  # noqa: E402
    HLMDiagnostics,
    pixel_distance,
    pixel_to_world_xy,
)


def test_pixel_distance():
    assert pixel_distance((0, 0), (3, 4)) == 5.0


def test_pixel_to_world_xy():
    # 1px = 0.5m; pred 10px right, 4px down from start at world (100, 200)
    w = pixel_to_world_xy((110, 204), (100, 200), (100.0, 200.0), (0.5, 0.5))
    assert abs(w[0] - (100.0 + 10 * 0.5)) < 1e-9   # x increases
    assert abs(w[1] - (200.0 - 4 * 0.5)) < 1e-9     # y decreases (image y is down)


def test_hlm_better_than_glance():
    d = HLMDiagnostics(success_threshold_m=20.0)
    target_px = (1000, 1000)
    start_px = (0, 0)
    start_world = (0.0, 0.0)
    target_world = pixel_to_world_xy(target_px, start_px, start_world, (0.1, 0.1))
    # coarse is 100px off; fine is 5px off -> HLM clearly better
    d.record(
        coarse_xy_4k=(1100, 1000), final_xy_4k=(1005, 1000),
        target_px=target_px, start_px=start_px,
        start_world_xy=start_world, target_world_xy=target_world,
        px_real_size=(0.1, 0.1), used_fine=True,
    )
    s = d.summary()
    assert s["glance"]["mean_pixel_ne"] == 100.0
    assert s["hlm"]["mean_pixel_ne"] == 5.0
    assert s["improvement"]["mean_pixel_ne"] == 95.0
    assert s["improvement"]["mean_world_ne"] > 0  # HLM closer in world too
    assert s["used_fine_ratio"] == 1.0


def test_success_rate_threshold():
    d = HLMDiagnostics(success_threshold_m=20.0)
    # world scale 0.1 m/px: glance 100px=10m (success), hlm 5px=0.5m (success)
    target_px = (1000, 1000)
    d.record((900, 1000), (995, 1000), target_px, (0, 0), (0.0, 0.0),
             pixel_to_world_xy(target_px, (0, 0), (0.0, 0.0), (0.1, 0.1)),
             (0.1, 0.1), used_fine=True)
    # a far miss: 300px = 30m (failure for both)
    d.record((1300, 1000), (1300, 1000), target_px, (0, 0), (0.0, 0.0),
             pixel_to_world_xy(target_px, (0, 0), (0.0, 0.0), (0.1, 0.1)),
             (0.1, 0.1), used_fine=False)
    s = d.summary()
    assert s["num_predictions"] == 2
    assert s["used_fine_ratio"] == 0.5
    assert s["hlm"]["success_rate"] == 0.5  # one within 20m, one not


def test_empty_summary_safe():
    d = HLMDiagnostics()
    s = d.summary()
    assert s["num_predictions"] == 0
    assert math.isnan(s["glance"]["mean_pixel_ne"])
    d.print_summary("empty")  # must not raise


def test_save_roundtrip():
    d = HLMDiagnostics()
    d.record((10, 0), (1, 0), (0, 0), (0, 0), (0.0, 0.0), (0.0, 0.0), (1.0, 1.0), True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sub", "ab.json")
        d.save(path, label="easy")
        assert os.path.exists(path)
        with open(path) as f:
            payload = json.load(f)
        assert payload["label"] == "easy"
        assert payload["num_predictions"] == 1


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(funcs)}/{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
