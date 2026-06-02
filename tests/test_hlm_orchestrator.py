"""End-to-end smoke tests for the HLM orchestrator using a stub VLM agent.

No vLLM server is required: the stub returns canned responses keyed on whether
the prompt is the Glance or the Focus prompt. Runs under pytest or as a script.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from navgym.hlm import HierarchicalLocalizationModule, HLMConfig  # noqa: E402


class StubAgent:
    """Returns canned responses; distinguishes phases via the prompt text."""

    target_description = "a stub target"

    def __init__(self, glance_response, fine_response):
        self.glance_response = glance_response
        self.fine_response = fine_response
        self.calls = []

    def act_on_image(self, image_path, prompt, system_prompt=None):
        self.calls.append((image_path, prompt))
        is_focus = "ZOOMED-IN" in prompt
        return self.fine_response if is_focus else self.glance_response


def _make_image(path, w=4000, h=3000):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def _glance(x, y, bbox=None):
    b = f' {{"landmark_bbox": {list(bbox)}}}' if bbox else ""
    return f'<think>{b}</think><answer>{{"target_location": [{x}, {y}]}}</answer>'


def _fine(x, y):
    return f'<think></think><answer>{{"target_location": [{x}, {y}]}}</answer>'


def _cfg(tmp, **kw):
    base = dict(adaptive_crop=False, fixed_crop_size=1024, model_input_size=None, work_dir=tmp)
    base.update(kw)
    return HLMConfig(**base)


def test_full_pipeline_fixed_crop():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        agent = StubAgent(_glance(1024, 768), _fine(512, 512))
        hlm = HierarchicalLocalizationModule(agent, _cfg(tmp))
        res = hlm.forward(img, "find it", start_px=[100, 100])

        # coarse [1024,768] @ scale 0.512 -> (2000, 1500)
        assert res.coarse_xy_4k == (2000, 1500)
        assert res.crop_size == (1024, 1024)
        assert res.crop_offset == (2000 - 512, 1500 - 512)
        # fine at crop center (512,512) -> back to (2000,1500)
        assert res.final_xy == (2000, 1500)
        assert res.used_fine is True
        assert len(agent.calls) == 2  # one glance + one focus


def test_fine_offset_recovers_global():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        # fine predicts near top-left of crop -> should equal crop offset + local.
        # (avoid local (0,0): that collides with the repo-wide [0,0] failure sentinel)
        agent = StubAgent(_glance(1024, 768), _fine(3, 7))
        hlm = HierarchicalLocalizationModule(agent, _cfg(tmp))
        res = hlm.forward(img, "find it", start_px=[0, 0])
        ox, oy = res.crop_offset
        assert res.final_xy == (ox + 3, oy + 7)
        assert res.used_fine is True


def test_glance_failure_skips_fine():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        agent = StubAgent("no coordinates at all", _fine(512, 512))
        hlm = HierarchicalLocalizationModule(agent, _cfg(tmp))
        res = hlm.forward(img, "find it", start_px=[0, 0])
        assert res.final_xy == (0, 0)
        assert res.used_fine is False
        assert len(agent.calls) == 1  # focus never called


def test_fine_failure_falls_back_to_coarse():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        agent = StubAgent(_glance(1024, 768), "garbage no point")
        hlm = HierarchicalLocalizationModule(agent, _cfg(tmp))
        res = hlm.forward(img, "find it", start_px=[0, 0])
        assert res.final_xy == res.coarse_xy_4k == (2000, 1500)
        assert res.used_fine is False


def test_enable_fine_false():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        agent = StubAgent(_glance(1024, 768), _fine(0, 0))
        hlm = HierarchicalLocalizationModule(agent, _cfg(tmp, enable_fine=False))
        res = hlm.forward(img, "find it", start_px=[0, 0])
        assert res.final_xy == res.coarse_xy_4k == (2000, 1500)
        assert res.used_fine is False
        assert len(agent.calls) == 1


def test_model_input_resize_scales_back():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        # crop is 1024; resize to 512 => r=0.5; fine at (256,256) center -> (2000,1500)
        agent = StubAgent(_glance(1024, 768), _fine(256, 256))
        hlm = HierarchicalLocalizationModule(agent, _cfg(tmp, model_input_size=512))
        res = hlm.forward(img, "find it", start_px=[0, 0])
        assert res.final_xy == (2000, 1500)
        # crop image actually sent was 512x512
        crop = cv2.imread(res.crop_image_path)
        assert crop.shape[0] == 512 and crop.shape[1] == 512


def test_adaptive_crop_uses_bbox():
    with tempfile.TemporaryDirectory() as tmp:
        img = _make_image(os.path.join(tmp, "map.jpg"))
        # bbox in 2K; back-projected diag drives the crop size
        agent = StubAgent(_glance(1024, 768, bbox=[1000, 700, 1100, 800]), _fine(10, 10))
        hlm = HierarchicalLocalizationModule(
            agent, _cfg(tmp, adaptive_crop=True, crop_alpha=3.0, crop_min=512, crop_max=2048)
        )
        res = hlm.forward(img, "find it", start_px=[0, 0])
        assert res.landmark_bbox_4k is not None
        # crop is square and within [crop_min, crop_max]
        assert res.crop_size[0] == res.crop_size[1]
        assert 512 <= res.crop_size[0] <= 2048
        assert res.used_fine is True


def test_invalid_agent_rejected():
    try:
        HierarchicalLocalizationModule(object(), HLMConfig())
    except TypeError:
        return
    raise AssertionError("expected TypeError for agent without act_on_image")


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(funcs)}/{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
