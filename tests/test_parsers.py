"""Tests for navgym.hlm.parsers. Runs under pytest or as a plain script."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from navgym.hlm.parsers import (  # noqa: E402
    parse_location,
    parse_bbox,
    location_parse_failed,
    bbox_parse_failed,
    LOCATION_SENTINEL,
    BBOX_SENTINEL,
)

_SAMPLE = (
    '<think> the landmark region is here '
    '{"landmark_bbox": [100, 50, 200, 150]} </think>'
    '<answer>{"target_location": [123, 456]}</answer>'
)


def test_parse_location_basic():
    assert parse_location(_SAMPLE) == [123, 456]


def test_parse_bbox_basic():
    assert parse_bbox(_SAMPLE, "landmark_bbox") == [100, 50, 200, 150]


def test_parse_location_no_space():
    assert parse_location('{"target_location":[7,8]}') == [7, 8]


def test_parse_location_extra_space():
    assert parse_location('{"target_location"  :  [ 7 ,  8 ]}') == [7, 8]


def test_parse_location_missing_returns_sentinel():
    assert parse_location("no coordinates here") == LOCATION_SENTINEL
    assert parse_location("") == LOCATION_SENTINEL


def test_parse_bbox_missing_returns_sentinel():
    assert parse_bbox("nothing") == BBOX_SENTINEL


def test_parse_location_none_safe():
    assert parse_location(None) == LOCATION_SENTINEL


def test_parse_bbox_custom_key():
    text = '{"pred_box": [1, 2, 3, 4]}'
    assert parse_bbox(text, "pred_box") == [1, 2, 3, 4]
    assert parse_bbox(text, "landmark_bbox") == BBOX_SENTINEL


def test_parse_location_first_match():
    text = '{"target_location": [1, 1]} ... {"target_location": [9, 9]}'
    assert parse_location(text) == [1, 1]


def test_location_parse_failed():
    assert location_parse_failed([0, 0]) is True
    assert location_parse_failed([1, 0]) is False


def test_bbox_parse_failed():
    assert bbox_parse_failed([0, 0, 0, 0]) is True
    assert bbox_parse_failed([0, 0, 1, 1]) is False


def test_sentinels_are_copies():
    # mutating a returned sentinel must not corrupt the module constant
    loc = parse_location("none")
    loc.append(99)
    assert LOCATION_SENTINEL == [0, 0]


def _run_all():
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in funcs:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(funcs)}/{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
