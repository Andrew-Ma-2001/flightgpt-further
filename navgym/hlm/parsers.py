"""Output parsers for VLM responses in the HLM pipeline.

Centralizes the regexes used to extract ``target_location`` and ``landmark_bbox``
from the model's ``<think>...</think><answer>...</answer>`` output. These mirror
the parsers in ``eval.py`` (``parse_bbox`` / ``parse_location``) and the reward
regexes in ``grpo_jsonl_citynav.py``, so both the Glance and Focus phases share a
single source of truth.

Whitespace after commas is tolerated (``\\s*``) to match the GRPO reward parser,
which is strictly more permissive than ``eval.py`` and never less. Failure
sentinels are preserved exactly: ``[0, 0]`` for a point and ``[0, 0, 0, 0]`` for a
bbox, so existing downstream code that special-cases ``[0, 0]`` keeps working.
"""

from __future__ import annotations

import re
from typing import List

LOCATION_SENTINEL: List[int] = [0, 0]
BBOX_SENTINEL: List[int] = [0, 0, 0, 0]

_LOCATION_RE = re.compile(r'"target_location"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]')


def _bbox_re(key: str) -> "re.Pattern[str]":
    return re.compile(
        rf'"{re.escape(key)}"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    )


def parse_location(result_str: str) -> List[int]:
    """Extract ``"target_location": [x, y]`` -> ``[x, y]`` (ints).

    Returns ``[0, 0]`` if no match is found (the repo-wide failure sentinel).
    """
    match = _LOCATION_RE.search(result_str or "")
    return [int(match.group(1)), int(match.group(2))] if match else list(LOCATION_SENTINEL)


def parse_bbox(result_str: str, key: str = "landmark_bbox") -> List[int]:
    """Extract ``"<key>": [x1, y1, x2, y2]`` -> ``[x1, y1, x2, y2]`` (ints).

    Returns ``[0, 0, 0, 0]`` if no match is found.
    """
    match = _bbox_re(key).search(result_str or "")
    return [int(g) for g in match.groups()] if match else list(BBOX_SENTINEL)


def location_parse_failed(location: List[int]) -> bool:
    """True if a parsed location is the failure sentinel ``[0, 0]``.

    The HLM orchestrator uses this to skip the Focus phase (and fall back to the
    coarse prediction) when the Glance phase produced no usable point.
    """
    return list(location) == LOCATION_SENTINEL


def bbox_parse_failed(bbox: List[int]) -> bool:
    """True if a parsed bbox is the failure sentinel ``[0, 0, 0, 0]``."""
    return list(bbox) == BBOX_SENTINEL
