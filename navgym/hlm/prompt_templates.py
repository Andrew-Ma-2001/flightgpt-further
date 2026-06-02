"""Prompt templates for the two HLM phases.

* Glance (Stage 1): re-exports the existing ``get_prompt`` from
  ``navgym.agents.CityNavAgent`` verbatim. The whole repo (eval parsers and GRPO
  reward regexes) depends on that prompt's output schema, so it must not change.
* Focus (Stage 2): ``get_fine_prompt`` instructs the model that it is looking at
  a zoomed-in crop and must answer in the crop's own pixel frame, while keeping
  the exact same ``<think>``/``<answer>`` + ``landmark_bbox`` / ``target_location``
  output protocol so the shared parsers keep working.
"""

from __future__ import annotations

# Re-export the Glance-phase prompt unchanged.
from navgym.agents.CityNavAgent import get_prompt as get_glance_prompt  # noqa: F401


def get_fine_prompt(instruction: str, crop_w: int, crop_h: int) -> str:
    """Build the Focus-phase prompt for a high-resolution crop.

    Args:
        instruction: the natural-language target description.
        crop_w: width (px) of the crop image actually sent to the model.
        crop_h: height (px) of the crop image actually sent to the model.

    Returns:
        A prompt string whose answer must be a ``target_location`` expressed in
        THIS crop's pixel coordinates (top-left = (0, 0)).
    """
    return f"""
[Mission Objective]
This is a HIGH-RESOLUTION ZOOMED-IN crop of the region you already identified as
most likely containing the target. The search area has been narrowed for you.
Your job is to give the FINAL, precise pixel location of the target inside THIS image only.

[Details of the Target]
{instruction}

[Environmental Perception]
- This image is a {crop_w} x {crop_h} pixel close-up; it is NOT the full map.
- Coordinates MUST be given relative to THIS cropped image: the top-left corner is
  (0, 0) and the bottom-right corner is ({crop_w - 1}, {crop_h - 1}).
- Street-related landmark regions are marked with red masks; the target lies near one.

[Operational Guidance]
- Switch to fine-grained inspection: use precise landmark edges, textures, and small
  structures that are now clearly visible at this zoom level.
- Re-identify the landmark region, then localize the target relative to it.

[Output Format Specification]
- Present your reasoning within `<think>` and `</think>` tags, including the landmark
  region in the format:
  `{{"landmark_bbox": [x1, y1, x2, y2]}}`
- Then provide your final answer within `<answer>` and `</answer>` tags as:
  `{{"target_location": [x, y]}}`
"""
