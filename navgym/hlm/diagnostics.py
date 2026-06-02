"""A/B diagnostics: Glance-only vs full HLM localization accuracy.

For every prediction we compare the back-projected coarse (Glance) point against
the refined (Focus) point on two scales:

* Pixel NE  - Euclidean distance to the GT target in the 4K pixel frame.
* World NE  - Euclidean distance in world meters, using the same linear
  pixel->meter mapping as ``eval.py:compute_pose`` (the z/depth refinement there
  does not affect the XY distance that NE is computed from, so it is omitted).

The summary reports mean NE and success-rate (NE <= ``success_threshold_m``) for
both variants plus the HLM improvement, so one can verify HLM <= Glance-only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple


def pixel_distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def pixel_to_world_xy(pred_px, start_px, start_world_xy, px_real_size) -> Tuple[float, float]:
    """Replicate eval.py:compute_pose's linear pixel->world XY mapping.

    world_x = dx * m_x + start_x ;  world_y = start_y - dy * m_y
    """
    dx = pred_px[0] - start_px[0]
    dy = pred_px[1] - start_px[1]
    mx, my = px_real_size
    return (dx * mx + start_world_xy[0], start_world_xy[1] - dy * my)


@dataclass
class HLMDiagnostics:
    """Accumulates per-prediction Glance-vs-HLM errors and summarizes them."""

    success_threshold_m: float = 20.0
    glance_px: List[float] = field(default_factory=list)
    hlm_px: List[float] = field(default_factory=list)
    glance_world: List[float] = field(default_factory=list)
    hlm_world: List[float] = field(default_factory=list)
    used_fine_count: int = 0
    total: int = 0

    def record(
        self,
        coarse_xy_4k,
        final_xy_4k,
        target_px,
        start_px,
        start_world_xy,
        target_world_xy,
        px_real_size,
        used_fine: bool,
    ) -> None:
        self.total += 1
        if used_fine:
            self.used_fine_count += 1

        self.glance_px.append(pixel_distance(coarse_xy_4k, target_px))
        self.hlm_px.append(pixel_distance(final_xy_4k, target_px))

        gw = pixel_to_world_xy(coarse_xy_4k, start_px, start_world_xy, px_real_size)
        hw = pixel_to_world_xy(final_xy_4k, start_px, start_world_xy, px_real_size)
        self.glance_world.append(pixel_distance(gw, target_world_xy))
        self.hlm_world.append(pixel_distance(hw, target_world_xy))

    def record_from_navgym(self, navGym, hlm_result, true_start_px, true_target_px) -> None:
        """Convenience wrapper pulling the needed quantities from a NavGym."""
        start_world = (navGym.episode.start_pose.x, navGym.episode.start_pose.y)
        target_world = (navGym.episode.target_position.x, navGym.episode.target_position.y)
        self.record(
            coarse_xy_4k=hlm_result.coarse_xy_4k,
            final_xy_4k=hlm_result.final_xy,
            target_px=true_target_px,
            start_px=true_start_px,
            start_world_xy=start_world,
            target_world_xy=target_world,
            px_real_size=navGym.px_real_size,
            used_fine=hlm_result.used_fine,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else float("nan")

    def _success_rate(self, xs: List[float]) -> float:
        if not xs:
            return float("nan")
        return float(sum(1 for x in xs if x <= self.success_threshold_m) / len(xs))

    def summary(self) -> dict:
        return {
            "num_predictions": self.total,
            "used_fine_count": self.used_fine_count,
            "used_fine_ratio": (self.used_fine_count / self.total) if self.total else float("nan"),
            "success_threshold_m": self.success_threshold_m,
            "glance": {
                "mean_pixel_ne": self._mean(self.glance_px),
                "mean_world_ne": self._mean(self.glance_world),
                "success_rate": self._success_rate(self.glance_world),
            },
            "hlm": {
                "mean_pixel_ne": self._mean(self.hlm_px),
                "mean_world_ne": self._mean(self.hlm_world),
                "success_rate": self._success_rate(self.hlm_world),
            },
            "improvement": {
                "mean_pixel_ne": self._mean(self.glance_px) - self._mean(self.hlm_px),
                "mean_world_ne": self._mean(self.glance_world) - self._mean(self.hlm_world),
                "success_rate": self._success_rate(self.hlm_world) - self._success_rate(self.glance_world),
            },
        }

    def print_summary(self, label: str = "") -> None:
        if self.total == 0:
            print(f"[HLM A/B {label}] no predictions recorded")
            return
        s = self.summary()
        g, h, imp = s["glance"], s["hlm"], s["improvement"]
        print("\n" + "-" * 60)
        print(f"[HLM A/B diagnostics{(' ' + label) if label else ''}] "
              f"n={s['num_predictions']} used_fine={s['used_fine_ratio']*100:.1f}%")
        print(f"  {'metric':<16}{'glance':>12}{'hlm':>12}{'improve':>12}")
        print(f"  {'pixel_ne(mean)':<16}{g['mean_pixel_ne']:>12.2f}{h['mean_pixel_ne']:>12.2f}{imp['mean_pixel_ne']:>12.2f}")
        print(f"  {'world_ne(mean)':<16}{g['mean_world_ne']:>12.2f}{h['mean_world_ne']:>12.2f}{imp['mean_world_ne']:>12.2f}")
        print(f"  {'success_rate':<16}{g['success_rate']:>12.3f}{h['success_rate']:>12.3f}{imp['success_rate']:>12.3f}")
        print("-" * 60)

    def save(self, path: str, label: str = "") -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.summary()
        payload["label"] = label
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
