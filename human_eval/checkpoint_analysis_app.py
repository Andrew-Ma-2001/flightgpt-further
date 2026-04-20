import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageDraw

# Ensure project-root imports work when launched via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# HETT checkpoints may pickle classes from `multiagent.*`,
# whose package root is `<project>/HETT`.
HETT_ROOT = PROJECT_ROOT / "HETT"
if str(HETT_ROOT) not in sys.path:
    # Keep project root packages first (e.g. gsamllavanav), and only add HETT
    # as a fallback for unpickling `multiagent.*` classes from HETT checkpoints.
    sys.path.append(str(HETT_ROOT))

from gsamllavanav.defaultpaths import MTURK_TRAJECTORY_DIR
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym


SPLITS = ["easy", "medium", "hard"]
NEW_DATA_DIR = "/home/yjy/flightgpt/FlightGPT/refine_citynav/processed_citynav"
SR_THRESHOLDS = [5, 10, 15, 20, 25, 30]
SR_COMPARE_THRESHOLDS = [5, 10, 15, 20, 25, 30, 35, 40]

# Initial heading bins: [-180, 180], 15° wide (24 bins). -180° and +180° share one bin via wrapping.
HEADING_BIN_WIDTH_DEG = 15.0
HEADING_BIN_EDGES = np.arange(-180, 181, HEADING_BIN_WIDTH_DEG)
HEADING_BIN_CENTERS = (HEADING_BIN_EDGES[:-1] + HEADING_BIN_EDGES[1:]) / 2.0


def _wrap_heading_deg(deg: float) -> float:
    return float(((deg + 180.0) % 360.0) - 180.0)


def _initial_heading_deg_from_episode(ep) -> float:
    rad = float(ep.start_pose.yaw)
    deg = float(np.rad2deg(rad))
    return _wrap_heading_deg(deg)


def _heading_bin_index(deg: float) -> int:
    d = _wrap_heading_deg(float(deg))
    if d >= 180.0 - 1e-9:
        d = -180.0
    idx = int(np.floor((d + 180.0) / HEADING_BIN_WIDTH_DEG))
    return int(min(max(idx, 0), len(HEADING_BIN_CENTERS) - 1))


def _heading_bin_stats(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = len(HEADING_BIN_CENTERS)
    counts = np.zeros(n_bins, dtype=int)
    ne_sum = np.zeros(n_bins, dtype=float)
    if df.empty or "initial_heading_deg" not in df.columns:
        return counts, np.full(n_bins, np.nan)
    for _, row in df.iterrows():
        bi = _heading_bin_index(row["initial_heading_deg"])
        counts[bi] += 1
        ne_sum[bi] += float(row["ne"])
    mean_ne = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        if counts[i] > 0:
            mean_ne[i] = ne_sum[i] / counts[i]
    return counts, mean_ne


def _draw_initial_heading_polar(df: pd.DataFrame, title: str):
    """Polar bar + closed polyline of mean NE per heading bin (15°)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(projection="polar"))
    if df.empty or "initial_heading_deg" not in df.columns:
        ax.set_title(title, pad=12)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        fig.tight_layout()
        return fig

    counts, mean_ne = _heading_bin_stats(df)
    theta = np.deg2rad(HEADING_BIN_CENTERS)
    width = np.deg2rad(HEADING_BIN_WIDTH_DEG * 0.92)

    r_max = 1.0
    valid_any = np.any(counts > 0) and np.any(np.isfinite(mean_ne))
    if valid_any:
        r_max = float(np.nanmax(mean_ne[counts > 0])) * 1.12
        if r_max <= 0:
            r_max = 1.0

    idx_valid = np.where(counts > 0)[0]
    min_idxs: List[int] = []
    max_idxs: List[int] = []
    if idx_valid.size > 0:
        vals = np.array([mean_ne[i] for i in idx_valid], dtype=float)
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        min_idxs = [int(i) for i in idx_valid if np.isclose(mean_ne[i], min_val, rtol=0.0, atol=1e-6)]
        max_idxs = [int(i) for i in idx_valid if np.isclose(mean_ne[i], max_val, rtol=0.0, atol=1e-6)]
        if np.isclose(min_val, max_val):
            max_idxs = []

    def _bar_color(i: int) -> str:
        in_min = i in min_idxs
        in_max = i in max_idxs
        if in_min and in_max:
            return "#9467bd"
        if in_min:
            return "#2ca02c"
        if in_max:
            return "#d62728"
        return "#4c78a8"

    for i in range(len(HEADING_BIN_CENTERS)):
        if counts[i] == 0:
            continue
        ax.bar(
            theta[i],
            mean_ne[i],
            width=width,
            bottom=0.0,
            alpha=0.55,
            color=_bar_color(i),
            edgecolor="white",
            linewidth=0.6,
        )

    order = np.argsort(theta)
    th_s = theta[order]
    r_s = mean_ne[order].astype(float)
    mask = counts[order] > 0
    th_line = th_s[mask]
    r_line = r_s[mask]
    if th_line.size >= 2:
        th_closed = np.append(th_line, th_line[0])
        r_closed = np.append(r_line, r_line[0])
        ax.plot(
            th_closed,
            r_closed,
            color="#d62728",
            linewidth=2.2,
            marker="o",
            markersize=4,
            zorder=5,
        )
    elif th_line.size == 1:
        ax.plot(th_line, r_line, color="#d62728", linewidth=2.2, marker="o", markersize=5, zorder=5)

    ax.set_ylim(0.0, r_max)
    ax.set_title(title, pad=16)
    ax.grid(alpha=0.35)
    n = int(len(df))
    k = int(np.sum(counts > 0))
    fig.text(
        0.5,
        0.02,
        f"n={n}  |  bins with data={k}  |  bin={HEADING_BIN_WIDTH_DEG:.0f}°  |  range=[-180,180]°"
        f"  |  green=min mean NE  |  red=max mean NE",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    fig.tight_layout()
    return fig


@st.cache_resource(show_spinner=False)
def load_citynav_data(split: str) -> CityNavData:
    if split == "new":
        data_path = os.path.join(NEW_DATA_DIR, "citynav_val_unseen_new.json")
    else:
        data_path = os.path.join(MTURK_TRAJECTORY_DIR, f"citynav_val_unseen_{split}.json")
    return CityNavData(data_path)


def _path_length(trajectory) -> float:
    if len(trajectory) < 2:
        return 0.0
    return float(
        sum(curr.xy.dist_to(prev.xy) for curr, prev in zip(trajectory[1:], trajectory[:-1]))
    )


def _list_experiment_checkpoints(experiment_dir: str) -> List[str]:
    if not os.path.isdir(experiment_dir):
        return []
    files = []
    for name in os.listdir(experiment_dir):
        full = os.path.join(experiment_dir, name)
        if os.path.isfile(full) and name.lower().endswith(".pkl"):
            files.append(name)
    files.sort()
    return files


def _default_checkpoint_for_split(files: List[str], split: str) -> str:
    # Prefer explicit custom defaults first.
    preferred = {
        "easy": "checkpoint_easy_mymethod_newdxdy.pkl",
        "medium": "checkpoint_medium_mymethod_newdxdy.pkl",
        "hard": "checkpoint_hard_mymethod_newdxdy.pkl",
    }
    if split in preferred and preferred[split] in files:
        return preferred[split]

    # Then prefer names containing split keyword, then fallback to generic default.
    for name in files:
        if split in name.lower():
            return name
    fallback = f"checkpoint_{split}.pkl"
    if fallback in files:
        return fallback
    return files[0] if files else "(none)"


def _normalize_checkpoint_data(raw):
    # Current eval.py saves dict[episode_id] = list[Pose4D].
    # Keep a fallback for common wrapped structures.
    if isinstance(raw, dict):
        if all(isinstance(v, list) for v in raw.values()):
            return raw
        if "trajectory" in raw and isinstance(raw["trajectory"], dict):
            return raw["trajectory"]
        if "trajectories" in raw and isinstance(raw["trajectories"], dict):
            return raw["trajectories"]
    return {}


def _load_checkpoint_pickle(checkpoint_path: str):
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _build_split_records(split: str, checkpoint_path: str) -> Tuple[pd.DataFrame, Dict]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return pd.DataFrame(), {"count": 0, "ne": np.nan, "sr": np.nan, "osr": np.nan, "spl": np.nan}

    raw = _load_checkpoint_pickle(checkpoint_path)
    traj_by_id = _normalize_checkpoint_data(raw)

    citynav = load_citynav_data(split)
    records: List[Dict] = []
    for idx, ep in enumerate(citynav.episodes):
        if ep.id not in traj_by_id:
            continue
        traj = traj_by_id[ep.id]
        if not traj:
            continue

        final_pose = traj[-1]
        final_dist = float(final_pose.xy.dist_to(ep.target_position.xy))
        oracle_dist = float(min(p.xy.dist_to(ep.target_position.xy) for p in traj))
        sr = float(final_dist <= 20.0)
        osr = float(oracle_dist <= 20.0)
        path_len = _path_length(traj)
        optimal_len = float(ep.target_position.xy.dist_to(ep.start_pose.xy))
        denom = max(path_len, optimal_len)
        spl = float(sr * optimal_len / denom) if denom > 0 else 0.0

        records.append(
            {
                "split": split,
                "index": idx,
                # Normalize to string to avoid mixed object dtype
                # (e.g., int/bytes/tuple across checkpoints) breaking Arrow conversion.
                "episode_id": str(ep.id),
                "ne": final_dist,
                "sr": sr,
                "osr": osr,
                "spl": spl,
                "oracle_ne": oracle_dist,
                "dx_to_target": float(final_pose.x - ep.target_position.x),
                "dy_to_target": float(final_pose.y - ep.target_position.y),
                "path_len": path_len,
                "optimal_len": optimal_len,
                "steps": max(0, len(traj) - 1),
                "initial_heading_deg": _initial_heading_deg_from_episode(ep),
            }
        )

    if not records:
        return pd.DataFrame(), {"count": 0, "ne": np.nan, "sr": np.nan, "osr": np.nan, "spl": np.nan}

    df = pd.DataFrame(records)
    summary = {
        "count": int(len(df)),
        "ne": float(df["ne"].mean()),
        "sr": float(df["sr"].mean()),
        "osr": float(df["osr"].mean()),
        "spl": float(df["spl"].mean()),
    }
    return df, summary


def _summary_from_df(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {"count": 0, "ne": np.nan, "sr": np.nan, "osr": np.nan, "spl": np.nan}
    return {
        "count": int(len(df)),
        "ne": float(df["ne"].mean()),
        "sr": float(df["sr"].mean()),
        "osr": float(df["osr"].mean()),
        "spl": float(df["spl"].mean()),
    }


def _build_hett_records(checkpoint_path: str) -> Tuple[pd.DataFrame, Dict]:
    """Build HETT records against the union of easy/medium/hard episodes."""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return pd.DataFrame(), {"count": 0, "ne": np.nan, "sr": np.nan, "osr": np.nan, "spl": np.nan}

    raw = _load_checkpoint_pickle(checkpoint_path)
    traj_by_id = _normalize_checkpoint_data(raw)
    if not traj_by_id:
        return pd.DataFrame(), {"count": 0, "ne": np.nan, "sr": np.nan, "osr": np.nan, "spl": np.nan}

    records: List[Dict] = []
    seen_ids = set()
    global_index = 0

    for split in SPLITS:
        citynav = load_citynav_data(split)
        for ep in citynav.episodes:
            global_index += 1
            if ep.id in seen_ids:
                continue
            if ep.id not in traj_by_id:
                continue
            traj = traj_by_id[ep.id]
            if not traj:
                continue

            seen_ids.add(ep.id)
            final_pose = traj[-1]
            final_dist = float(final_pose.xy.dist_to(ep.target_position.xy))
            oracle_dist = float(min(p.xy.dist_to(ep.target_position.xy) for p in traj))
            sr = float(final_dist <= 20.0)
            osr = float(oracle_dist <= 20.0)
            path_len = _path_length(traj)
            optimal_len = float(ep.target_position.xy.dist_to(ep.start_pose.xy))
            denom = max(path_len, optimal_len)
            spl = float(sr * optimal_len / denom) if denom > 0 else 0.0

            records.append(
                {
                    "split": "hett",
                    "index": global_index,
                    "episode_id": str(ep.id),
                    "ne": final_dist,
                    "sr": sr,
                    "osr": osr,
                    "spl": spl,
                    "oracle_ne": oracle_dist,
                    "dx_to_target": float(final_pose.x - ep.target_position.x),
                    "dy_to_target": float(final_pose.y - ep.target_position.y),
                    "path_len": path_len,
                    "optimal_len": optimal_len,
                    "steps": max(0, len(traj) - 1),
                    "initial_heading_deg": _initial_heading_deg_from_episode(ep),
                }
            )

    if not records:
        return pd.DataFrame(), {"count": 0, "ne": np.nan, "sr": np.nan, "osr": np.nan, "spl": np.nan}

    df = pd.DataFrame(records)
    return df, _summary_from_df(df)


def _compute_sr_at_thresholds(df: pd.DataFrame, thresholds: List[float]) -> np.ndarray:
    if df.empty:
        return np.array([np.nan] * len(thresholds), dtype=float)
    ne_values = df["ne"].to_numpy()
    return np.array([float((ne_values <= t).mean()) for t in thresholds], dtype=float)


def _draw_sr_threshold_comparison(
    thresholds: List[float], emh_sr: np.ndarray, hett_sr: np.ndarray
):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if np.all(np.isnan(emh_sr)) and np.all(np.isnan(hett_sr)):
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("SR threshold comparison")
        return fig

    x = np.array(thresholds, dtype=float)
    if not np.all(np.isnan(emh_sr)):
        ax.plot(x, emh_sr, marker="o", linewidth=2, color="#4c78a8", label="total (easy+medium+hard)")
    if not np.all(np.isnan(hett_sr)):
        ax.plot(x, hett_sr, marker="s", linewidth=2, color="#e45756", label="hett")

    ax.set_xlabel("SR threshold (m)")
    ax.set_ylabel("SR")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.set_title("SR threshold comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _draw_ne_distribution(df: pd.DataFrame, split: str):
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(f"{split} - NE distribution")
        return fig

    ne_values = df["ne"].to_numpy()
    ne_mean = ne_values.mean()
    ax.hist(ne_values, bins=min(30, max(10, len(ne_values) // 4)), color="#4c78a8", alpha=0.85, edgecolor="white")
    ax.axvline(ne_mean, color="#e45756", linestyle="--", linewidth=2, label=f"mean={ne_mean:.2f}")
    ax.set_title(f"{split} - NE distribution")
    ax.set_xlabel("NE (m)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _shared_sr_offset_limit(*dfs: pd.DataFrame) -> float:
    max_abs = max(SR_THRESHOLDS) + 3.0
    has_data = False
    for df in dfs:
        if df is None or df.empty:
            continue
        has_data = True
        dx = df["dx_to_target"].to_numpy()
        dy = df["dy_to_target"].to_numpy()
        max_abs = max(
            max_abs,
            float(np.abs(dx).max(initial=0.0)),
            float(np.abs(dy).max(initial=0.0)),
        )
    if not has_data:
        max_abs = max(SR_THRESHOLDS) + 3.0
    return float(max_abs + 2.0)


def _draw_sr_offset_scatter(df: pd.DataFrame, split: str, fixed_lim: Optional[float] = None):
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_title(f"{split} - Final offset to target (target at 0,0)")
    ax.set_xlabel("dx to target (m)")
    ax.set_ylabel("dy to target (m)")

    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    dx = df["dx_to_target"].to_numpy()
    dy = df["dy_to_target"].to_numpy()
    sr = df["sr"].to_numpy()
    colors = np.where(sr > 0.5, "#2ca02c", "#d62728")
    ax.scatter(dx, dy, s=14, c=colors, alpha=0.7)

    if fixed_lim is None:
        max_abs = max(
            max(np.abs(dx).max(initial=0.0), np.abs(dy).max(initial=0.0)),
            max(SR_THRESHOLDS) + 3,
        )
        lim = float(max_abs + 2.0)
    else:
        lim = float(fixed_lim)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    for r in SR_THRESHOLDS:
        circ = plt.Circle((0, 0), r, fill=False, linestyle="--", linewidth=1, alpha=0.4, color="#1f77b4")
        ax.add_patch(circ)
    ax.text(0.02, 0.98, "Green=SR1, Red=SR0", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    return fig


def _draw_ne_pareto(ne_values: np.ndarray, title: str, top_k: int = 100):
    fig, ax1 = plt.subplots(figsize=(6.0, 3.8))
    finite_ne = ne_values[np.isfinite(ne_values)]
    if finite_ne.size == 0:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")
        ax1.set_title(title)
        return fig

    sorted_ne = np.sort(finite_ne)[::-1]
    k = min(top_k, sorted_ne.size)
    top_vals = sorted_ne[:k]
    total = float(sorted_ne.sum())
    if total <= 0:
        ax1.text(0.5, 0.5, "Total NE is zero", ha="center", va="center")
        ax1.set_title(title)
        return fig

    # If truncated, aggregate tail into one "others" bar so cumulative line reaches 100%.
    if k < sorted_ne.size:
        others = float(sorted_ne[k:].sum())
        vals = np.concatenate([top_vals, np.array([others])])
        x = np.arange(1, k + 2)
        tick_labels = [str(i) for i in range(1, k + 1)] + ["others"]
    else:
        vals = top_vals
        x = np.arange(1, k + 1)
        tick_labels = [str(i) for i in range(1, k + 1)]

    cum_pct = np.cumsum(vals) / total * 100.0

    ax1.bar(x, vals, color="#4c78a8", alpha=0.85, width=0.9, label="NE by rank")
    ax1.set_xlabel("Case rank (NE descending)")
    ax1.set_ylabel("NE (m)")
    ax1.set_title(title)
    if len(x) <= 25:
        ax1.set_xticks(x)
        ax1.set_xticklabels(tick_labels, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, cum_pct, color="#e45756", linewidth=2, marker="o", markersize=2, label="Cumulative %")
    ax2.set_ylabel("Cumulative NE contribution (%)")
    ax2.set_ylim(0, 105)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right")
    fig.tight_layout()
    return fig


def _draw_ne_sorted_points(ne_values: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if ne_values.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return fig

    sorted_ne = np.sort(ne_values)[::-1]
    x = np.arange(1, sorted_ne.size + 1)
    mean_ne = float(sorted_ne.mean())
    p90_ne = float(np.percentile(sorted_ne, 90))
    p95_ne = float(np.percentile(sorted_ne, 95))

    ax.plot(x, sorted_ne, marker="o", markersize=2.5, linewidth=1.2, color="#4c78a8", alpha=0.9)
    ax.axhline(mean_ne, linestyle="--", color="#f58518", linewidth=1.8, label=f"mean={mean_ne:.2f}")
    ax.axhline(p90_ne, linestyle="--", color="#54a24b", linewidth=1.6, label=f"p90={p90_ne:.2f}")
    ax.axhline(p95_ne, linestyle="--", color="#e45756", linewidth=1.6, label=f"p95={p95_ne:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Case rank (NE descending)")
    ax.set_ylabel("NE (m)")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _draw_ne_threshold_effect(ne_values: np.ndarray, title: str):
    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    if ne_values.size == 0:
        ax1.text(0.5, 0.5, "No data", ha="center", va="center")
        ax1.set_title(title)
        return fig

    max_ne = float(np.max(ne_values))
    upper = max(30.0, max_ne)
    thresholds = np.linspace(5.0, upper, 40)
    clipped_mean = []
    kept_mean = []
    kept_ratio = []
    overflow_ratio = []

    for t in thresholds:
        clipped_mean.append(float(np.minimum(ne_values, t).mean()))
        keep_mask = ne_values <= t
        kept_ratio.append(float(keep_mask.mean()))
        overflow_ratio.append(float((~keep_mask).mean()))
        kept_mean.append(float(ne_values[keep_mask].mean()) if keep_mask.any() else np.nan)

    ax1.plot(thresholds, clipped_mean, color="#4c78a8", linewidth=2, label="mean(min(NE, t))")
    ax1.plot(thresholds, kept_mean, color="#f58518", linewidth=1.8, linestyle="--", label="mean(NE | NE<=t)")
    ax1.set_xlabel("Threshold t (m)")
    ax1.set_ylabel("NE objective (m)")
    ax1.set_title(title)
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, kept_ratio, color="#54a24b", linewidth=2, label="kept ratio (NE<=t)")
    ax2.plot(thresholds, overflow_ratio, color="#e45756", linewidth=1.8, linestyle="--", label="overflow ratio (NE>t)")
    ax2.set_ylabel("Ratio")
    ax2.set_ylim(-0.02, 1.02)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    fig.tight_layout()
    return fig


def _prepare_case_map(split: str, index: int, checkpoint_path: str) -> Tuple[Image.Image, str] | Tuple[None, str]:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None, "Checkpoint path invalid."

    raw = _load_checkpoint_pickle(checkpoint_path)
    traj_by_id = _normalize_checkpoint_data(raw)
    citynav = load_citynav_data(split)
    if index < 0 or index >= len(citynav):
        return None, "Index out of range for selected split."

    episode = citynav.episodes[index]
    if episode.id not in traj_by_id:
        return None, "This index does not exist in checkpoint."

    nav = NavGym(citynav[index])
    trajectory = traj_by_id[episode.id]
    if not trajectory:
        return None, "Trajectory is empty."

    image = Image.open(nav.cur_whole_map).convert("RGB")
    draw = ImageDraw.Draw(image)

    gt_x, gt_y = nav.target_px
    start_x, start_y = nav._get_px(nav.start_pose)
    final_x, final_y = nav._get_px(trajectory[-1])

    # Draw trajectory
    px_points = [nav._get_px(p) for p in trajectory]
    if len(px_points) >= 2:
        draw.line([(x, y) for x, y in px_points], fill=(255, 255, 0), width=9)

    # Draw start / target / final
    draw.ellipse((start_x - 10, start_y - 10, start_x + 10, start_y + 10), outline=(0, 255, 0), width=9)
    draw.ellipse((gt_x - 20, gt_y - 20, gt_x + 20, gt_y + 20), outline=(255, 0, 0), width=24)
    draw.ellipse((final_x - 14, final_y - 14, final_x + 14, final_y + 14), outline=(255, 0, 255), width=12)

    # SR threshold circle (20m) around target
    rx = max(1, int(20 / nav.px_real_size[0]))
    ry = max(1, int(20 / nav.px_real_size[1]))
    draw.ellipse((gt_x - rx, gt_y - ry, gt_x + rx, gt_y + ry), outline=(255, 165, 0), width=24)

    ne = float(trajectory[-1].xy.dist_to(episode.target_position.xy))
    caption = (
        f"split={split}, index={index}, episode_id={episode.id}, "
        f"NE={ne:.2f}m. Green=start, Red=target, Magenta=final, Yellow=trajectory, Orange=SR(20m)."
    )
    return image, caption


def main():
    st.set_page_config(page_title="Checkpoint Analyzer", layout="wide")
    st.title("Checkpoint Analyzer (easy / medium / hard)")
    st.caption("用于分析 eval.py 保存的 checkpoint（当前仅包含 trajectory，不包含模型输入输出文本）。")

    with st.sidebar:
        st.subheader("Checkpoint paths")
        experiment_dir = "./experiment"
        ckpt_files = _list_experiment_checkpoints(experiment_dir)
        options = ["(none)"] + ckpt_files

        default_easy = _default_checkpoint_for_split(ckpt_files, "easy")
        default_medium = _default_checkpoint_for_split(ckpt_files, "medium")
        default_hard = _default_checkpoint_for_split(ckpt_files, "hard")

        easy_choice = st.selectbox(
            "easy checkpoint",
            options=options,
            index=options.index(default_easy) if default_easy in options else 0,
        )
        medium_choice = st.selectbox(
            "medium checkpoint",
            options=options,
            index=options.index(default_medium) if default_medium in options else 0,
        )
        hard_choice = st.selectbox(
            "hard checkpoint",
            options=options,
            index=options.index(default_hard) if default_hard in options else 0,
        )

        ckpt_easy = os.path.join(experiment_dir, easy_choice) if easy_choice != "(none)" else ""
        ckpt_medium = os.path.join(experiment_dir, medium_choice) if medium_choice != "(none)" else ""
        ckpt_hard = os.path.join(experiment_dir, hard_choice) if hard_choice != "(none)" else ""

        st.markdown("---")
        st.subheader("HETT checkpoint")
        default_hett_ckpt = "./HETT/checkpoints/multi/checkpoint_val_unseen.pkl"
        ckpt_hett = st.text_input("HETT checkpoint path", value=default_hett_ckpt)
        refresh = st.button("Refresh / Recompute", use_container_width=True)

    if refresh:
        st.cache_resource.clear()

    split_to_path = {"easy": ckpt_easy, "medium": ckpt_medium, "hard": ckpt_hard}
    split_df: Dict[str, pd.DataFrame] = {}
    split_summary: Dict[str, Dict] = {}

    for split in SPLITS:
        df, summary = _build_split_records(split, split_to_path[split])
        split_df[split] = df
        split_summary[split] = summary

    hett_df, hett_summary = _build_hett_records(ckpt_hett)
    emh_df = pd.concat([split_df[s] for s in SPLITS if not split_df[s].empty], ignore_index=True)
    emh_summary = _summary_from_df(emh_df)

    # Row 1: table
    total_count = emh_summary["count"]
    rows = []
    for split in SPLITS:
        c = split_summary[split]["count"]
        w = (c / total_count) if total_count > 0 else np.nan
        rows.append(
            {
                "Split": split,
                "NE": split_summary[split]["ne"],
                "SR": split_summary[split]["sr"],
                "OSR": split_summary[split]["osr"],
                "SPL": split_summary[split]["spl"],
                "Weight": w,
                "Count": c,
            }
        )
    rows.append(
        {
            "Split": "hett",
            "NE": hett_summary["ne"],
            "SR": hett_summary["sr"],
            "OSR": hett_summary["osr"],
            "SPL": hett_summary["spl"],
            "Weight": np.nan,
            "Count": hett_summary["count"],
        }
    )

    rows.append(
        {
            "Split": "total",
            "NE": emh_summary["ne"],
            "SR": emh_summary["sr"],
            "OSR": emh_summary["osr"],
            "SPL": emh_summary["spl"],
            "Weight": 1.0 if total_count > 0 else np.nan,
            "Count": total_count,
        }
    )
    table_df = pd.DataFrame(rows).set_index("Split")

    st.subheader("1) Split-level metrics table")
    st.dataframe(
        table_df.style.format(
            {
                "NE": "{:.4f}",
                "SR": "{:.4f}",
                "OSR": "{:.4f}",
                "SPL": "{:.4f}",
                "Weight": "{:.4f}",
                "Count": "{:.0f}",
            }
        ),
        use_container_width=True,
    )

    # Row 2: NE distributions
    st.subheader("2) NE distribution by split")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.pyplot(_draw_ne_distribution(split_df["easy"], "easy"), use_container_width=True)
    with c2:
        st.pyplot(_draw_ne_distribution(split_df["medium"], "medium"), use_container_width=True)
    with c3:
        st.pyplot(_draw_ne_distribution(split_df["hard"], "hard"), use_container_width=True)
    c4, c5 = st.columns(2)
    with c4:
        st.pyplot(
            _draw_ne_distribution(emh_df, "total (easy+medium+hard)"),
            use_container_width=True,
        )
    with c5:
        st.pyplot(_draw_ne_distribution(hett_df, "hett"), use_container_width=True)

    # Row 3: SR offset scatter
    st.subheader("3) SR offset analysis (target normalized to (0,0))")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.pyplot(_draw_sr_offset_scatter(split_df["easy"], "easy"), use_container_width=True)
    with s2:
        st.pyplot(_draw_sr_offset_scatter(split_df["medium"], "medium"), use_container_width=True)
    with s3:
        st.pyplot(_draw_sr_offset_scatter(split_df["hard"], "hard"), use_container_width=True)
    compare_scatter_lim = _shared_sr_offset_limit(emh_df, hett_df)
    s4, s5 = st.columns(2)
    with s4:
        st.pyplot(
            _draw_sr_offset_scatter(emh_df, "total (easy+medium+hard)", fixed_lim=compare_scatter_lim),
            use_container_width=False,
        )
    with s5:
        st.pyplot(
            _draw_sr_offset_scatter(hett_df, "hett", fixed_lim=compare_scatter_lim),
            use_container_width=False,
        )

    st.subheader("Initial Heading NE analysis初始航向东北分析")
    st.caption(
        "bin size = 15° | range = [-180, 180]° | metric = mean NE to goal (mean_final_pos_to_goal_dist) | "
        "initial heading = drone start_pose.yaw (degrees, wrapped)"
    )
    ih_c1, ih_c2 = st.columns(2)
    with ih_c1:
        st.pyplot(
            _draw_initial_heading_polar(emh_df, "total (easy+medium+hard) — initial heading vs mean NE"),
            use_container_width=True,
        )
    with ih_c2:
        st.pyplot(
            _draw_initial_heading_polar(hett_df, "hett — initial heading vs mean NE"),
            use_container_width=True,
        )

    # Row 4: NE deep dive
    st.subheader("4) NE deep-dive analysis")
    ne_scope_options = ["total", "easy", "medium", "hard"]
    ne_scope = st.selectbox("NE analysis scope", ne_scope_options, index=0)

    if ne_scope == "total":
        ne_df = pd.concat([split_df[s] for s in SPLITS if not split_df[s].empty], ignore_index=True)
    else:
        ne_df = split_df[ne_scope].copy()
    ne_values = ne_df["ne"].to_numpy() if not ne_df.empty else np.array([])

    p1, p2 = st.columns(2)
    with p1:
        st.pyplot(
            _draw_ne_sorted_points(ne_values, f"Sorted NE Point Plot ({ne_scope})"),
            use_container_width=True,
        )
    with p2:
        st.pyplot(
            _draw_ne_threshold_effect(
                ne_values,
                f"Threshold truncation effect ({ne_scope})",
            ),
            use_container_width=True,
        )
    if ne_values.size > 0:
        t_candidates = [400, 350, 300, 250, 200, 150, 100]
        rows_obj = []
        for t in t_candidates:
            keep = ne_values <= t
            rows_obj.append(
                {
                    "threshold": t,
                    "kept_ratio": float(keep.mean()),
                    "overflow_ratio": float((~keep).mean()),
                    "mean_min_ne": float(np.minimum(ne_values, t).mean()),
                    "mean_ne_kept": float(ne_values[keep].mean()) if keep.any() else np.nan,
                }
            )
        st.dataframe(
            pd.DataFrame(rows_obj).style.format(
                {
                    "threshold": "{:.0f}",
                    "kept_ratio": "{:.3f}",
                    "overflow_ratio": "{:.3f}",
                    "mean_min_ne": "{:.3f}",
                    "mean_ne_kept": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    # Row 5: failure case list + case map
    st.subheader("5) Case explorer (for manual_eval_app index drill-down)")
    fs1, fs2, fs3 = st.columns([1.0, 1.0, 1.2])
    with fs1:
        selected_split = st.selectbox("Split", SPLITS, index=0)
    with fs2:
        ne_threshold = st.number_input("NE threshold (for SR=0)", min_value=0.0, value=50.0, step=5.0)
    with fs3:
        only_sr0 = st.checkbox("Only SR=0", value=True)

    case_df = split_df[selected_split].copy()
    if only_sr0 and not case_df.empty:
        case_df = case_df[case_df["sr"] < 0.5]
    if not case_df.empty:
        case_df = case_df[case_df["ne"] > float(ne_threshold)]
        case_df = case_df.sort_values(by="ne", ascending=False)

    if case_df.empty:
        st.info("No matching cases under current filter.")
    else:
        case_df["episode_id"] = case_df["episode_id"].astype(str)
        display_cols = ["index", "episode_id", "ne", "sr", "osr", "spl", "dx_to_target", "dy_to_target", "steps"]
        st.dataframe(
            case_df[display_cols].style.format(
                {
                    "ne": "{:.2f}",
                    "sr": "{:.0f}",
                    "osr": "{:.0f}",
                    "spl": "{:.4f}",
                    "dx_to_target": "{:.2f}",
                    "dy_to_target": "{:.2f}",
                    "steps": "{:.0f}",
                }
            ),
            use_container_width=True,
            height=280,
        )
        idx_list = case_df["index"].astype(int).tolist()
        st.code(
            f"manual_eval_app.py 可直接复盘的 {selected_split} index:\n"
            + ", ".join(map(str, idx_list[:200])),
            language=None,
        )

        st.markdown("**Top 9 cases map view (3x3, in current sorted order):**")
        page_size = 9
        total_pages = max(1, int(np.ceil(len(idx_list) / page_size)))
        signature = (selected_split, float(ne_threshold), bool(only_sr0), len(idx_list))
        if "case_grid_page" not in st.session_state:
            st.session_state.case_grid_page = 0
        if "case_grid_signature" not in st.session_state:
            st.session_state.case_grid_signature = signature
        if st.session_state.case_grid_signature != signature:
            st.session_state.case_grid_signature = signature
            st.session_state.case_grid_page = 0

        nav1, nav2, nav3 = st.columns([1, 2, 1])
        with nav1:
            if st.button("Previous 9", use_container_width=True, disabled=st.session_state.case_grid_page <= 0):
                st.session_state.case_grid_page -= 1
        with nav2:
            start_idx = st.session_state.case_grid_page * page_size
            end_idx = min(start_idx + page_size, len(idx_list))
            st.markdown(
                f"<div style='text-align:center;'>"
                f"Page {st.session_state.case_grid_page + 1}/{total_pages} "
                f"(showing {start_idx + 1}-{end_idx} / {len(idx_list)})"
                f"</div>",
                unsafe_allow_html=True,
            )
        with nav3:
            if st.button(
                "Next 9",
                use_container_width=True,
                disabled=st.session_state.case_grid_page >= total_pages - 1,
            ):
                st.session_state.case_grid_page += 1

        start_idx = st.session_state.case_grid_page * page_size
        end_idx = min(start_idx + page_size, len(idx_list))
        top9_idx = idx_list[start_idx:end_idx]
        for row_start in range(0, len(top9_idx), 3):
            cols = st.columns(3)
            row_indices = top9_idx[row_start: row_start + 3]
            for col, idx_case in zip(cols, row_indices):
                with col:
                    image, caption = _prepare_case_map(
                        selected_split, int(idx_case), split_to_path[selected_split]
                    )
                    if image is None:
                        st.warning(f"index={idx_case}: {caption}")
                    else:
                        st.image(image, width=640)
                        st.caption(caption)

    # Row 6: SR threshold comparison
    st.subheader("6) SR threshold comparison (total vs HETT)")
    emh_sr_curve = _compute_sr_at_thresholds(emh_df, SR_COMPARE_THRESHOLDS)
    hett_sr_curve = _compute_sr_at_thresholds(hett_df, SR_COMPARE_THRESHOLDS)
    st.pyplot(
        _draw_sr_threshold_comparison(SR_COMPARE_THRESHOLDS, emh_sr_curve, hett_sr_curve),
        use_container_width=True,
    )

    threshold_table = pd.DataFrame(
        {
            "threshold": SR_COMPARE_THRESHOLDS,
            "sr_total_emh": emh_sr_curve,
            "sr_hett": hett_sr_curve,
            "sr_delta_hett_minus_total": hett_sr_curve - emh_sr_curve,
        }
    )
    st.dataframe(
        threshold_table.style.format(
            {
                "threshold": "{:.0f}",
                "sr_total_emh": "{:.4f}",
                "sr_hett": "{:.4f}",
                "sr_delta_hett_minus_total": "{:+.4f}",
            }
        ),
        use_container_width=True,
    )


if __name__ == "__main__":
    main()
