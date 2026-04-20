"""
Streamlit dashboard: spatial expression taxonomy over CityNav instructions (target_description).
Single-file app; run: streamlit run human_eval/prompt_analysis.py
"""

from __future__ import annotations

import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
HETT_ROOT = PROJECT_ROOT / "HETT"
if str(HETT_ROOT) not in sys.path:
    sys.path.append(str(HETT_ROOT))

from gsamllavanav.defaultpaths import MTURK_TRAJECTORY_DIR
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym

NEW_DATA_DIR = "/home/yjy/flightgpt/FlightGPT/refine_citynav/processed_citynav"
SPLITS = ["easy", "medium", "hard", "new"]
# Dataset A in the main comparison is always the union of these three (see plan / eval convention).
EMH_SPLITS: Tuple[str, ...] = ("easy", "medium", "hard")
DATASET_A_LABEL = "easy+medium+hard"

# Fixed axis order for charts and tables
CATEGORY_ORDER: List[str] = [
    "left",
    "right",
    "mixed",
    "between",
    "around",
    "near",
    "front",
    "behind",
    "inside",
    "outside",
    "corner",
    "directional",
    "other",
    "none",
]

PRIORITY_ORDER: List[str] = [
    "left",
    "right",
    "between",
    "around",
    "near",
    "front",
    "behind",
    "inside",
    "outside",
    "corner",
    "directional",
]

# phrase -> primary label (built in _build_phrase_table)
_PHRASE_ROWS: List[Tuple[str, str]] = [
    # left
    ("to the left of", "left"),
    ("left side of", "left"),
    ("on the left", "left"),
    ("left of", "left"),
    # right
    ("to the right of", "right"),
    ("right side of", "right"),
    ("on the right", "right"),
    ("right of", "right"),
    # between
    ("in between", "between"),
    ("between", "between"),
    # around
    ("surrounding area", "around"),
    ("surrounding", "around"),
    ("around", "around"),
    # near
    ("adjacent to", "near"),
    ("close to", "near"),
    ("next to", "near"),
    ("nearby", "near"),
    ("near", "near"),
    # front
    ("in front of", "front"),
    ("ahead of", "front"),
    # behind
    ("at the back of", "behind"),
    ("behind", "behind"),
    # inside / outside
    ("within", "inside"),
    ("inside", "inside"),
    ("outside", "outside"),
    ("out of", "outside"),
    # corner
    ("at the corner", "corner"),
    # directional
    ("north of", "directional"),
    ("south of", "directional"),
    ("east of", "directional"),
    ("west of", "directional"),
]

# Single-token phrases matched with word boundaries to reduce false positives
_WORD_BOUNDARY_LABELS: Dict[str, str] = {
    "left": "left",
    "right": "right",
    "corner": "corner",
}


def _build_sorted_phrases() -> List[Tuple[str, str]]:
    rows = list(_PHRASE_ROWS)
    for word, lab in _WORD_BOUNDARY_LABELS.items():
        rows.append((word, lab))
    rows.sort(key=lambda x: len(x[0]), reverse=True)
    return rows


_SORTED_PHRASES: List[Tuple[str, str]] = _build_sorted_phrases()


def _match_labels(text_lower: str) -> Set[str]:
    """Return set of matched primary labels (left/right/.../directional/corner)."""
    found: Set[str] = set()
    for phrase, lab in _SORTED_PHRASES:
        if " " in phrase or len(phrase) > 5:
            if phrase in text_lower:
                found.add(lab)
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", text_lower):
                found.add(lab)
    return found


def spatial_label_for_instruction(instruction: str) -> str:
    """Single primary label per prompt."""
    text = instruction.lower()
    labels = _match_labels(text)
    if "left" in labels and "right" in labels:
        return "mixed"
    if not labels:
        return "none"
    for lab in PRIORITY_ORDER:
        if lab in labels:
            return lab
    # Matched only labels outside PRIORITY_ORDER (should not happen)
    return "other"


@st.cache_resource(show_spinner=False)
def load_citynav_data(split: str) -> CityNavData:
    if split == "new":
        data_path = os.path.join(NEW_DATA_DIR, "citynav_val_unseen_new.json")
    else:
        data_path = os.path.join(MTURK_TRAJECTORY_DIR, f"citynav_val_unseen_{split}.json")
    return CityNavData(data_path)


@st.cache_data(show_spinner="Loading episodes…")
def load_labeled_split(split: str) -> pd.DataFrame:
    citynav = load_citynav_data(split)
    rows: List[Dict] = []
    for idx, ep in enumerate(citynav.episodes):
        instruction = ep.target_description
        label = spatial_label_for_instruction(instruction)
        rows.append(
            {
                "index": idx,
                "instruction": instruction,
                "label": label,
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Loading easy + medium + hard…")
def load_labeled_emh_combined() -> pd.DataFrame:
    """Concatenate labeled rows for easy, medium, hard (one DataFrame for dataset A)."""
    parts: List[pd.DataFrame] = []
    for s in EMH_SPLITS:
        df = load_labeled_split(s)
        df = df.copy()
        df["split"] = s
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def _normalize_checkpoint_data(raw):
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


def _contour_bbox_px(px_arr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """px_arr: int32 points; return (x0,y0,x1,y1) in PIL coords."""
    if px_arr is None or px_arr.size == 0:
        return None
    flat = np.asarray(px_arr).reshape(-1, 2)
    if flat.size < 2:
        return None
    # CityNavData stores [row, col] as [y, x] in _get_pose_px — manual_eval treats _get_px as x,y pixel
    ys = flat[:, 0]
    xs = flat[:, 1]
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1)


def render_map_view(
    nav: NavGym,
    checkpoint_path: Optional[str],
    caption_extra: str = "",
) -> Tuple[Optional[Image.Image], str]:
    """
    Green=start, Red=target, Blue=GT teacher path, Orange=SR(20m),
    Magenta=checkpoint final pose if available, Yellow=landmark bbox from GT landmarks.
    """
    episode = nav.episode
    image = Image.open(nav.cur_whole_map).convert("RGB")
    draw = ImageDraw.Draw(image)

    cur_x, cur_y = nav._get_px(nav.start_pose)
    tgt_x, tgt_y = nav.target_px

    r_cur = 14
    r_tgt = 28
    draw.ellipse((cur_x - r_cur, cur_y - r_cur, cur_x + r_cur, cur_y + r_cur), outline=(0, 255, 0), width=4)
    draw.ellipse((tgt_x - r_tgt, tgt_y - r_tgt, tgt_x + r_tgt, tgt_y + r_tgt), outline=(255, 0, 0), width=16)

    gt_px = [nav._get_px(p) for p in nav.episode.teacher_trajectory]
    if len(gt_px) >= 2:
        line_points = [(px[0], px[1]) for px in gt_px]
        draw.line(line_points, fill=(30, 144, 255), width=3)
    for px_x, px_y in gt_px:
        draw.ellipse((px_x - 3, px_y - 3, px_x + 3, px_y + 3), fill=(30, 144, 255))

    rx = max(1, int(20 / nav.px_real_size[0]))
    ry = max(1, int(20 / nav.px_real_size[1]))
    draw.ellipse((tgt_x - rx, tgt_y - ry, tgt_x + rx, tgt_y + ry), outline=(255, 165, 0), width=16)

    # Yellow boxes: landmark regions for description landmarks (aligned index with px_list)
    lm_names = [lm.name for lm in nav.map.landmark_map.landmarks]
    for i, _name in enumerate(lm_names):
        if i >= len(nav.px_list):
            break
        bbox = _contour_bbox_px(nav.px_list[i])
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=3)

    ne = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        raw = _load_checkpoint_pickle(checkpoint_path)
        traj_by_id = _normalize_checkpoint_data(raw)
        if episode.id in traj_by_id and traj_by_id[episode.id]:
            traj = traj_by_id[episode.id]
            final_pose = traj[-1]
            fx, fy = nav._get_px(final_pose)
            draw.ellipse((fx - 14, fy - 14, fx + 14, fy + 14), fill=(255, 0, 255), outline=(255, 255, 255), width=2)
            ne = float(final_pose.xy.dist_to(episode.target_position.xy))

    cap = (
        f"Green=start, Red=target, Magenta=final (if ckpt), Yellow=landmark bbox, "
        f"Blue=GT path, Orange=SR(20m). {caption_extra}"
    )
    if ne is not None:
        cap += f" NE={ne:.2f}m."
    return image, cap


def _counts_by_category(df: pd.DataFrame) -> Dict[str, int]:
    out = {c: 0 for c in CATEGORY_ORDER}
    if df.empty:
        return out
    vc = df["label"].value_counts()
    for c in CATEGORY_ORDER:
        out[c] = int(vc.get(c, 0))
    return out


def _grouped_bar_fig(
    categories: List[str],
    series_a: np.ndarray,
    series_b: np.ndarray,
    name_a: str,
    name_b: str,
    ylabel: str,
    title: str,
) -> plt.Figure:
    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(10.0, len(categories) * 0.55), 4.2))
    ax.bar(x - width / 2, series_a, width, label=name_a)
    ax.bar(x + width / 2, series_b, width, label=name_b)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


def _sample_instructions(df: pd.DataFrame, label: str, n: int, rng: random.Random) -> List[str]:
    sub = df[df["label"] == label]["instruction"].tolist()
    if not sub:
        return []
    k = min(n, len(sub))
    return rng.sample(sub, k=k)


def main() -> None:
    st.set_page_config(page_title="Prompt spatial analysis", layout="wide")

    st.sidebar.header("Parameters")
    n_examples = st.sidebar.number_input("Examples per category (each split)", min_value=1, max_value=20, value=4)
    top_k = st.sidebar.number_input("Top-k largest ratio gaps", min_value=1, max_value=len(CATEGORY_ORDER), value=3)
    rng_seed = st.sidebar.number_input("Random seed (examples)", min_value=0, max_value=2**31 - 1, value=42)
    ckpt_map = st.sidebar.text_input(
        "Checkpoint .pkl (optional, Magenta=final pose)",
        value="",
        help="Must contain trajectories keyed by episode id; same format as eval.py checkpoints.",
    )

    st.title("Prompt spatial analysis")
    st.caption(
        "Word-list taxonomy over `target_description`; **Dataset A** is always "
        f"**{DATASET_A_LABEL}** combined. Pick **Dataset B** as a single split to compare."
    )

    st.subheader("A) Dataset selection")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.markdown(f"**Dataset A:** `{DATASET_A_LABEL}` (combined)")
    with c2:
        split_b = st.selectbox("Dataset B", SPLITS, index=min(3, len(SPLITS) - 1), key="split_b")

    df_a = load_labeled_emh_combined()
    df_b = load_labeled_split(split_b)

    with c3:
        st.metric(f"Samples A ({DATASET_A_LABEL})", len(df_a))
        st.metric(f"Samples B ({split_b})", len(df_b))

    # —— Stats & plots ——
    st.subheader("B) Statistics & comparison")
    counts_a = _counts_by_category(df_a)
    counts_b = _counts_by_category(df_b)
    total_a = max(len(df_a), 1)
    total_b = max(len(df_b), 1)
    ratio_a = {k: counts_a[k] / total_a for k in CATEGORY_ORDER}
    ratio_b = {k: counts_b[k] / total_b for k in CATEGORY_ORDER}

    arr_ca = np.array([counts_a[c] for c in CATEGORY_ORDER], dtype=float)
    arr_cb = np.array([counts_b[c] for c in CATEGORY_ORDER], dtype=float)
    arr_ra = np.array([ratio_a[c] for c in CATEGORY_ORDER], dtype=float)
    arr_rb = np.array([ratio_b[c] for c in CATEGORY_ORDER], dtype=float)

    g1, g2 = st.columns(2)
    with g1:
        fig1 = _grouped_bar_fig(
            CATEGORY_ORDER,
            arr_ca,
            arr_cb,
            f"A ({DATASET_A_LABEL})",
            f"B ({split_b})",
            "Count",
            "Absolute counts",
        )
        st.pyplot(fig1)
        plt.close(fig1)
    with g2:
        fig2 = _grouped_bar_fig(
            CATEGORY_ORDER,
            arr_ra,
            arr_rb,
            f"A ({DATASET_A_LABEL})",
            f"B ({split_b})",
            "Ratio",
            "Normalized ratio (within dataset)",
        )
        st.pyplot(fig2)
        plt.close(fig2)

    diffs = [(c, abs(ratio_a[c] - ratio_b[c])) for c in CATEGORY_ORDER]
    diffs.sort(key=lambda x: x[1], reverse=True)
    top = diffs[:top_k]

    st.markdown("**Largest |ratio_A − ratio_B|**")
    tbl = []
    col_a = "ratio_A_EMH"  # easy + medium + hard
    col_b = f"ratio_B_{split_b}"
    for cat, d in top:
        tbl.append(
            {
                "category": cat,
                col_a: ratio_a[cat],
                col_b: ratio_b[cat],
                "|diff|": d,
            }
        )
    st.dataframe(pd.DataFrame(tbl), use_container_width=True)

    rng = random.Random(rng_seed)
    with st.expander("Representative examples per category", expanded=False):
        for cat in CATEGORY_ORDER:
            st.markdown(f"**{cat}**")
            ex_a = _sample_instructions(df_a, cat, n_examples, rng)
            ex_b = _sample_instructions(df_b, cat, n_examples, rng)
            ca, cb = st.columns(2)
            with ca:
                st.markdown(f"*{DATASET_A_LABEL}*")
                for t in ex_a:
                    st.markdown(f"- {t}")
            with cb:
                st.markdown(f"*{split_b}*")
                for t in ex_b:
                    st.markdown(f"- {t}")

    with st.expander("Top gap categories — sample prompts", expanded=False):
        for cat, d in top:
            st.markdown(f"**{cat}** (|Δ|={d:.4f})")
            ex_a = _sample_instructions(df_a, cat, min(3, n_examples), rng)
            ex_b = _sample_instructions(df_b, cat, min(3, n_examples), rng)
            ca, cb = st.columns(2)
            with ca:
                st.caption(DATASET_A_LABEL)
                for t in ex_a:
                    st.text(t)
            with cb:
                st.caption(split_b)
                for t in ex_b:
                    st.text(t)

    # —— Left / right module ——
    st.subheader("C) Left / right / mixed explorer + map")
    split_lr = st.selectbox("Split for this list", SPLITS, index=0, key="split_lr")
    df_lr = load_labeled_split(split_lr)
    lr_df = df_lr[df_lr["label"].isin(["left", "right", "mixed"])].copy()
    if lr_df.empty:
        st.info("No left/right/mixed prompts in this split under current rules.")
    else:
        lr_df = lr_df.reset_index(drop=True)

        def _format_lr_row(i: int) -> str:
            r = lr_df.iloc[i]
            prev = str(r["instruction"])[:120]
            return f"[ep_idx={int(r['index'])}] {prev}"

        row_i = st.selectbox(
            "Select a prompt",
            list(range(len(lr_df))),
            index=0,
            format_func=_format_lr_row,
        )
        row = lr_df.iloc[row_i]
        idx = int(row["index"])
        lab = row["label"]
        st.text(row["instruction"])

        nav = NavGym(load_citynav_data(split_lr)[idx])
        extra = f"split={split_lr}, index={idx}, episode_id={nav.episode.id}, label={lab}"
        img, cap = render_map_view(nav, ckpt_map if ckpt_map.strip() else None, caption_extra=extra)
        if img is not None:
            st.image(img, use_container_width=True)
            st.caption(cap)


if __name__ == "__main__":
    main()
