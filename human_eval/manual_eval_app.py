import os
import re
import sys
import time
import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw

# Ensure project-root imports work when launched via Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gsamllavanav.defaultpaths import MTURK_TRAJECTORY_DIR
from gsamllavanav.space import Point3D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from navgym.agents.CityNavAgent import GPTAgent, get_prompt
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.tools.EvalTools import eval_planning_metrics

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None


SPLITS = ["easy", "medium", "hard", "new"]
NEW_DATA_DIR = "/home/yjy/flightgpt/FlightGPT/refine_citynav/processed_citynav"
ACTION_TO_ID = {
    "Move Forward": 1,
    "Turn Left": 3,
    "Turn Right": 2,
}

DEFAULT_API_CONFIG = {
    "api_key": "EMPTY",
    "api_base": "http://0.0.0.0:8989/v1",
    "api_version": "2024-05-01-preview",
    "model": "qwen_2_5_vl_7b",
    "system_prompt": (
        "You are an intelligent autonomous aerial vehicle (UAV) "
        "equipped for real-world navigation and visual target localization."
    ),
    "compress_images": True,
}


def _default_state() -> None:
    if "active_split" not in st.session_state:
        st.session_state.active_split = "easy"
    if "episode_index" not in st.session_state:
        st.session_state.episode_index = 0
    if "navgym" not in st.session_state:
        st.session_state.navgym = None
    if "episode_id" not in st.session_state:
        st.session_state.episode_id = None
    if "trajectories" not in st.session_state:
        st.session_state.trajectories = {s: {} for s in SPLITS}
    for s in SPLITS:
        if s not in st.session_state.trajectories:
            st.session_state.trajectories[s] = {}
    if "is_busy" not in st.session_state:
        st.session_state.is_busy = False
    if "control_mode" not in st.session_state:
        st.session_state.control_mode = "Manual actions"
    if "clicked_px" not in st.session_state:
        st.session_state.clicked_px = None
    if "show_gt_overlay" not in st.session_state:
        st.session_state.show_gt_overlay = False
    if "vllm_test_results" not in st.session_state:
        st.session_state.vllm_test_results = None
    # vLLM model state
    if "model_input" not in st.session_state:
        st.session_state.model_input = None
    if "model_response" not in st.session_state:
        st.session_state.model_response = None
    if "model_pred_px" not in st.session_state:
        st.session_state.model_pred_px = None
    if "model_landmark_bbox" not in st.session_state:
        st.session_state.model_landmark_bbox = None
    if "model_step_count" not in st.session_state:
        st.session_state.model_step_count = 0
    if "model_elapsed" not in st.session_state:
        st.session_state.model_elapsed = 0.0


@st.cache_resource(show_spinner=False)
def load_citynav_data(split: str) -> CityNavData:
    if split == "new":
        data_path = os.path.join(NEW_DATA_DIR, "citynav_val_unseen_new.json")
    else:
        data_path = os.path.join(MTURK_TRAJECTORY_DIR, f"citynav_val_unseen_{split}.json")
    return CityNavData(data_path)


def weighted_metrics(split_metrics: Dict, split_counts: Dict):
    valid = [s for s in SPLITS if split_metrics.get(s) is not None and split_counts.get(s, 0) > 0]
    if not valid:
        return None
    total = sum(split_counts[s] for s in valid)
    ne = sum(split_metrics[s].mean_final_pos_to_goal_dist * split_counts[s] / total for s in valid)
    sr = sum(split_metrics[s].success_rate_final_pos_to_goal * split_counts[s] / total for s in valid)
    osr = sum(split_metrics[s].success_rate_oracle_pos_to_goal * split_counts[s] / total for s in valid)
    spl = sum(split_metrics[s].success_rate_weighted_by_path_length * split_counts[s] / total for s in valid)
    return ne, sr, osr, spl


def load_episode(split: str, index: int) -> None:
    citynav = load_citynav_data(split)
    if index < 0 or index >= len(citynav):
        st.error("Episode index out of range.")
        return
    nav = NavGym(citynav[index])
    st.session_state.navgym = nav
    st.session_state.episode_id = citynav.episodes[index].id
    st.session_state.clicked_px = None
    st.session_state.show_gt_overlay = False
    _clear_model_state()


def _clear_model_state() -> None:
    st.session_state.model_input = None
    st.session_state.model_response = None
    st.session_state.model_pred_px = None
    st.session_state.model_landmark_bbox = None
    st.session_state.model_step_count = 0
    st.session_state.model_elapsed = 0.0


def _format_model_response(raw: str) -> str:
    """Convert raw <think>/<answer> model output into readable markdown."""
    think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    parts = []
    if think_match:
        parts.append("#### Reasoning\n" + think_match.group(1).strip())
    if answer_match:
        parts.append("#### Answer\n" + answer_match.group(1).strip())
    if not parts:
        return raw
    return "\n\n---\n\n".join(parts)


def _parse_bbox(text: str, key: str = "landmark_bbox") -> Optional[List[int]]:
    pattern = rf'"{key}"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    m = re.search(pattern, text)
    return list(map(int, m.groups())) if m else None


def _parse_location(text: str) -> Optional[List[int]]:
    m = re.search(r'"target_location"\s*:\s*\[(\d+),\s*(\d+)\]', text)
    return list(map(int, m.groups())) if m else None


def _test_vllm_connection() -> Dict:
    """Run a quick health / text / vision test against the configured vLLM server."""
    import requests
    from openai import OpenAI

    cfg = DEFAULT_API_CONFIG
    results: Dict = {}

    # 1. Health endpoint
    health_url = cfg["api_base"].replace("/v1", "") + "/health"
    try:
        r = requests.get(health_url, timeout=5)
        results["health"] = {"ok": r.status_code == 200, "status": r.status_code, "time": 0.0}
    except Exception as e:
        results["health"] = {"ok": False, "error": str(e), "time": 0.0}

    if not results["health"]["ok"]:
        results["text"] = {"ok": False, "error": "Skipped (server unreachable)"}
        results["vision"] = {"ok": False, "error": "Skipped (server unreachable)"}
        return results

    client = OpenAI(base_url=cfg["api_base"], api_key=cfg["api_key"], timeout=60)

    # 2. Text-only inference
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' and nothing else."},
            ],
            max_tokens=50,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content
        results["text"] = {"ok": True, "response": text, "time": elapsed,
                           "tokens": resp.usage.completion_tokens}
    except Exception as e:
        results["text"] = {"ok": False, "error": str(e), "time": 0.0}

    # 3. Vision inference (tiny synthetic image)
    try:
        img = Image.new("RGB", (64, 64), color="red")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        data_url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

        t0 = time.time()
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? One word."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=30,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        text = resp.choices[0].message.content
        results["vision"] = {"ok": True, "response": text, "time": elapsed,
                             "tokens": resp.usage.completion_tokens}
    except Exception as e:
        results["vision"] = {"ok": False, "error": str(e), "time": 0.0}

    return results


def run_model_step(nav: NavGym, action_budget: int = 75) -> str:
    """Call the vLLM model once on the current NavGym state, then auto-move
    the agent toward the predicted target pixel. Returns the raw response."""
    cfg = DEFAULT_API_CONFIG
    agent = GPTAgent(
        api_key=cfg["api_key"],
        api_base=cfg["api_base"],
        api_version=cfg["api_version"],
        model=cfg["model"],
        system_prompt=cfg["system_prompt"],
        target_description=nav.target_description,
        drone_see_shape=nav.drone_view_shape,
        scale=nav.px_real_size,
        top_left=nav.top_left,
        compress_images=cfg["compress_images"],
    )

    cur_px = nav._get_px(nav.cur_pos)
    t0 = time.time()
    response_str = agent.act(
        cur_whole_map=nav.cur_whole_map,
        cur_rgb_drone=nav.cur_rgb_drone,
        cur_position=cur_px,
    )
    elapsed = time.time() - t0

    # Parse & scale coordinates back to original image space
    bbox_resized = _parse_bbox(response_str)
    loc_resized = _parse_location(response_str)
    bbox_orig = agent.scale_coordinates_to_original(bbox_resized) if bbox_resized else None
    loc_orig = agent.scale_coordinates_to_original(loc_resized) if loc_resized else None

    scale_x, scale_y = agent.image_scale_factor
    scaled_pos = [int(cur_px[0] * scale_x), int(cur_px[1] * scale_y)]
    model_input_text = (
        f"[System]\n{cfg['system_prompt']}\n\n"
        f"[User Prompt]\n{get_prompt(nav.target_description, scaled_pos)}\n\n"
        f"[Images]\n"
        f"  Map:   {nav.cur_whole_map}\n"
        f"  Drone: {nav.cur_rgb_drone}\n"
        f"  Image scale factor: x={scale_x:.4f}, y={scale_y:.4f}\n"
        f"  Original cur_px: {cur_px}  ->  Scaled: {scaled_pos}"
    )

    st.session_state.model_input = model_input_text
    st.session_state.model_response = response_str
    st.session_state.model_landmark_bbox = bbox_orig
    st.session_state.model_pred_px = loc_orig
    st.session_state.model_elapsed = elapsed
    st.session_state.model_step_count += 1

    # Auto-move toward predicted pixel (same logic as auto_go_to_clicked_point)
    if loc_orig and loc_orig != [0, 0]:
        px_x, px_y = loc_orig
        world_x, world_y = nav.raster.xy(px_y, px_x)
        dst = Point3D(world_x, world_y, nav.cur_pos.z)
        for _ in range(action_budget):
            action = lookahead_discrete_action(nav.cur_pos, [dst])
            if action.name == "STOP":
                break
            if action.name == "MOVE_FORWARD":
                nav.step(ACTION_TO_ID["Move Forward"])
            elif action.name == "TURN_LEFT":
                nav.step(ACTION_TO_ID["Turn Left"])
            elif action.name == "TURN_RIGHT":
                nav.step(ACTION_TO_ID["Turn Right"])
            else:
                break

    return response_str


def finalize_episode() -> None:
    nav = st.session_state.navgym
    split = st.session_state.active_split
    if nav is None or st.session_state.episode_id is None:
        return
    st.session_state.trajectories[split][st.session_state.episode_id] = list(nav.trajectory)
    st.session_state.show_gt_overlay = True


def overlay_map_points(nav: NavGym, show_gt: bool = True) -> Image.Image:
    image = Image.open(nav.cur_whole_map).convert("RGB")
    draw = ImageDraw.Draw(image)

    # _get_px returns [col, row] == [x_pixel, y_pixel], matching both cv2 and PIL (x, y) convention
    cur_x, cur_y = nav._get_px(nav.cur_pos)
    tgt_x, tgt_y = nav.target_px

    r_cur = 14
    r_tgt = 28
    draw.ellipse((cur_x - r_cur, cur_y - r_cur, cur_x + r_cur, cur_y + r_cur), outline=(0, 255, 0), width=4)
    draw.ellipse((tgt_x - r_tgt, tgt_y - r_tgt, tgt_x + r_tgt, tgt_y + r_tgt), outline=(255, 0, 0), width=16)

    if show_gt:
        gt_px = [nav._get_px(p) for p in nav.episode.teacher_trajectory]
        if len(gt_px) >= 2:
            line_points = [(px[0], px[1]) for px in gt_px]
            draw.line(line_points, fill=(30, 144, 255), width=3)
        for px_x, px_y in gt_px:
            draw.ellipse((px_x - 3, px_y - 3, px_x + 3, px_y + 3), fill=(30, 144, 255))

        # Success region follows eval_planning_metrics threshold: dist <= 20m.
        rx = max(1, int(20 / nav.px_real_size[0]))
        ry = max(1, int(20 / nav.px_real_size[1]))
        draw.ellipse((tgt_x - rx, tgt_y - ry, tgt_x + rx, tgt_y + ry), outline=(255, 165, 0), width=16)

    # Model prediction overlays
    pred = st.session_state.model_pred_px
    if pred is not None:
        px, py = pred
        r_pred = 14
        draw.ellipse((px - r_pred, py - r_pred, px + r_pred, py + r_pred), fill=(255, 0, 255), outline=(255, 255, 255), width=2)
    bbox = st.session_state.model_landmark_bbox
    if bbox is not None:
        draw.rectangle(bbox, outline=(255, 255, 0), width=3)

    if st.session_state.clicked_px is not None:
        click_x, click_y = st.session_state.clicked_px
        draw.ellipse((click_x - 6, click_y - 6, click_x + 6, click_y + 6), outline=(255, 255, 0), width=3)
    return image


def _busy_run(action_fn) -> None:
    if st.session_state.is_busy:
        return
    st.session_state.is_busy = True
    try:
        action_fn()
    finally:
        st.session_state.is_busy = False


def auto_go_to_clicked_point(nav: NavGym, click_x: int, click_y: int, max_actions: int = 300) -> int:
    world_x, world_y = nav.raster.xy(click_y, click_x)
    dst = Point3D(world_x, world_y, nav.cur_pos.z)
    steps = 0
    while steps < max_actions:
        action = lookahead_discrete_action(nav.cur_pos, [dst])
        if action.name == "STOP":
            break
        if action.name == "MOVE_FORWARD":
            nav.step(ACTION_TO_ID["Move Forward"])
        elif action.name == "TURN_LEFT":
            nav.step(ACTION_TO_ID["Turn Left"])
        elif action.name == "TURN_RIGHT":
            nav.step(ACTION_TO_ID["Turn Right"])
        else:
            break
        steps += 1
    return steps


def main() -> None:
    st.set_page_config(page_title="Human-in-the-loop CityNav Eval", layout="wide")
    _default_state()

    st.title("Human-in-the-loop CityNav Evaluator")
    st.caption("You act as the agent, then evaluate with NE / SR / OSR / SPL.")

    with st.sidebar:
        st.subheader("Dataset")
        split = st.selectbox("Split", SPLITS, index=SPLITS.index(st.session_state.active_split))
        if split != st.session_state.active_split:
            st.session_state.active_split = split
            st.session_state.episode_index = 0
            st.session_state.navgym = None
            st.session_state.episode_id = None

        citynav = load_citynav_data(st.session_state.active_split)
        max_idx = max(0, len(citynav) - 1)
        idx = st.number_input(
            "Episode index",
            min_value=0,
            max_value=max_idx,
            value=min(st.session_state.episode_index, max_idx),
            step=1,
        )
        st.session_state.episode_index = idx

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Load Episode", use_container_width=True):
                load_episode(st.session_state.active_split, st.session_state.episode_index)
        with c2:
            if st.button("Reset Episode", use_container_width=True):
                load_episode(st.session_state.active_split, st.session_state.episode_index)

        st.divider()
        st.subheader("vLLM Server")
        DEFAULT_API_CONFIG["api_base"] = st.text_input(
            "API base URL", value=DEFAULT_API_CONFIG["api_base"]
        )
        DEFAULT_API_CONFIG["model"] = st.text_input(
            "Model name", value=DEFAULT_API_CONFIG["model"]
        )
        DEFAULT_API_CONFIG["compress_images"] = st.checkbox(
            "Compress images before inference", value=DEFAULT_API_CONFIG["compress_images"]
        )
        if st.button("Test vLLM Connection", use_container_width=True):
            with st.spinner("Testing vLLM server..."):
                st.session_state.vllm_test_results = _test_vllm_connection()

        tr = st.session_state.vllm_test_results
        if tr is not None:
            for name in ("health", "text", "vision"):
                r = tr[name]
                icon = "+" if r["ok"] else "-"
                label = f"{name.capitalize()}: {'OK' if r['ok'] else 'FAIL'}"
                if r["ok"] and "time" in r and r["time"] > 0:
                    label += f" ({r['time']:.1f}s)"
                with st.expander(label, expanded=not r["ok"]):
                    if r["ok"]:
                        if "response" in r:
                            st.code(r["response"], language=None)
                    else:
                        st.error(r.get("error", "Unknown error"))

    if st.session_state.navgym is None:
        st.info("Choose split/index in the sidebar, then click `Load Episode`.")
        return

    nav = st.session_state.navgym
    episode_id = st.session_state.episode_id
    map_img = overlay_map_points(nav, show_gt=st.session_state.show_gt_overlay)

    left, right = st.columns([1.15, 1.0])
    with left:
        st.markdown(
            f"<div style='font-size:24px; font-weight:700; margin-bottom:8px;'>"
            f"Target: {nav.target_description}</div>",
            unsafe_allow_html=True,
        )
        st.subheader("Map View")
        st.image(map_img, use_container_width=True)
        st.caption("Green=current, Red=target, Magenta=model prediction, Yellow box=landmark bbox, Blue=GT path (after save), Orange=SR radius")

    with right:
        st.subheader("Drone RGB View")
        st.image(nav.cur_rgb_drone, use_container_width=True)
        st.markdown(f"**Episode ID:** `{episode_id}`")
        st.markdown(f"**Steps taken:** {len(nav.trajectory) - 1}")
        cur_dist = nav.cur_pos.xy.dist_to(nav.episode.target_position.xy)
        st.markdown(f"**Current distance to target:** `{cur_dist:.2f}` m")
        if st.session_state.clicked_px is not None:
            st.markdown(f"**Selected map point:** `{st.session_state.clicked_px}`")

    st.divider()
    st.subheader("Control (you are the agent)")
    st.session_state.control_mode = st.radio(
        "Control mode",
        ["Manual actions", "Click map + auto-go"],
        index=0 if st.session_state.control_mode == "Manual actions" else 1,
        horizontal=True,
        disabled=st.session_state.is_busy,
    )

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button(
            "Move Forward",
            use_container_width=True,
            disabled=st.session_state.is_busy or st.session_state.control_mode != "Manual actions",
        ):
            _busy_run(lambda: nav.step(ACTION_TO_ID["Move Forward"]))
            st.rerun()
    with a2:
        if st.button(
            "Turn Left",
            use_container_width=True,
            disabled=st.session_state.is_busy or st.session_state.control_mode != "Manual actions",
        ):
            _busy_run(lambda: nav.step(ACTION_TO_ID["Turn Left"]))
            st.rerun()
    with a3:
        if st.button(
            "Turn Right",
            use_container_width=True,
            disabled=st.session_state.is_busy or st.session_state.control_mode != "Manual actions",
        ):
            _busy_run(lambda: nav.step(ACTION_TO_ID["Turn Right"]))
            st.rerun()
    with a4:
        if st.button(
            "Stop & Save Trajectory",
            use_container_width=True,
            disabled=st.session_state.is_busy,
        ):
            _busy_run(finalize_episode)
            st.success("Trajectory saved for this episode.")
            st.rerun()

    if st.session_state.control_mode == "Click map + auto-go":
        disabled_auto = st.session_state.is_busy or st.session_state.clicked_px is None
        if st.button("Auto-Go to selected map point", disabled=disabled_auto):
            click_x, click_y = st.session_state.clicked_px

            def _do_auto():
                auto_go_to_clicked_point(nav, click_x, click_y)

            _busy_run(_do_auto)
            st.rerun()

    # ---- vLLM Model Section ----
    st.divider()
    st.subheader("vLLM Model Test")

    vm1, vm2, vm3 = st.columns(3)
    with vm1:
        action_budget = st.number_input("Action budget per step", min_value=1, max_value=500, value=75, step=5)
    with vm2:
        total_steps = st.number_input("Total model steps", min_value=1, max_value=10, value=2, step=1)
    with vm3:
        st.metric("Completed model steps", st.session_state.model_step_count)

    mb1, mb2, mb3 = st.columns(3)
    with mb1:
        if st.button("Run 1 Model Step", use_container_width=True, disabled=st.session_state.is_busy):
            with st.spinner("Calling vLLM model..."):
                _busy_run(lambda: run_model_step(nav, action_budget=action_budget))
            st.rerun()
    with mb2:
        if st.button(f"Run All {total_steps} Steps", use_container_width=True, disabled=st.session_state.is_busy):
            remaining = total_steps - st.session_state.model_step_count
            if remaining <= 0:
                st.warning("All steps already completed. Reset episode to rerun.")
            else:
                with st.spinner(f"Running {remaining} model step(s)..."):
                    def _run_all():
                        for _ in range(remaining):
                            run_model_step(nav, action_budget=action_budget)
                    _busy_run(_run_all)
                st.rerun()
    with mb3:
        if st.button("Run All + Save", use_container_width=True, disabled=st.session_state.is_busy):
            remaining = total_steps - st.session_state.model_step_count
            def _run_and_save():
                for _ in range(max(remaining, 0)):
                    run_model_step(nav, action_budget=action_budget)
                finalize_episode()
            with st.spinner(f"Running model & saving..."):
                _busy_run(_run_and_save)
            st.success("Model trajectory saved.")
            st.rerun()

    if st.session_state.model_response is not None:
        st.markdown(f"**Inference time:** `{st.session_state.model_elapsed:.2f}s`")
        if st.session_state.model_pred_px:
            st.markdown(f"**Predicted target pixel:** `{st.session_state.model_pred_px}`")
        if st.session_state.model_landmark_bbox:
            st.markdown(f"**Landmark bbox:** `{st.session_state.model_landmark_bbox}`")
        cur_dist_now = nav.cur_pos.xy.dist_to(nav.episode.target_position.xy)
        st.markdown(f"**Distance to target after move:** `{cur_dist_now:.2f}` m")

        with st.expander("Model Input", expanded=False):
            st.text_area(
                "model_input_display",
                value=st.session_state.model_input or "",
                height=300,
                label_visibility="collapsed",
                key="ta_model_input",
            )

        with st.expander("Model Response (formatted)", expanded=True):
            st.markdown(_format_model_response(st.session_state.model_response))

        with st.expander("Model Response (raw / editable)", expanded=False):
            st.text_area(
                "model_response_display",
                value=st.session_state.model_response,
                height=400,
                label_visibility="collapsed",
                key="ta_model_response",
            )

    # ---- Navigation buttons ----
    st.divider()
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("Previous Episode", use_container_width=True, disabled=st.session_state.is_busy):
            st.session_state.episode_index = max(0, st.session_state.episode_index - 1)
            load_episode(st.session_state.active_split, st.session_state.episode_index)
            st.rerun()
    with col_next:
        if st.button("Next Episode", use_container_width=True, disabled=st.session_state.is_busy):
            st.session_state.episode_index = min(max_idx, st.session_state.episode_index + 1)
            load_episode(st.session_state.active_split, st.session_state.episode_index)
            st.rerun()

    st.divider()
    st.subheader("Metrics")

    split_metrics = {}
    split_counts = {}
    for s in SPLITS:
        cur_data = load_citynav_data(s)
        cur_traj = st.session_state.trajectories[s]
        done_eps = [ep for ep in cur_data.episodes if ep.id in cur_traj]
        split_counts[s] = len(done_eps)
        split_metrics[s] = eval_planning_metrics(done_eps, cur_traj) if done_eps else None

    active = st.session_state.active_split
    st.markdown(f"**Current split:** `{active}`")
    st.markdown(f"Completed episodes in split: `{split_counts[active]}`")

    if split_metrics[active] is not None:
        m = split_metrics[active]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NE", f"{m.mean_final_pos_to_goal_dist:.4f}")
        m2.metric("SR", f"{m.success_rate_final_pos_to_goal:.4f}")
        m3.metric("OSR", f"{m.success_rate_oracle_pos_to_goal:.4f}")
        m4.metric("SPL", f"{m.success_rate_weighted_by_path_length:.4f}")
    else:
        st.info("No saved trajectories yet for this split.")

    overall = weighted_metrics(split_metrics, split_counts)
    if overall is not None:
        ne, sr, osr, spl = overall
        st.markdown("**Weighted overall (over completed episodes across splits):**")
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("NE", f"{ne:.4f}")
        o2.metric("SR", f"{sr:.4f}")
        o3.metric("OSR", f"{osr:.4f}")
        o4.metric("SPL", f"{spl:.4f}")


if __name__ == "__main__":
    main()
