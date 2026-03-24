import argparse
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.CityNavAgent import GPTAgent, get_prompt
from navgym.tools.EvalTools import eval_planning_metrics
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, Point3D, view_area_corners
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose

cropclient.load_image_cache()

# Model config (match eval.py defaults)
API_CONFIG = {
    "api_key": "EMPTY",
    "api_base": "http://0.0.0.0:8989/v1",
    "api_version": "2024-05-01-preview",
    "model": "qwen_2_5_vl_7b",
    "system_prompt": (
        "You are an intelligent autonomous aerial vehicle (UAV) equipped for "
        "real-world navigation and visual target localization."
    ),
}


@dataclass
class StepArtifacts:
    step_index: int
    map_path: str
    drone_path: str
    prompt_path: str
    response_path: str
    parsed_path: str
    actions_path: str
    composite_path: str
    map_overlay_path: str
    drone_overlay_path: str


def create_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def initialize_agent(nav_gym: NavGym) -> GPTAgent:
    return GPTAgent(
        api_key=API_CONFIG["api_key"],
        api_base=API_CONFIG["api_base"],
        api_version=API_CONFIG["api_version"],
        model=API_CONFIG["model"],
        system_prompt=API_CONFIG["system_prompt"],
        target_description=nav_gym.target_description,
        drone_see_shape=nav_gym.drone_view_shape,
        scale=nav_gym.px_real_size,
        top_left=nav_gym.top_left,
    )


def parse_bbox(result_str: str, key: str = "landmark_bbox") -> list[int]:
    import re

    pattern = fr'"{key}"\s*:\s*\[(\d+), (\d+), (\d+), (\d+)\]'
    match = re.search(pattern, result_str)
    return list(map(int, match.groups())) if match else [0, 0, 0, 0]


def parse_location(result_str: str) -> list[int]:
    import re

    match = re.search(r'"target_location"\s*:\s*\[(\d+), (\d+)\]', result_str)
    return list(map(int, match.groups())) if match else [0, 0]


def compute_pose(nav_gym: NavGym, predicted_px: list[int], true_start_px: list[int], map_name: str) -> Pose4D:
    if predicted_px == [0, 0]:
        return nav_gym.start_pose

    dx, dy = predicted_px[0] - true_start_px[0], predicted_px[1] - true_start_px[1]
    world_x = dx / 10 + nav_gym.episode.start_pose.x
    world_y = nav_gym.episode.start_pose.y - dy / 10
    base_pose = Pose4D(world_x, world_y, 66.05, 0)

    _ = view_area_corners(base_pose, GROUND_LEVEL[map_name])
    depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
    center_depth = depth_img[45:55, 45:55].mean()
    refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
    return refined_pose


def move_with_actions(pose: Pose4D, dst: Pose4D, iterations: int):
    dst_point = Point3D(dst.x, dst.y, pose.z)
    trajectory = []
    actions = []
    for _ in range(iterations):
        action = lookahead_discrete_action(pose, [dst_point])
        actions.append({"name": action.name, "value": action.value})
        if action.name == "STOP":
            break
        pose = _moved_pose(pose, *action.value)
        trajectory.append(pose)
    return trajectory, actions


def draw_overlays(
    nav_gym: NavGym,
    map_path: str,
    drone_path: str,
    landmark_bbox: list[int],
    target_pred: list[int],
    true_target: list[int],
    step_index: int,
    actions: list[dict],
    save_dir: str,
) -> tuple[str, str, str]:
    create_dir(save_dir)
    map_img = cv2.imread(map_path)
    drone_img = cv2.imread(drone_path)

    if map_img is None:
        raise FileNotFoundError(f"Map image not found: {map_path}")
    if drone_img is None:
        raise FileNotFoundError(f"Drone image not found: {drone_path}")

    # Draw predicted landmark bbox and target points
    cv2.rectangle(
        map_img,
        (landmark_bbox[0], landmark_bbox[1]),
        (landmark_bbox[2], landmark_bbox[3]),
        (0, 0, 255),
        2,
    )
    cv2.circle(map_img, tuple(target_pred), radius=18, color=(0, 255, 0), thickness=-1)
    cv2.circle(map_img, tuple(true_target), radius=18, color=(255, 0, 0), thickness=-1)

    # Overlay text
    header = f"Step {step_index + 1}"
    action_names = [a["name"] for a in actions]
    action_text = ", ".join(action_names)
    if len(action_text) > 120:
        action_text = action_text[:117] + "..."

    cv2.putText(map_img, header, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    cv2.putText(map_img, "Pred: green / True: blue", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
    cv2.putText(map_img, f"Actions: {action_text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

    map_overlay_path = os.path.join(save_dir, "map_overlay.jpg")
    drone_overlay_path = os.path.join(save_dir, "drone_overlay.jpg")
    composite_path = os.path.join(save_dir, "composite.jpg")

    cv2.imwrite(map_overlay_path, map_img)
    cv2.imwrite(drone_overlay_path, drone_img)

    # Composite frame: map + drone
    target_height = map_img.shape[0]
    drone_resized = cv2.resize(drone_img, (int(drone_img.shape[1] * target_height / drone_img.shape[0]), target_height))
    composite = cv2.hconcat([map_img, drone_resized])
    cv2.imwrite(composite_path, composite)

    return map_overlay_path, drone_overlay_path, composite_path


def write_video(frames: list[str], output_path: str, fps: int = 1) -> None:
    if not frames:
        return

    first = cv2.imread(frames[0])
    if first is None:
        return

    height, width = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)
    writer.release()


def run_visualization(
    split: str,
    sample_index: int,
    step_num: int,
    action_num: int,
    output_dir: str,
    data_path: str | None = None,
) -> None:
    if data_path is None:
        data_path = f"./data/citynav/citynav_val_unseen_{split}.json"

    create_dir(output_dir)
    citynav_data = CityNavData(data_path)
    if sample_index < 0 or sample_index >= len(citynav_data):
        raise IndexError(f"sample_index {sample_index} is out of range (0..{len(citynav_data)-1})")

    cur_citynav_data = citynav_data[sample_index]
    sample_id = citynav_data.episodes[sample_index].id

    step_artifacts: list[StepArtifacts] = []
    pose_history = []
    full_trajectory = []
    raw_steps = []

    for step_idx in range(step_num):
        if pose_history:
            cur_citynav_data.episode.teacher_trajectory[0] = pose_history[-1]

        nav_gym = NavGym(cur_citynav_data)
        agent = initialize_agent(nav_gym)

        # Log prompt and inputs
        prompt_text = get_prompt(nav_gym.target_description, nav_gym._get_px(nav_gym.start_pose))
        step_dir = os.path.join(output_dir, f"step_{step_idx + 1:02d}")
        create_dir(step_dir)

        prompt_path = os.path.join(step_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        # Copy input images for reference
        map_path = os.path.join(step_dir, "input_map.jpg")
        drone_path = os.path.join(step_dir, "input_drone.jpg")
        shutil.copy(nav_gym.cur_whole_map, map_path)
        shutil.copy(nav_gym.cur_rgb_drone, drone_path)

        # Call model
        start_time = time.time()
        result_str = agent.act(
            cur_whole_map=nav_gym.cur_whole_map,
            cur_rgb_drone=nav_gym.cur_rgb_drone,
            cur_position=nav_gym._get_px(nav_gym.start_pose),
        )
        act_time = time.time() - start_time

        response_path = os.path.join(step_dir, "response.txt")
        with open(response_path, "w", encoding="utf-8") as f:
            f.write(result_str)

        # Parse outputs
        landmark_bbox_resized = parse_bbox(result_str, "landmark_bbox")
        target_pred_px_resized = parse_location(result_str)
        landmark_bbox = agent.scale_coordinates_to_original(landmark_bbox_resized)
        target_pred_px = agent.scale_coordinates_to_original(target_pred_px_resized)

        true_start_px = nav_gym.px_trajectory[0]
        true_target_px = nav_gym.target_px
        pred_pose = compute_pose(nav_gym, target_pred_px, true_start_px, nav_gym.episode.id[0])

        move_trajectory, actions = move_with_actions(nav_gym.start_pose, pred_pose, action_num)

        if not full_trajectory:
            full_trajectory = [nav_gym.start_pose]
        full_trajectory.extend(move_trajectory)

        if move_trajectory:
            pose_history.append(move_trajectory[-1])

        parsed_path = os.path.join(step_dir, "parsed.json")
        parsed_payload = {
            "step_index": step_idx,
            "sample_id": sample_id,
            "map_name": nav_gym.episode.id[0],
            "act_time_sec": round(act_time, 3),
            "input_map": nav_gym.cur_whole_map,
            "input_drone": nav_gym.cur_rgb_drone,
            "prompt_path": prompt_path,
            "response_path": response_path,
            "landmark_bbox_resized": landmark_bbox_resized,
            "landmark_bbox": landmark_bbox,
            "target_pred_px_resized": target_pred_px_resized,
            "target_pred_px": target_pred_px,
            "true_start_px": true_start_px,
            "true_target_px": true_target_px,
            "pred_pose": [pred_pose.x, pred_pose.y, pred_pose.z, pred_pose.yaw],
            "actions_count": len(actions),
        }
        with open(parsed_path, "w", encoding="utf-8") as f:
            json.dump(parsed_payload, f, indent=2)

        actions_path = os.path.join(step_dir, "actions.json")
        with open(actions_path, "w", encoding="utf-8") as f:
            json.dump(actions, f, indent=2)

        map_overlay_path, drone_overlay_path, composite_path = draw_overlays(
            nav_gym,
            nav_gym.cur_whole_map,
            nav_gym.cur_rgb_drone,
            landmark_bbox,
            target_pred_px,
            true_target_px,
            step_idx,
            actions,
            step_dir,
        )

        step_artifacts.append(
            StepArtifacts(
                step_index=step_idx,
                map_path=map_path,
                drone_path=drone_path,
                prompt_path=prompt_path,
                response_path=response_path,
                parsed_path=parsed_path,
                actions_path=actions_path,
                composite_path=composite_path,
                map_overlay_path=map_overlay_path,
                drone_overlay_path=drone_overlay_path,
            )
        )

        raw_steps.append(
            {
                "step_index": step_idx,
                "actions_count": len(actions),
                "landmark_bbox": landmark_bbox,
                "target_pred_px": target_pred_px,
                "true_target_px": true_target_px,
            }
        )

    # Compute metrics for this single episode
    traj_dict = {sample_id: full_trajectory}
    metrics = eval_planning_metrics([citynav_data.episodes[sample_index]], traj_dict)

    summary = {
        "sample_id": sample_id,
        "split": split,
        "sample_index": sample_index,
        "step_num": step_num,
        "action_num": action_num,
        "metrics": {
            "mean_final_pos_to_goal_dist": metrics.mean_final_pos_to_goal_dist,
            "success_rate_final_pos_to_goal": metrics.success_rate_final_pos_to_goal,
            "success_rate_oracle_pos_to_goal": metrics.success_rate_oracle_pos_to_goal,
            "success_rate_weighted_by_path_length": metrics.success_rate_weighted_by_path_length,
        },
        "steps": raw_steps,
        "artifacts": [asdict(x) for x in step_artifacts],
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write videos from composite frames
    composite_frames = [x.composite_path for x in step_artifacts]
    write_video(composite_frames, os.path.join(output_dir, "steps_composite.mp4"), fps=1)

    print(f"Saved visualization to: {output_dir}")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize eval process for one sample.")
    parser.add_argument("--split", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--index", type=int, default=0, help="Sample index in split file.")
    parser.add_argument("--steps", type=int, default=2, help="Total steps for the agent.")
    parser.add_argument("--actions", type=int, default=75, help="Actions per step.")
    parser.add_argument("--output", default="./experiment/visualize_eval", help="Output directory.")
    parser.add_argument("--data-path", default=None, help="Override CityNav JSON path.")
    args = parser.parse_args()

    run_visualization(
        split=args.split,
        sample_index=args.index,
        step_num=args.steps,
        action_num=args.actions,
        output_dir=args.output,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
