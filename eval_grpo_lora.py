"""
Evaluate GRPO LoRA model using vLLM's built-in LoRA support.

This script uses vLLM's native LoRA adapter loading capability, which allows
testing LoRA models without merging them into the base model.

This is an alternative to merging - faster to test different checkpoints.

Usage:
1. Start vLLM with LoRA support:
   CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve ./model_weight/Qwen2.5-VL-7B-Instruct \
     --dtype auto \
     --trust-remote-code \
     --served-model-name qwen_2_5_vl_7b \
     --host 0.0.0.0 \
     -tp 4 \
     --port 8989 \
     --enable-lora \
     --lora-modules grpo_lora=./experiment/FlightGPT/checkpoint-2379 \
     --limit-mm-per-prompt image=2,video=0 \
     --max-model-len=32000 \
     --max-lora-rank 64

2. Run evaluation:
   python eval_grpo_lora.py
"""

import os
import sys
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import time
import traceback
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.CityNavAgent import GPTAgent
from navgym.tools.EvalTools import eval_planning_metrics
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, view_area_corners
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
cropclient.load_image_cache()

# Model config - using LoRA adapter name
API_CONFIG = {
    "api_key": "EMPTY",
    "api_base": "http://0.0.0.0:8989/v1",
    "api_version": "2024-05-01-preview",
    "model": "grpo_lora",  # Use the LoRA adapter name specified in --lora-modules
    "system_prompt": "You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization."
}
SAVE_PATH = "./experiment/grpo_eval"


def create_dir(file_path):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

def initialize_agent(navGym):
    return GPTAgent(
        api_key=API_CONFIG["api_key"],
        api_base=API_CONFIG["api_base"],
        api_version=API_CONFIG["api_version"],
        model=API_CONFIG["model"],
        system_prompt=API_CONFIG["system_prompt"],
        target_description=navGym.target_description,
        drone_see_shape=navGym.drone_view_shape,
        scale=navGym.px_real_size,
        top_left=navGym.top_left
    )

def parse_bbox(result_str, key="landmark_bbox"):
    pattern = fr'"{key}"\s*:\s*\[(\d+), (\d+), (\d+), (\d+)\]'
    match = re.search(pattern, result_str)
    return list(map(int, match.groups())) if match else [0, 0, 0, 0]

def parse_location(result_str):
    match = re.search(r'"target_location"\s*:\s*\[(\d+), (\d+)\]', result_str)
    return list(map(int, match.groups())) if match else [0, 0]

def visualize_prediction(navGym, source_path, landmark_box, target_pred, true_target, save_path):
    image = cv2.imread(source_path)

    for landmark in navGym.map.landmark_map.landmarks:
        top_left = navGym._get_px(landmark.bbox_corners[0])
        bottom_right = navGym._get_px(landmark.bbox_corners[2])
        cv2.rectangle(image, top_left, bottom_right, color=(255, 0, 255), thickness=2)

    cv2.rectangle(image, (landmark_box[0], landmark_box[1]), (landmark_box[2], landmark_box[3]), (0, 0, 255), 2)
    cv2.circle(image, tuple(target_pred), radius=30, color=(0, 255, 0), thickness=-1)
    cv2.circle(image, tuple(true_target), radius=30, color=(255, 0, 0), thickness=-1)

    create_dir(save_path)
    cv2.imwrite(save_path, image)

def compute_pose(navGym, predicted_px, true_start_px, map_name):
    if predicted_px == [0, 0]:
        return navGym.start_pose

    dx, dy = predicted_px[0] - true_start_px[0], predicted_px[1] - true_start_px[1]
    world_x = dx / 10 + navGym.episode.start_pose.x
    world_y = navGym.episode.start_pose.y - dy / 10
    base_pose = Pose4D(world_x, world_y, 66.05, 0)

    corners = view_area_corners(base_pose, GROUND_LEVEL[map_name])
    depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
    center_depth = depth_img[45:55, 45:55].mean()
    refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
    return refined_pose

from gsamllavanav.space import Point2D, Point3D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose
def move(pose: Pose4D, dst: Pose4D, iterations: int):

    dst = Point3D(dst.x, dst.y, pose.z)
    trajectory = []
    for _ in range(iterations):
        action = lookahead_discrete_action(pose, [dst])
        if action.name == 'STOP':
            return trajectory
        pose = _moved_pose(pose, *action.value)
        trajectory.append(pose)
    return trajectory

def calculate_mean_metrics(results, nums):
    total_nums = nums['easy'] + nums['medium'] + nums['hard']
    NE = results['easy'].mean_final_pos_to_goal_dist * nums['easy']/total_nums + \
        results['medium'].mean_final_pos_to_goal_dist * nums['medium']/total_nums + \
        results['hard'].mean_final_pos_to_goal_dist * nums['hard']/total_nums

    SR = results['easy'].success_rate_final_pos_to_goal * nums['easy']/total_nums + \
        results['medium'].success_rate_final_pos_to_goal * nums['medium']/total_nums + \
        results['hard'].success_rate_final_pos_to_goal * nums['hard']/total_nums
        
    OSR = results['easy'].success_rate_oracle_pos_to_goal  * nums['easy']/total_nums + \
        results['medium'].success_rate_oracle_pos_to_goal  * nums['medium']/total_nums + \
        results['hard'].success_rate_oracle_pos_to_goal  * nums['hard']/total_nums

    SPL = results['easy'].success_rate_weighted_by_path_length  * nums['easy']/total_nums + \
        results['medium'].success_rate_weighted_by_path_length  * nums['medium']/total_nums + \
        results['hard'].success_rate_weighted_by_path_length  * nums['hard']/total_nums
    
    return NE, SR, OSR, SPL

def run_nav_gym(citynavData, split, step, action_num):
    trajectory = {}
    errors = []
    max_workers = 1

    def process_sample(i):
        sample_start = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"[Sample {i}/{len(citynavData)}] Starting...")
            
            pose_history = []
            cur_trajectory = []
            cur_citynavData = citynavData[i]
            
            for step_idx in range(step):
                print(f"[Sample {i}] Step {step_idx+1}/{step}")
                
                if pose_history != []:
                    cur_citynavData.episode.teacher_trajectory[0] = pose_history[-1]
                
                # Time NavGym creation
                navgym_start = time.time()
                navGym = NavGym(cur_citynavData)
                navgym_time = time.time() - navgym_start
                
                start_pose = navGym.start_pose
                map_name = navGym.episode.id[0]
                
                # Log image paths and check if they exist
                print(f"[Sample {i}] Map: {navGym.cur_whole_map}")
                print(f"[Sample {i}] Drone: {navGym.cur_rgb_drone}")
                
                if not os.path.exists(navGym.cur_whole_map):
                    raise FileNotFoundError(f"Map image not found: {navGym.cur_whole_map}")
                if not os.path.exists(navGym.cur_rgb_drone):
                    raise FileNotFoundError(f"Drone image not found: {navGym.cur_rgb_drone}")
                
                print(f"[Sample {i}] NavGym created in {navgym_time:.2f}s")
                
                # Initialize agent
                agent = initialize_agent(navGym)
                
                # Call agent.act with detailed timing
                print(f"[Sample {i}] Calling agent.act()...")
                act_start = time.time()
                
                result_str = agent.act(
                    cur_whole_map=navGym.cur_whole_map,
                    cur_rgb_drone=navGym.cur_rgb_drone,
                    cur_position=navGym._get_px(start_pose)
                )
                
                act_time = time.time() - act_start
                print(f"[Sample {i}] agent.act() completed in {act_time:.2f}s")
                print(f"[Sample {i}] Response length: {len(result_str)} chars")

                # Parse coordinates from model response (these are in resized image space)
                landmark_bbox_resized = parse_bbox(result_str, "landmark_bbox")
                target_pred_px_resized = parse_location(result_str)
                
                # Scale coordinates back to original image dimensions
                landmark_bbox = agent.scale_coordinates_to_original(landmark_bbox_resized)
                target_pred_px = agent.scale_coordinates_to_original(target_pred_px_resized)
                
                print(f"[Sample {i}] Predicted target (resized): {target_pred_px_resized}")
                print(f"[Sample {i}] Predicted target (original): {target_pred_px}")
                if landmark_bbox_resized != landmark_bbox:
                    print(f"[Sample {i}] Landmark bbox scaled: {landmark_bbox_resized} -> {landmark_bbox}")
                
                true_start_px = navGym.px_trajectory[0]
                true_target_px = navGym.target_px

                save_path = f"{SAVE_PATH}/visualized_image/{os.path.basename(navGym.cur_whole_map)}"
                # visualize_prediction(navGym, navGym.cur_whole_map, landmark_bbox, target_pred_px, true_target_px, save_path)

                pred_pose = compute_pose(navGym, target_pred_px, true_start_px, map_name)
                
                if pose_history == []:
                    cur_trajectory = [start_pose]
                    move_trajectory = move(start_pose, pred_pose, action_num)
                    if len(move_trajectory) > 0:
                        pose_history.append(move_trajectory[-1])
                    cur_trajectory.extend(move_trajectory)
                else:
                    move_trajectory = move(start_pose, pred_pose, action_num)
                    
                    if len(move_trajectory) > 0:
                        pose_history.append(move_trajectory[-1])
                    cur_trajectory.extend(move_trajectory)
            
            total_time = time.time() - sample_start
            print(f"[Sample {i}] ✓ SUCCESS in {total_time:.2f}s")
            print(f"{'='*60}\n")
            
            trajectory[citynavData.episodes[i].id] = cur_trajectory
            return citynavData.episodes[i].id, cur_trajectory, None

        except Exception as e:
            total_time = time.time() - sample_start
            print(f"[Sample {i}] ✗ FAILED after {total_time:.2f}s")
            print(f"[Sample {i}] Error: {e}")
            print(f"[Sample {i}] Error type: {type(e).__name__}")
            print(f"[Sample {i}] Traceback:")
            traceback.print_exc()
            print(f"{'='*60}\n")
            return None, None, i

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_sample, i) for i in range(len(citynavData))]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {split}"):
            traj_id, cur_trajectory, err_idx = future.result()
            if err_idx is not None:
                errors.append(err_idx)
            else:
                trajectory[traj_id] = cur_trajectory

    return trajectory, errors, SAVE_PATH 


def main():
    print("\n" + "🚀" * 30)
    print("FlightGPT GRPO LoRA Evaluation")
    print("🚀" * 30)
    print(f"vLLM Server: {API_CONFIG['api_base']}")
    print(f"Model (LoRA adapter): {API_CONFIG['model']}")
    print(f"Save Path: {SAVE_PATH}\n")
    
    results = {}
    nums = {}
    total_errors = {}
    step_num = 2    #total steps that agent take
    action_num = 75     #actions per step
    
    overall_start = time.time()
    
    # for split in ["hard"]:
    for split in ["easy", "medium", "hard"]:
        print("\n" + "=" * 60)
        print(f"Processing split: {split.upper()}")
        print("=" * 60)
        
        data_path = f"./data/citynav/citynav_val_unseen_{split}.json"
        citynavData = CityNavData(data_path)
        print(f"Total samples in {split}: {len(citynavData)}")

        split_start = time.time()
        traj, errors, image_dir = run_nav_gym(citynavData, split, step_num, action_num)
        split_time = time.time() - split_start
        
        print(f"\n{'='*60}")
        print(f"{split.upper()} Split Summary:")
        print(f"  Total samples: {len(citynavData)}")
        print(f"  Successful: {len(traj)}")
        print(f"  Failed: {len(errors)}")
        print(f"  Success rate: {len(traj)/len(citynavData)*100:.1f}%")
        print(f"  Time taken: {split_time/60:.1f} minutes")
        if len(traj) > 0:
            print(f"  Avg time per sample: {split_time/len(citynavData):.1f}s")
        print(f"  Errors: {errors}")
        print(f"{'='*60}\n")

        episodes = [ep for ep in citynavData.episodes if ep.id in traj]
        if len(episodes) > 0:
            metrics = eval_planning_metrics(episodes, traj)
            print(f"{split} result:", metrics)
            results[split] = metrics
            nums[split] = len(episodes)
        else:
            print(f"⚠ WARNING: No successful episodes for {split} split!")
            results[split] = None
            nums[split] = 0
        
        total_errors[split] = len(errors)
    
    overall_time = time.time() - overall_start
    
    print("\n" + "🎯" * 30)
    print("FINAL RESULTS")
    print("🎯" * 30)
    
    if all(results.values()):
        NE, SR, OSR, SPL = calculate_mean_metrics(results, nums)
        print(f"\nMetrics:")
        print(f"  NE (Navigation Error): {NE:.4f}")
        print(f"  SR (Success Rate): {SR:.4f}")
        print(f"  OSR (Oracle Success Rate): {OSR:.4f}")
        print(f"  SPL (Success weighted by Path Length): {SPL:.4f}")
    else:
        print("\n⚠ WARNING: Some splits failed completely, cannot calculate metrics")
    
    print(f"\nExecution Summary:")
    print(f"  Total time: {overall_time/60:.1f} minutes")
    print(f"  Total errors: {sum(total_errors.values())}")
    for split, err_count in total_errors.items():
        print(f"    {split}: {err_count} errors")
    
    print("\n" + "✓" * 60)

if __name__ == "__main__":
    main()
