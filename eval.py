import os
import sys
import re
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import time
import traceback
from openai import APITimeoutError
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.CityNavAgent import GPTAgent
from navgym.tools.EvalTools import eval_planning_metrics
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, view_area_corners
from gsamllavanav.space import Point2D, Point3D, Pose4D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose
from gsamllavanav.defaultpaths import CITYREFER_DATA_DIR, MTURK_TRAJECTORY_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
cropclient.load_image_cache()

# model config
API_CONFIG = {
    "api_key": "EMPTY",
    "api_base": "http://0.0.0.0:8989/v1",   #use your port
    "api_version": "2024-05-01-preview",
    "model": "qwen_2_5_vl_7b",
    "system_prompt": "You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization.",
    # Whether to compress/resize images before VLLM inference.
    # Keep True to reduce throughput pressure; set False to disable compression.
    "compress_images": True,
}
SAVE_PATH = "./experiment"

NEW_DATA_DIR = "/home/yjy/flightgpt/FlightGPT/refine_citynav/processed_citynav"

DATA_DIR = MTURK_TRAJECTORY_DIR
CITYREFER_DIR = CITYREFER_DATA_DIR

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
        top_left=navGym.top_left,
        compress_images=API_CONFIG["compress_images"],
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
    # world_x = dx / 10 + navGym.episode.start_pose.x
    # world_y = navGym.episode.start_pose.y - dy / 10
    # Convert pixel displacement into meters using raster metadata to stay
    # resolution-invariant (e.g., 4000px vs 2048px maps).
    meter_per_px_x, meter_per_px_y = navGym.px_real_size
    world_x = dx * meter_per_px_x + navGym.episode.start_pose.x
    world_y = navGym.episode.start_pose.y - dy * meter_per_px_y 
    base_pose = Pose4D(world_x, world_y, 66.05, 0)

    corners = view_area_corners(base_pose, GROUND_LEVEL[map_name])
    depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
    center_depth = depth_img[45:55, 45:55].mean()
    refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
    return refined_pose

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


# ============ Checkpoint utilities ============

def get_checkpoint_path(split):
    """Get the pickle checkpoint path for a given split."""
    os.makedirs(SAVE_PATH, exist_ok=True)
    return os.path.join(SAVE_PATH, f"checkpoint_{split}.pkl")

def load_checkpoint(split):
    """Load existing checkpoint. Returns trajectory dict or empty dict."""
    ckpt_path = get_checkpoint_path(split)
    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            data = pickle.load(f)
        print(f"  [Checkpoint] Loaded {len(data)} completed samples from {ckpt_path}")
        return data
    return {}

def save_checkpoint(split, trajectory):
    """Save trajectory dict to checkpoint file."""
    ckpt_path = get_checkpoint_path(split)
    with open(ckpt_path, "wb") as f:
        pickle.dump(trajectory, f)


# ============ Main evaluation loop ============

def run_nav_gym(citynavData, split, step, action_num):
    # Load existing checkpoint — only successful samples are saved
    trajectory = load_checkpoint(split)
    errors = []

    completed_ids = set(trajectory.keys())
    total = len(citynavData)
    skipped = 0

    for i in tqdm(range(total), desc=f"Running {split}"):
        episode_id = citynavData.episodes[i].id

        # Skip already completed samples
        if episode_id in completed_ids:
            skipped += 1
            continue

        sample_start = time.time()
        try:
            print(f"\n{'='*60}")
            print(f"[Sample {i}/{total}] id={episode_id}  Starting...")
            
            pose_history = []
            cur_trajectory = []
            cur_citynavData = citynavData[i]
            
            for step_idx in range(step):
                print(f"[Sample {i}] Step {step_idx+1}/{step}")
                
                if pose_history != []:
                    cur_citynavData.episode.teacher_trajectory[0] = pose_history[-1]
                
                navgym_start = time.time()
                navGym = NavGym(cur_citynavData)
                navgym_time = time.time() - navgym_start
                
                start_pose = navGym.start_pose
                map_name = navGym.episode.id[0]
                
                print(f"[Sample {i}] Map: {navGym.cur_whole_map}")
                print(f"[Sample {i}] Drone: {navGym.cur_rgb_drone}")
                
                if not os.path.exists(navGym.cur_whole_map):
                    raise FileNotFoundError(f"Map image not found: {navGym.cur_whole_map}")
                if not os.path.exists(navGym.cur_rgb_drone):
                    raise FileNotFoundError(f"Drone image not found: {navGym.cur_rgb_drone}")
                
                print(f"[Sample {i}] NavGym created in {navgym_time:.2f}s")
                
                agent = initialize_agent(navGym)
                
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

                landmark_bbox_resized = parse_bbox(result_str, "landmark_bbox")
                target_pred_px_resized = parse_location(result_str)
                
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
            
            # Save to trajectory and checkpoint immediately
            trajectory[episode_id] = cur_trajectory
            save_checkpoint(split, trajectory)
            print(f"  [Checkpoint] Saved ({len(trajectory)}/{total} completed)")

        except APITimeoutError as e:
            total_time = time.time() - sample_start
            print(f"\n{'!'*60}")
            print(f"[Sample {i}] ✗ TIMEOUT after {total_time:.2f}s: {e}")
            print(f"vLLM 可能已经挂了，请检查后重新运行 python eval.py")
            print(f"已完成 {len(trajectory)}/{total} 个样本，进度已保存。")
            print(f"{'!'*60}\n")
            save_checkpoint(split, trajectory)
            sys.exit(1)

        except Exception as e:
            total_time = time.time() - sample_start
            print(f"[Sample {i}] ✗ FAILED after {total_time:.2f}s: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            errors.append(i)

    if skipped > 0:
        print(f"\n  [Checkpoint] Skipped {skipped} already-completed samples")

    return trajectory, errors, SAVE_PATH 




def main():
    print("\n" + "🚀" * 30)
    print("FlightGPT Evaluation Starting")
    print("🚀" * 30)
    print(f"vLLM Server: {API_CONFIG['api_base']}")
    print(f"Model: {API_CONFIG['model']}")
    print(f"Image Compression: {API_CONFIG['compress_images']}")
    print(f"Save Path: {SAVE_PATH}\n")
    
    print("⚠️ " + "=" * 56 + " ⚠️")
    print("⚠️  WARNING: Using custom dataset paths!")
    print(f"⚠️  Data Dir:     {DATA_DIR}")
    print(f"⚠️  CityRefer Dir: {CITYREFER_DIR}")
    print("⚠️ " + "=" * 56 + " ⚠️\n")
    
    results = {}
    nums = {}
    total_errors = {}
    step_num = 2    #total steps that agent take
    action_num = 75     #actions per step
    
    overall_start = time.time()
    
    # 用 new 来表示新数据集，修改上面 绝对路径以及 gsmllavanav 的 default path 路径
    for split in ["new"]:
    # for split in ["easy", "medium", "hard"]:
        print("\n" + "=" * 60)
        print(f"Processing split: {split.upper()}")
        print("=" * 60)
        
        if split == "new":
            data_path = f"{NEW_DATA_DIR}/citynav_val_unseen_new.json"
        else:
            data_path = f"{DATA_DIR}/citynav_val_unseen_{split}.json"
        
        citynavData = CityNavData(data_path)
        total_samples = len(citynavData)
        print(f"Total samples in {split}: {total_samples}")

        # Only use the first N samples for quick testing.
        # test_sample_limit = 100
        # if test_sample_limit is not None and test_sample_limit < total_samples:
        #     citynavData.episodes = citynavData.episodes[:test_sample_limit]
        #     citynavData.maps = citynavData.maps[:test_sample_limit]
        #     citynavData.data_len = test_sample_limit
        #     print(f"Using first {len(citynavData)} samples for test run.")

        split_start = time.time()
        traj, errors, image_dir = run_nav_gym(citynavData, split, step_num, action_num)
        split_time = time.time() - split_start
        
        print(f"\n{'='*60}")
        print(f"{split.upper()} Split Summary:")
        print(f"  Total samples: {len(citynavData)}")
        print(f"  Successful: {len(traj)}")
        print(f"  Failed (this run): {len(errors)}")
        print(f"  Success rate: {len(traj)/len(citynavData)*100:.1f}%")
        print(f"  Time taken: {split_time/60:.1f} minutes")
        if len(traj) > 0:
            print(f"  Avg time per sample: {split_time/len(citynavData):.1f}s")
        print(f"  Errors (this run): {errors}")
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

        # Check if all samples are done
        if len(traj) + len(errors) == len(citynavData) and len(errors) == 0:
            print(f"  ✅ All samples for {split} completed successfully!")
    
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
        
        # Save final results to JSON
        final_results = {
            "NE": NE, "SR": SR, "OSR": OSR, "SPL": SPL,
            "per_split": {s: {"num": nums[s]} for s in nums},
        }
        result_path = os.path.join(SAVE_PATH, "final_results.json")
        with open(result_path, "w") as f:
            json.dump(final_results, f, indent=2)
        print(f"\n  Results saved to {result_path}")
    else:
        print("\n⚠ WARNING: Some splits have no successful episodes, cannot calculate overall metrics")
        print("  Re-run to continue processing failed samples.")
    
    print(f"\nExecution Summary:")
    print(f"  Total time: {overall_time/60:.1f} minutes")
    print(f"  Total errors (this run): {sum(total_errors.values())}")
    for split, err_count in total_errors.items():
        print(f"    {split}: {err_count} errors")
    
    print("\n" + "✓" * 60)

if __name__ == "__main__":
    main()
