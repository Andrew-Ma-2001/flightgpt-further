import os
import sys
import re
import gc
import argparse
import cv2
import json
import hashlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import pickle
import time
import traceback
import tracemalloc
import psutil
from openai import APITimeoutError
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.CityNavAgent import GPTAgent, get_prompt
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


_MEM_DEBUG = os.environ.get("MEM_DEBUG", "0") == "1"
_MEM_TRACE_EVERY = int(os.environ.get("MEM_TRACE_EVERY", "100"))
_MEM_REPORT_EVERY = int(os.environ.get("MEM_REPORT_EVERY", "20"))
_PROCESS = psutil.Process(os.getpid())
_MEM_SNAPSHOT_START = None


def _rss_mb():
    return _PROCESS.memory_info().rss / 1024 / 1024


def mem_report(tag, extra=None, force_gc=True):
    if not _MEM_DEBUG:
        return
    if force_gc:
        gc.collect()
    msg = f"[MEM] {tag} RSS={_rss_mb():.2f} MB"
    if extra:
        msg += f" | {extra}"
    print(msg, flush=True)


def mem_trace_report(i):
    if not _MEM_DEBUG or _MEM_SNAPSHOT_START is None:
        return
    if i <= 0 or i % _MEM_TRACE_EVERY != 0:
        return
    gc.collect()
    snapshot_now = tracemalloc.take_snapshot()
    stats = snapshot_now.compare_to(_MEM_SNAPSHOT_START, "lineno")
    print("\n[TRACEMALLOC] Top memory growth:", flush=True)
    for stat in stats[:20]:
        print(stat, flush=True)


def maybe_print_torch_mem(tag):
    if not _MEM_DEBUG:
        return
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            print(f"[MEM][CUDA] {tag} allocated={alloc:.2f}MB reserved={reserved:.2f}MB", flush=True)
    except Exception:
        return

def create_dir(file_path):
    dir_path = os.path.dirname(file_path)
    if dir_path:
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

def parse_args():
    parser = argparse.ArgumentParser(description="FlightGPT eval runner with memory diagnostics")
    parser.add_argument("--split", default="new", help="Dataset split name (default: new)")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N samples")
    parser.add_argument("--step-num", type=int, default=2, help="Total planning steps")
    parser.add_argument("--action-num", type=int, default=75, help="Max actions per planning step")
    parser.add_argument("--max-workers", type=int, default=1, help="Reserved concurrency arg for controlled experiments")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Generation max_tokens for vLLM")
    parser.add_argument("--stream", action="store_true", help="Use OpenAI streaming mode")
    parser.add_argument("--dry-run", action="store_true", help="Build sample/prompt only, skip vLLM request")
    parser.add_argument("--skip-image", action="store_true", help="Send text-only prompt without image")
    parser.add_argument("--compress-images", type=str, default=None, help="Override API_CONFIG compress_images true/false")
    parser.add_argument("--disable-checkpoint", action="store_true", help="Do not read/write pickle checkpoint")
    parser.add_argument("--output-jsonl", type=str, default=None, help="Stream eval records to this JSONL")
    parser.add_argument("--resume-jsonl", type=str, default=None, help="Read completed IDs from existing JSONL")
    parser.add_argument("--mem-report-every", type=int, default=20, help="Print memory every N samples")
    parser.add_argument("--mem-trace-every", type=int, default=100, help="Print tracemalloc diff every N samples")
    parser.add_argument("--print-env", action="store_true", help="Print environment/package diagnostics and exit")
    parser.add_argument("--deepcopy-arrays", action="store_true", help="Deepcopy RGB/depth arrays in CityNavData.__getitem__")
    return parser.parse_args()


def apply_sample_limit(citynavData, limit):
    if limit is None or limit <= 0:
        return
    if limit >= len(citynavData):
        return
    citynavData.episodes = citynavData.episodes[:limit]
    citynavData.maps = citynavData.maps[:limit]
    citynavData.data_len = limit


def hash_text(text):
    if text is None:
        return None
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_jsonl(path):
    if path is None or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_completed_ids_from_jsonl(path):
    completed = set()
    for row in iter_jsonl(path):
        if row.get("ok"):
            sid = row.get("id")
            if sid is not None:
                completed.add(sid)
    return completed


def append_jsonl(path, record):
    create_dir(path)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()


def print_env_diagnostics():
    print("\n[ENV] python executable:", sys.executable)
    print("[ENV] python version:", sys.version.replace("\n", " "))
    print("[ENV] uname:", os.uname())
    print("[ENV] cpu_count:", os.cpu_count())
    print("[ENV] MALLOC_ARENA_MAX:", os.environ.get("MALLOC_ARENA_MAX", "<unset>"))
    libs = [
        "openai", "httpx", "requests", "aiohttp", "pydantic",
        "PIL", "numpy", "torch", "transformers", "datasets", "vllm",
    ]
    for name in libs:
        try:
            mod = __import__(name)
            version = getattr(mod, "__version__", "<unknown>")
            print(f"[ENV] {name}=={version}")
        except Exception as e:
            print(f"[ENV] {name}=<unavailable> ({e})")

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

def run_nav_gym(citynavData, split, step, action_num, args):
    # Load existing checkpoint — only successful samples are saved
    trajectory = {} if args.disable_checkpoint else load_checkpoint(split)
    errors = []

    completed_ids = set(trajectory.keys())
    if args.resume_jsonl:
        completed_from_jsonl = load_completed_ids_from_jsonl(args.resume_jsonl)
        completed_ids.update(completed_from_jsonl)
        if completed_from_jsonl:
            print(f"  [Resume JSONL] Loaded {len(completed_from_jsonl)} completed IDs from {args.resume_jsonl}")

    output_jsonl = args.output_jsonl or os.path.join(SAVE_PATH, f"eval_records_{split}.jsonl")
    total = len(citynavData)
    skipped = 0

    mem_report("before_loop", extra=f"total={total} completed_ids={len(completed_ids)}")
    maybe_print_torch_mem("before_loop")

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
            prompt_preview_len = 0
            response_len = 0
            last_image_size = None
            last_image_bytes = 0
            last_base64_len = 0
            map_path_for_record = None
            
            for step_idx in range(step):
                print(f"[Sample {i}] Step {step_idx+1}/{step}")
                
                if pose_history != []:
                    cur_citynavData.episode.teacher_trajectory[0] = pose_history[-1]
                
                navgym_start = time.time()
                navGym = NavGym(cur_citynavData)
                navgym_time = time.time() - navgym_start
                
                start_pose = navGym.start_pose
                map_name = navGym.episode.id[0]
                map_path_for_record = navGym.cur_whole_map

                if os.path.exists(navGym.cur_whole_map):
                    last_image_bytes = os.path.getsize(navGym.cur_whole_map)
                    try:
                        with Image.open(navGym.cur_whole_map) as img:
                            last_image_size = img.size
                    except Exception:
                        last_image_size = None
                
                print(f"[Sample {i}] Map: {navGym.cur_whole_map}")
                print(f"[Sample {i}] Drone: {navGym.cur_rgb_drone}")
                
                if not os.path.exists(navGym.cur_whole_map):
                    raise FileNotFoundError(f"Map image not found: {navGym.cur_whole_map}")
                if not os.path.exists(navGym.cur_rgb_drone):
                    raise FileNotFoundError(f"Drone image not found: {navGym.cur_rgb_drone}")
                
                print(f"[Sample {i}] NavGym created in {navgym_time:.2f}s")
                
                agent = initialize_agent(navGym)
                prompt_preview_len = len(get_prompt(
                    instruction=navGym.target_description,
                    cur_pose=navGym._get_px(start_pose),
                ))
                
                print(f"[Sample {i}] Calling agent.act()...")
                act_start = time.time()

                if args.dry_run:
                    result_str = '{"landmark_bbox": [0, 0, 0, 0], "target_location": [0, 0]}'
                else:
                    result_str = agent.act(
                        cur_whole_map=navGym.cur_whole_map,
                        cur_rgb_drone=navGym.cur_rgb_drone,
                        cur_position=navGym._get_px(start_pose),
                        max_tokens=args.max_tokens,
                        stream=args.stream,
                        use_image=not args.skip_image,
                    )
                    last_base64_len = int(agent.last_debug.get("map_base64_len", 0))
                
                act_time = time.time() - act_start
                print(f"[Sample {i}] agent.act() completed in {act_time:.2f}s")
                print(f"[Sample {i}] Response length: {len(result_str)} chars")
                response_len = len(result_str)

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
                agent.close()
                del navGym, agent, move_trajectory
            
            total_time = time.time() - sample_start
            print(f"[Sample {i}] ✓ SUCCESS in {total_time:.2f}s")
            print(f"{'='*60}\n")

            # Save compact record immediately (streaming JSONL)
            record = {
                "id": episode_id,
                "ok": True,
                "split": split,
                "sample_index": i,
                "duration_s": total_time,
                "step_num": step,
                "action_num": action_num,
                "prediction_text_chars": response_len,
                "target_pred_px": target_pred_px,
                "image_path": map_path_for_record,
                "image_bytes": last_image_bytes,
                "image_size": last_image_size,
                "prompt_chars": prompt_preview_len,
                "base64_chars": last_base64_len,
                "target_desc_hash": hash_text(cur_citynavData.episode.target_description),
                "stream": bool(args.stream),
                "dry_run": bool(args.dry_run),
                "skip_image": bool(args.skip_image),
            }
            append_jsonl(output_jsonl, record)

            # Save to trajectory and checkpoint immediately
            trajectory[episode_id] = cur_trajectory
            completed_ids.add(episode_id)
            if not args.disable_checkpoint:
                save_checkpoint(split, trajectory)
                print(f"  [Checkpoint] Saved ({len(trajectory)}/{total} completed)")

            if _MEM_DEBUG and (i % max(1, args.mem_report_every) == 0):
                extra_items = [
                    f"results_len={len(trajectory)}",
                    f"prompt_chars={prompt_preview_len}",
                    f"pred_chars={response_len}",
                    f"base64_chars={last_base64_len}",
                    f"image_size={last_image_size}",
                    f"image_bytes={last_image_bytes}",
                ]
                mem_report(f"after sample {i}", " ".join(extra_items))
                maybe_print_torch_mem(f"after sample {i}")
            mem_trace_report(i)
            del record, result_str, cur_citynavData, cur_trajectory, pose_history, target_pred_px, landmark_bbox, target_pred_px_resized, landmark_bbox_resized
            if _MEM_DEBUG and i % max(1, args.mem_report_every) == 0:
                gc.collect()

        except APITimeoutError as e:
            total_time = time.time() - sample_start
            print(f"\n{'!'*60}")
            print(f"[Sample {i}] ✗ TIMEOUT after {total_time:.2f}s: {e}")
            print(f"vLLM 可能已经挂了，请检查后重新运行 python eval.py")
            print(f"已完成 {len(trajectory)}/{total} 个样本，进度已保存。")
            print(f"{'!'*60}\n")
            if not args.disable_checkpoint:
                save_checkpoint(split, trajectory)
            sys.exit(1)

        except Exception as e:
            total_time = time.time() - sample_start
            print(f"[Sample {i}] ✗ FAILED after {total_time:.2f}s: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            errors.append(i)
            append_jsonl(output_jsonl, {
                "id": episode_id,
                "ok": False,
                "split": split,
                "sample_index": i,
                "duration_s": total_time,
                "error_type": type(e).__name__,
                "error": str(e),
            })

    if skipped > 0:
        print(f"\n  [Checkpoint] Skipped {skipped} already-completed samples")

    return trajectory, errors, SAVE_PATH 




def main():
    global _MEM_TRACE_EVERY, _MEM_REPORT_EVERY, _MEM_SNAPSHOT_START
    args = parse_args()
    _MEM_TRACE_EVERY = max(1, args.mem_trace_every)
    _MEM_REPORT_EVERY = max(1, args.mem_report_every)
    if _MEM_DEBUG:
        tracemalloc.start(25)
        _MEM_SNAPSHOT_START = tracemalloc.take_snapshot()
        mem_report("start")

    if args.compress_images is not None:
        API_CONFIG["compress_images"] = args.compress_images.lower() in {"1", "true", "yes", "y", "on"}

    if args.print_env:
        print_env_diagnostics()
        return

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
    step_num = args.step_num
    action_num = args.action_num
    
    overall_start = time.time()
    
    # 用 new 来表示新数据集，修改上面 绝对路径以及 gsmllavanav 的 default path 路径
    for split in [args.split]:
    # for split in ["easy", "medium", "hard"]:
        print("\n" + "=" * 60)
        print(f"Processing split: {split.upper()}")
        print("=" * 60)
        
        if split == "new":
            data_path = f"{NEW_DATA_DIR}/citynav_val_unseen_new.json"
        else:
            data_path = f"{DATA_DIR}/citynav_val_unseen_{split}.json"
        
        citynavData = CityNavData(data_path, deepcopy_arrays_on_getitem=args.deepcopy_arrays)
        apply_sample_limit(citynavData, args.limit)
        total_samples = len(citynavData)
        print(f"Total samples in {split}: {total_samples}")
        print(f"max_workers={args.max_workers} (current eval loop is sequential; use 1 for lowest memory pressure)")
        print(f"stream={args.stream}, dry_run={args.dry_run}, skip_image={args.skip_image}, max_tokens={args.max_tokens}")
        print(f"deepcopy_arrays_on_getitem={args.deepcopy_arrays}")

        # Only use the first N samples for quick testing.
        # test_sample_limit = 100
        # if test_sample_limit is not None and test_sample_limit < total_samples:
        #     citynavData.episodes = citynavData.episodes[:test_sample_limit]
        #     citynavData.maps = citynavData.maps[:test_sample_limit]
        #     citynavData.data_len = test_sample_limit
        #     print(f"Using first {len(citynavData)} samples for test run.")

        split_start = time.time()
        traj, errors, image_dir = run_nav_gym(citynavData, split, step_num, action_num, args)
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

        if len(traj) > 0:
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
        else:
            print(f"⚠ WARNING: Empty trajectory for {split} split!")
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
        final_results = {"per_split": {s: {"num": nums[s]} for s in nums}}
        if {"easy", "medium", "hard"}.issubset(results.keys()):
            NE, SR, OSR, SPL = calculate_mean_metrics(results, nums)
            print(f"\nMetrics:")
            print(f"  NE (Navigation Error): {NE:.4f}")
            print(f"  SR (Success Rate): {SR:.4f}")
            print(f"  OSR (Oracle Success Rate): {OSR:.4f}")
            print(f"  SPL (Success weighted by Path Length): {SPL:.4f}")
            final_results.update({"NE": NE, "SR": SR, "OSR": OSR, "SPL": SPL})

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
    mem_report("end", extra=f"overall_minutes={overall_time/60:.2f}", force_gc=True)
    maybe_print_torch_mem("end")

if __name__ == "__main__":
    main()
