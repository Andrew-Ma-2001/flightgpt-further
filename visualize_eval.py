#!/usr/bin/env python3
"""
FlightGPT Evaluation Visualization Script

This script creates detailed visualizations of the evaluation process,
showing step-by-step what happens during navigation inference.

Usage:
    python visualize_eval.py --sample_idx 0 --split easy --output_dir ./visualization_output

The script will:
1. Run evaluation on a single sample
2. Visualize each step showing:
   - The map image with landmarks, UAV position, and drone view area
   - The model's prediction (target location, landmark bbox)
   - The ground truth target location
   - The model's reasoning
   - Trajectory so far
3. Create a summary panel with all metrics
4. Save as images and optionally create an animated GIF

Requires: vLLM server running at the configured endpoint
"""

import os
import sys
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
import argparse
import json
import time
import math
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple
import textwrap

# FlightGPT imports
from navgym.models.CityNavData import CityNavData
from navgym.models.NavGym import NavGym
from navgym.agents.CityNavAgent import GPTAgent, get_prompt
from navgym.tools.EvalTools import eval_planning_metrics
from gsamllavanav.observation import cropclient
from gsamllavanav.mapdata import GROUND_LEVEL
from gsamllavanav.space import Pose4D, Point2D
from gsamllavanav.teacher.algorithm.lookahead import lookahead_discrete_action
from gsamllavanav.teacher.trajectory import _moved_pose

# Suppress proxy environment variables
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
cropclient.load_image_cache()

# ============================================================================
# Configuration
# ============================================================================

API_CONFIG = {
    "api_key": "EMPTY",
    "api_base": "http://0.0.0.0:8989/v1",
    "api_version": "2024-05-01-preview",
    "model": "qwen_2_5_vl_7b",
    "system_prompt": "You are an intelligent autonomous aerial vehicle (UAV) equipped for real-world navigation and visual target localization."
}


@dataclass
class StepVisualization:
    """Container for step visualization data"""
    step_idx: int
    map_image_path: str
    drone_image_path: str
    prompt: str
    model_response: str
    thinking: str
    answer: str
    landmark_bbox_pred: List[int]
    target_pred_px: List[int]
    target_true_px: List[int]
    start_px: List[int]
    current_px: List[int]
    trajectory_px: List[List[int]]
    distance_to_target: float
    inference_time: float


# ============================================================================
# Parsing Utilities
# ============================================================================

def parse_bbox(result_str: str, key: str = "landmark_bbox") -> List[int]:
    """Parse bounding box from model response"""
    pattern = fr'"{key}"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, result_str)
    return list(map(int, match.groups())) if match else [0, 0, 0, 0]


def parse_location(result_str: str) -> List[int]:
    """Parse target location from model response"""
    match = re.search(r'"target_location"\s*:\s*\[(\d+),\s*(\d+)\]', result_str)
    return list(map(int, match.groups())) if match else [0, 0]


def parse_thinking(result_str: str) -> str:
    """Extract thinking from <think></think> tags"""
    match = re.search(r'<think>(.*?)</think>', result_str, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_answer(result_str: str) -> str:
    """Extract answer from <answer></answer> tags"""
    match = re.search(r'<answer>(.*?)</answer>', result_str, re.DOTALL)
    return match.group(1).strip() if match else ""


# ============================================================================
# Movement Functions
# ============================================================================

def move(pose: Pose4D, dst: Pose4D, iterations: int) -> List[Pose4D]:
    """Move drone towards destination using lookahead discrete actions"""
    from gsamllavanav.space import Point3D
    dst = Point3D(dst.x, dst.y, pose.z)
    trajectory = []
    for _ in range(iterations):
        action = lookahead_discrete_action(pose, [dst])
        if action.name == 'STOP':
            return trajectory
        pose = _moved_pose(pose, *action.value)
        trajectory.append(pose)
    return trajectory


def compute_pose(navGym, predicted_px, true_start_px, map_name):
    """Compute 3D pose from predicted pixel coordinates"""
    if predicted_px == [0, 0]:
        return navGym.start_pose

    dx, dy = predicted_px[0] - true_start_px[0], predicted_px[1] - true_start_px[1]
    world_x = dx / 10 + navGym.episode.start_pose.x
    world_y = navGym.episode.start_pose.y - dy / 10
    base_pose = Pose4D(world_x, world_y, 66.05, 0)

    try:
        depth_img = cropclient.crop_image(map_name, base_pose, (100, 100), "depth")
        center_depth = depth_img[45:55, 45:55].mean()
        refined_pose = Pose4D(base_pose.x, base_pose.y, base_pose.z - center_depth + 5, 0)
        return refined_pose
    except:
        return base_pose


# ============================================================================
# Visualization Functions
# ============================================================================

def create_color_palette():
    """Create a consistent color palette for visualization"""
    return {
        'pred_target': (0, 255, 0),       # Green - predicted target
        'true_target': (0, 0, 255),        # Red (BGR) - true target
        'landmark_bbox': (255, 165, 0),    # Orange - landmark bbox
        'trajectory': (255, 0, 255),       # Magenta - trajectory
        'current_pos': (0, 255, 255),      # Cyan - current position
        'start_pos': (255, 255, 0),        # Yellow - start position
        'drone_view': (0, 200, 200),       # Teal - drone view area
        'success': (50, 205, 50),          # LimeGreen
        'failure': (220, 20, 60),          # Crimson
    }


def draw_text_with_background(img, text, position, font_scale=0.7, thickness=2, 
                               bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Draw text with a semi-transparent background"""
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = position
    padding = 5
    
    # Draw background
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    return text_height + 2 * padding + 5


def create_step_visualization(step_data: StepVisualization, output_path: str, colors: dict):
    """Create a detailed visualization image for a single step"""
    
    # Load the map image
    map_img = cv2.imread(step_data.map_image_path)
    if map_img is None:
        print(f"Warning: Could not load map image from {step_data.map_image_path}")
        return
    
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = map_img.shape[:2]
    
    # Create a large canvas for the visualization
    fig = plt.figure(figsize=(24, 16), dpi=100)
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Main Map Panel (left side, spanning 2 rows)
    # -------------------------------------------------------------------------
    ax_map = fig.add_subplot(gs[0:2, 0:2])
    ax_map.imshow(map_img)
    ax_map.set_title(f'Navigation Map - Step {step_data.step_idx}', fontsize=14, fontweight='bold')
    
    # Draw ground truth target (red circle)
    if step_data.target_true_px != [0, 0]:
        circle_true = plt.Circle(step_data.target_true_px, 25, color='red', fill=True, alpha=0.7, label='Ground Truth Target')
        ax_map.add_patch(circle_true)
        ax_map.annotate('TRUE TARGET', step_data.target_true_px, 
                       xytext=(step_data.target_true_px[0] + 50, step_data.target_true_px[1] - 50),
                       fontsize=10, color='red', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Draw predicted target (green circle)
    if step_data.target_pred_px != [0, 0]:
        circle_pred = plt.Circle(step_data.target_pred_px, 25, color='lime', fill=True, alpha=0.7, label='Predicted Target')
        ax_map.add_patch(circle_pred)
        ax_map.annotate('PREDICTED', step_data.target_pred_px, 
                       xytext=(step_data.target_pred_px[0] + 50, step_data.target_pred_px[1] + 50),
                       fontsize=10, color='lime', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='lime', lw=2))
    
    # Draw predicted landmark bbox (orange rectangle)
    if step_data.landmark_bbox_pred != [0, 0, 0, 0]:
        x1, y1, x2, y2 = step_data.landmark_bbox_pred
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor='orange', facecolor='none',
                                   linestyle='--', label='Predicted Landmark')
        ax_map.add_patch(rect)
    
    # Draw trajectory
    if len(step_data.trajectory_px) > 1:
        traj = np.array(step_data.trajectory_px)
        ax_map.plot(traj[:, 0], traj[:, 1], 'c-', linewidth=3, alpha=0.8, label='Trajectory')
        ax_map.scatter(traj[:-1, 0], traj[:-1, 1], c='cyan', s=30, zorder=5)
    
    # Draw start position (yellow star)
    ax_map.scatter(*step_data.start_px, marker='*', s=500, c='yellow', edgecolors='black', 
                   linewidths=2, zorder=10, label='Start Position')
    
    # Draw current position (cyan diamond)
    ax_map.scatter(*step_data.current_px, marker='D', s=300, c='cyan', edgecolors='black',
                   linewidths=2, zorder=10, label='Current Position')
    
    ax_map.legend(loc='upper right', fontsize=9)
    ax_map.axis('off')
    
    # -------------------------------------------------------------------------
    # Drone View Panel (top right)
    # -------------------------------------------------------------------------
    ax_drone = fig.add_subplot(gs[0, 2])
    if os.path.exists(step_data.drone_image_path):
        drone_img = cv2.imread(step_data.drone_image_path)
        drone_img = cv2.cvtColor(drone_img, cv2.COLOR_BGR2RGB)
        ax_drone.imshow(drone_img)
    ax_drone.set_title('Drone View (RGB)', fontsize=12, fontweight='bold')
    ax_drone.axis('off')
    
    # -------------------------------------------------------------------------
    # Metrics Panel (top right)
    # -------------------------------------------------------------------------
    ax_metrics = fig.add_subplot(gs[0, 3])
    ax_metrics.axis('off')
    
    metrics_text = f"""
    ╔══════════════════════════════════╗
    ║       STEP {step_data.step_idx} METRICS       ║
    ╠══════════════════════════════════╣
    ║ Distance to Target: {step_data.distance_to_target:.2f}m        ║
    ║ Inference Time: {step_data.inference_time:.2f}s          ║
    ║ Prediction: [{step_data.target_pred_px[0]}, {step_data.target_pred_px[1]}]          ║
    ║ Ground Truth: [{step_data.target_true_px[0]}, {step_data.target_true_px[1]}]        ║
    ╚══════════════════════════════════╝
    """
    
    # Success/failure indicator
    success = step_data.distance_to_target <= 20
    status = "✓ SUCCESS" if success else "✗ IN PROGRESS"
    status_color = 'green' if success else 'orange'
    
    ax_metrics.text(0.5, 0.9, status, fontsize=18, fontweight='bold', 
                   color=status_color, ha='center', va='top',
                   transform=ax_metrics.transAxes,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=status_color, linewidth=2))
    
    ax_metrics.text(0.5, 0.6, metrics_text, fontsize=10, family='monospace',
                   ha='center', va='top', transform=ax_metrics.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='gray'))
    
    # -------------------------------------------------------------------------
    # Thinking Panel (middle right)
    # -------------------------------------------------------------------------
    ax_think = fig.add_subplot(gs[1, 2:4])
    ax_think.axis('off')
    
    # Wrap the thinking text
    thinking_wrapped = textwrap.fill(step_data.thinking[:1500] if step_data.thinking else "No thinking available", 
                                      width=80)
    
    ax_think.text(0.02, 0.98, "🧠 Model Reasoning:", fontsize=12, fontweight='bold',
                 transform=ax_think.transAxes, va='top')
    ax_think.text(0.02, 0.88, thinking_wrapped, fontsize=9, family='monospace',
                 transform=ax_think.transAxes, va='top', wrap=True,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='#fffacd', edgecolor='#daa520'))
    
    # -------------------------------------------------------------------------
    # Answer Panel (bottom)
    # -------------------------------------------------------------------------
    ax_answer = fig.add_subplot(gs[2, :])
    ax_answer.axis('off')
    
    answer_text = f"📍 Final Answer: {step_data.answer}" if step_data.answer else "No answer parsed"
    ax_answer.text(0.02, 0.7, answer_text, fontsize=11, fontweight='bold',
                  transform=ax_answer.transAxes, va='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#e0ffe0', edgecolor='green'))
    
    # Landmark bbox info
    bbox_text = f"📦 Detected Landmark BBox: {step_data.landmark_bbox_pred}"
    ax_answer.text(0.02, 0.3, bbox_text, fontsize=11,
                  transform=ax_answer.transAxes, va='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff0e0', edgecolor='orange'))
    
    # Save
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved step visualization: {output_path}")


def create_summary_visualization(step_visualizations: List[StepVisualization], 
                                  episode_info: dict, metrics: dict, output_path: str):
    """Create a summary visualization showing all steps and final metrics"""
    
    n_steps = len(step_visualizations)
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = GridSpec(3, max(n_steps, 4), figure=fig, hspace=0.4, wspace=0.3)
    
    # -------------------------------------------------------------------------
    # Header with episode info
    # -------------------------------------------------------------------------
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis('off')
    
    header_text = f"""
    ════════════════════════════════════════════════════════════════════════════════
                                    FLIGHTGPT EVALUATION SUMMARY
    ════════════════════════════════════════════════════════════════════════════════
    
    📍 Episode ID: {episode_info.get('episode_id', 'N/A')}
    🗺️ Map: {episode_info.get('map_name', 'N/A')}
    🎯 Target: {episode_info.get('target_description', 'N/A')[:100]}...
    📏 Initial Distance: {episode_info.get('initial_distance', 0):.2f}m
    🔢 Total Steps: {n_steps}
    """
    
    ax_header.text(0.5, 0.5, header_text, fontsize=11, family='monospace',
                  ha='center', va='center', transform=ax_header.transAxes,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4fc', edgecolor='#2196F3', linewidth=2))
    
    # -------------------------------------------------------------------------
    # Step-by-step distance progression
    # -------------------------------------------------------------------------
    ax_progress = fig.add_subplot(gs[1, :2])
    
    steps = [s.step_idx for s in step_visualizations]
    distances = [s.distance_to_target for s in step_visualizations]
    
    # Plot distance over steps
    ax_progress.plot(steps, distances, 'b-o', linewidth=2, markersize=10, label='Distance to Target')
    ax_progress.axhline(y=20, color='g', linestyle='--', linewidth=2, label='Success Threshold (20m)')
    ax_progress.fill_between(steps, 0, 20, alpha=0.2, color='green')
    
    ax_progress.set_xlabel('Step', fontsize=12)
    ax_progress.set_ylabel('Distance to Target (m)', fontsize=12)
    ax_progress.set_title('Navigation Progress', fontsize=14, fontweight='bold')
    ax_progress.legend(loc='upper right')
    ax_progress.grid(True, alpha=0.3)
    ax_progress.set_ylim(bottom=0)
    
    # -------------------------------------------------------------------------
    # Metrics panel
    # -------------------------------------------------------------------------
    ax_metrics = fig.add_subplot(gs[1, 2:])
    ax_metrics.axis('off')
    
    final_distance = distances[-1] if distances else float('inf')
    success = final_distance <= 20
    
    metrics_text = f"""
    ╔════════════════════════════════════════╗
    ║         FINAL EVALUATION METRICS        ║
    ╠════════════════════════════════════════╣
    ║                                         ║
    ║  Navigation Error (NE): {metrics.get('NE', final_distance):>10.2f}m  ║
    ║  Success (SR):          {'✓ YES' if success else '✗ NO':>10}   ║
    ║  Final Distance:        {final_distance:>10.2f}m  ║
    ║  Total Inference Time:  {sum(s.inference_time for s in step_visualizations):>10.2f}s  ║
    ║                                         ║
    ╚════════════════════════════════════════╝
    """
    
    status_color = 'green' if success else 'red'
    ax_metrics.text(0.5, 0.5, metrics_text, fontsize=12, family='monospace',
                   ha='center', va='center', transform=ax_metrics.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor=status_color, linewidth=3))
    
    # -------------------------------------------------------------------------
    # Trajectory overview (simplified map thumbnails per step)
    # -------------------------------------------------------------------------
    for i, step_data in enumerate(step_visualizations):
        if i >= 4:  # Limit to 4 thumbnails
            break
        ax_thumb = fig.add_subplot(gs[2, i])
        
        if os.path.exists(step_data.map_image_path):
            map_img = cv2.imread(step_data.map_image_path)
            map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
            
            # Draw predictions on thumbnail
            if step_data.target_true_px != [0, 0]:
                cv2.circle(map_img, tuple(step_data.target_true_px), 30, (255, 0, 0), -1)
            if step_data.target_pred_px != [0, 0]:
                cv2.circle(map_img, tuple(step_data.target_pred_px), 30, (0, 255, 0), -1)
            
            ax_thumb.imshow(map_img)
        
        ax_thumb.set_title(f'Step {step_data.step_idx}\nDist: {step_data.distance_to_target:.1f}m', fontsize=10)
        ax_thumb.axis('off')
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved summary visualization: {output_path}")


def create_animated_gif(step_images: List[str], output_path: str, duration: int = 2000):
    """Create an animated GIF from step images"""
    if not step_images:
        return
    
    images = []
    for img_path in step_images:
        if os.path.exists(img_path):
            img = Image.open(img_path)
            images.append(img)
    
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"  ✓ Created animated GIF: {output_path}")


# ============================================================================
# Main Evaluation Visualization Function
# ============================================================================

def run_single_sample_visualization(
    sample_idx: int,
    citynavData,
    step_num: int,
    action_num: int,
    output_dir: str
):
    """
    Run evaluation on a single sample with detailed visualization
    """
    print(f"\n{'='*70}")
    print(f"  VISUALIZING SAMPLE {sample_idx}")
    print(f"{'='*70}")
    
    colors = create_color_palette()
    step_visualizations = []
    pose_history = []
    cur_trajectory = []
    
    # Get the sample data
    cur_citynavData = citynavData[sample_idx]
    episode = citynavData.episodes[sample_idx]
    
    # Episode info for summary
    episode_info = {
        'episode_id': str(episode.id),
        'map_name': episode.map_name,
        'target_description': episode.target_description,
        'initial_distance': episode.start_pose.xy.dist_to(episode.target_position.xy)
    }
    
    print(f"\n📍 Episode: {episode_info['episode_id']}")
    print(f"🗺️ Map: {episode_info['map_name']}")
    print(f"🎯 Target: {episode_info['target_description'][:80]}...")
    print(f"📏 Initial Distance: {episode_info['initial_distance']:.2f}m")
    
    # Create output directory for this sample
    sample_output_dir = os.path.join(output_dir, f"sample_{sample_idx}")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Save episode info
    with open(os.path.join(sample_output_dir, "episode_info.json"), 'w') as f:
        json.dump(episode_info, f, indent=2)
    
    for step_idx in range(step_num):
        print(f"\n{'─'*50}")
        print(f"  Step {step_idx + 1}/{step_num}")
        print(f"{'─'*50}")
        
        if pose_history:
            cur_citynavData.episode.teacher_trajectory[0] = pose_history[-1]
        
        # Create NavGym
        navGym = NavGym(cur_citynavData)
        start_pose = navGym.start_pose
        map_name = navGym.episode.id[0]
        
        print(f"  📂 Map Image: {navGym.cur_whole_map}")
        print(f"  🚁 Drone View: {navGym.cur_rgb_drone}")
        
        # Initialize agent
        agent = GPTAgent(
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
        
        # Get the prompt for logging
        cur_position_px = navGym._get_px(start_pose)
        prompt = get_prompt(navGym.target_description, cur_position_px)
        
        # Save prompt
        with open(os.path.join(sample_output_dir, f"step_{step_idx}_prompt.txt"), 'w') as f:
            f.write(prompt)
        
        # Call the model
        print(f"  🤖 Calling model...")
        inference_start = time.time()
        result_str = agent.act(
            cur_whole_map=navGym.cur_whole_map,
            cur_rgb_drone=navGym.cur_rgb_drone,
            cur_position=cur_position_px
        )
        inference_time = time.time() - inference_start
        print(f"  ⏱️ Inference time: {inference_time:.2f}s")
        
        # Save full response
        with open(os.path.join(sample_output_dir, f"step_{step_idx}_response.txt"), 'w') as f:
            f.write(result_str)
        
        # Parse response
        thinking = parse_thinking(result_str)
        answer = parse_answer(result_str)
        landmark_bbox_resized = parse_bbox(result_str, "landmark_bbox")
        target_pred_px_resized = parse_location(result_str)
        
        # Scale coordinates back to original image dimensions
        landmark_bbox = agent.scale_coordinates_to_original(landmark_bbox_resized)
        target_pred_px = agent.scale_coordinates_to_original(target_pred_px_resized)
        
        print(f"  🎯 Predicted Target (px): {target_pred_px}")
        print(f"  📦 Predicted Landmark BBox: {landmark_bbox}")
        
        # Get ground truth
        true_start_px = navGym.px_trajectory[0]
        true_target_px = navGym.target_px
        
        print(f"  ✓ True Target (px): {true_target_px}")
        
        # Compute predicted pose and distance
        pred_pose = compute_pose(navGym, target_pred_px, true_start_px, map_name)
        
        # Calculate distance to target
        if target_pred_px != [0, 0]:
            # Calculate in world coordinates
            target_world = episode.target_position
            current_distance = pred_pose.xy.dist_to(target_world.xy)
        else:
            current_distance = episode_info['initial_distance']
        
        print(f"  📏 Current Distance to Target: {current_distance:.2f}m")
        
        # Update trajectory
        if not pose_history:
            cur_trajectory = [start_pose]
            move_trajectory = move(start_pose, pred_pose, action_num)
            if move_trajectory:
                pose_history.append(move_trajectory[-1])
            cur_trajectory.extend(move_trajectory)
        else:
            move_trajectory = move(start_pose, pred_pose, action_num)
            if move_trajectory:
                pose_history.append(move_trajectory[-1])
            cur_trajectory.extend(move_trajectory)
        
        # Convert trajectory to pixel coordinates for visualization
        trajectory_px = [navGym._get_px(p) for p in cur_trajectory]
        current_px = trajectory_px[-1] if trajectory_px else true_start_px
        
        # Create step visualization data
        step_viz = StepVisualization(
            step_idx=step_idx,
            map_image_path=navGym.cur_whole_map,
            drone_image_path=navGym.cur_rgb_drone,
            prompt=prompt,
            model_response=result_str,
            thinking=thinking,
            answer=answer,
            landmark_bbox_pred=landmark_bbox,
            target_pred_px=target_pred_px,
            target_true_px=true_target_px,
            start_px=true_start_px,
            current_px=current_px,
            trajectory_px=trajectory_px,
            distance_to_target=current_distance,
            inference_time=inference_time
        )
        step_visualizations.append(step_viz)
        
        # Generate step visualization image
        step_output_path = os.path.join(sample_output_dir, f"step_{step_idx}_visualization.png")
        create_step_visualization(step_viz, step_output_path, colors)
    
    # Generate summary visualization
    summary_path = os.path.join(sample_output_dir, "summary.png")
    metrics = {
        'NE': step_visualizations[-1].distance_to_target if step_visualizations else 0
    }
    create_summary_visualization(step_visualizations, episode_info, metrics, summary_path)
    
    # Create animated GIF
    step_images = [os.path.join(sample_output_dir, f"step_{i}_visualization.png") 
                   for i in range(len(step_visualizations))]
    gif_path = os.path.join(sample_output_dir, "navigation_animation.gif")
    create_animated_gif(step_images, gif_path, duration=3000)
    
    print(f"\n{'='*70}")
    print(f"  ✓ VISUALIZATION COMPLETE")
    print(f"  📁 Output Directory: {sample_output_dir}")
    print(f"{'='*70}\n")
    
    return step_visualizations, episode_info


def create_data_flow_diagram(output_path: str):
    """Create a diagram explaining the data flow in evaluation"""
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    ax.axis('off')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    
    # Title
    ax.text(8, 9.5, "FlightGPT Evaluation Data Flow", fontsize=20, fontweight='bold', ha='center')
    
    # Boxes for each component
    boxes = [
        (1, 7, 3, 1.5, "CityNavData\n(Dataset Loader)", "#E3F2FD"),
        (1, 4.5, 3, 1.5, "NavGym\n(Simulation Environment)", "#E8F5E9"),
        (1, 2, 3, 1.5, "GPTAgent\n(Model Interface)", "#FFF3E0"),
        (6, 7, 3, 1.5, "Map Image\n+ Landmarks", "#F3E5F5"),
        (6, 4.5, 3, 1.5, "Drone View\n(RGB)", "#FCE4EC"),
        (6, 2, 3, 1.5, "Prompt\n(Target Description)", "#E0F7FA"),
        (11, 5.5, 4, 2, "VLM Model\n(Qwen2.5-VL)\n\nvLLM Server", "#FFF9C4"),
        (11, 2, 4, 2, "Output\n• Thinking\n• Landmark BBox\n• Target Location", "#C8E6C9"),
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='gray', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    arrows = [
        (4, 7.75, 6, 7.75),
        (4, 5.25, 6, 5.25),
        (4, 2.75, 6, 2.75),
        (9, 7.75, 11, 6.5),
        (9, 5.25, 11, 6),
        (9, 2.75, 11, 3.5),
        (13, 4, 13, 2),
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    
    # Legend
    ax.text(8, 0.5, "Data Flow: Dataset → Environment → Agent → Model → Prediction", 
           fontsize=12, ha='center', style='italic')
    
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Created data flow diagram: {output_path}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FlightGPT Evaluation Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize first sample from easy split
  python visualize_eval.py --sample_idx 0 --split easy
  
  # Visualize with more steps
  python visualize_eval.py --sample_idx 5 --split medium --step_num 3
  
  # Specify custom output directory
  python visualize_eval.py --sample_idx 0 --output_dir ./my_visualization
        """
    )
    
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample to visualize (default: 0)')
    parser.add_argument('--split', type=str, default='easy', choices=['easy', 'medium', 'hard'],
                        help='Data split to use (default: easy)')
    parser.add_argument('--step_num', type=int, default=2,
                        help='Number of navigation steps (default: 2)')
    parser.add_argument('--action_num', type=int, default=75,
                        help='Number of actions per step (default: 75)')
    parser.add_argument('--output_dir', type=str, default='./visualization_output',
                        help='Output directory for visualizations (default: ./visualization_output)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data JSON file (default: ./data/citynav/citynav_val_unseen_{split}.json)')
    parser.add_argument('--create_diagram', action='store_true',
                        help='Also create a data flow diagram')
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "🚁" * 35)
    print("   FLIGHTGPT EVALUATION VISUALIZATION")
    print("🚁" * 35)
    print(f"\n  Configuration:")
    print(f"    Sample Index: {args.sample_idx}")
    print(f"    Split: {args.split}")
    print(f"    Steps: {args.step_num}")
    print(f"    Actions per Step: {args.action_num}")
    print(f"    Output Directory: {args.output_dir}")
    print(f"    vLLM Server: {API_CONFIG['api_base']}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data flow diagram if requested
    if args.create_diagram:
        diagram_path = os.path.join(args.output_dir, "data_flow_diagram.png")
        create_data_flow_diagram(diagram_path)
    
    # Load data
    data_path = args.data_path or f"./data/citynav/citynav_val_unseen_{args.split}.json"
    print(f"📂 Loading data from: {data_path}")
    
    citynavData = CityNavData(data_path)
    print(f"✓ Loaded {len(citynavData)} samples")
    
    if args.sample_idx >= len(citynavData):
        print(f"❌ Error: sample_idx {args.sample_idx} is out of range (max: {len(citynavData)-1})")
        sys.exit(1)
    
    # Run visualization
    step_visualizations, episode_info = run_single_sample_visualization(
        sample_idx=args.sample_idx,
        citynavData=citynavData,
        step_num=args.step_num,
        action_num=args.action_num,
        output_dir=args.output_dir
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  📍 Episode: {episode_info['episode_id']}")
    print(f"  🗺️ Map: {episode_info['map_name']}")
    print(f"  📏 Initial Distance: {episode_info['initial_distance']:.2f}m")
    
    if step_visualizations:
        final_distance = step_visualizations[-1].distance_to_target
        success = final_distance <= 20
        print(f"  📏 Final Distance: {final_distance:.2f}m")
        print(f"  {'✓ SUCCESS' if success else '✗ NOT YET SUCCESSFUL'}")
        
        total_time = sum(s.inference_time for s in step_visualizations)
        print(f"  ⏱️ Total Inference Time: {total_time:.2f}s")
    
    print(f"\n  📁 All visualizations saved to: {args.output_dir}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
