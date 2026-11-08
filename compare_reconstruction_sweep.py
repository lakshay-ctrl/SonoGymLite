"""
compare_reconstruction_sweep.py
================================
Surface Reconstruction Benchmark replicating SonoGym (NeurIPS 2025 Section 5.3 & Figure 7).
Compares Autonomous RL Surface Sweeping against Traditional Heuristic Raster/Zig-Zag Sweeping.

Metrics:
  1. Surface Coverage Points / Coverage Ratio (%)
  2. Trajectory Length (m)
  3. Cumulative Rotation (rad)
  4. Efficiency Ratio: Discovered Points per meter traveled
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np

# Add workspace to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import PPO, SAC, A2C
from robotic_us_env import RoboticUltrasoundGymEnv


def run_heuristic_raster_sweep(env, max_steps=200):
    """
    Traditional Heuristic Raster (Zig-Zag) Sweep across the region:
    Scans in a serpentine pattern along X and Y while maintaining normal contact.
    """
    obs, info = env.reset()
    visited_points = set()
    total_traj_len = 0.0
    total_cum_rot = 0.0
    prev_tip = env.home_pos[:3].copy()
    prev_orn = obs["pose"][3:7].copy()

    # Serpentine raster motion parameters
    sweep_speed_x = 0.6
    direction = 1.0
    step_in_row = 0
    row_len = 25

    for step in range(max_steps):
        is_3dof = (env.action_space.shape[0] == 3)
        if step_in_row >= row_len:
            # Shift Y row
            action = np.array([0.0, 0.4, 0.0], dtype=np.float32) if is_3dof else np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            direction *= -1.0
            step_in_row = 0
        else:
            # Move along X in current direction
            action = np.array([direction * sweep_speed_x, 0.0, 0.0], dtype=np.float32) if is_3dof else np.array([direction * sweep_speed_x, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            step_in_row += 1

        obs, reward, terminated, truncated, info = env.step(action)

        ee_pos = obs["pose"][:3]
        ee_orn = obs["pose"][3:7]

        step_dist = float(np.linalg.norm(ee_pos - prev_tip))
        total_traj_len += step_dist
        prev_tip = ee_pos.copy()

        # Orientation delta
        rot_delta = float(np.linalg.norm(ee_orn - prev_orn))
        total_cum_rot += rot_delta
        prev_orn = ee_orn.copy()

        if terminated or truncated:
            break

    coverage = info.get("coverage_points", len(env.visited_voxels))
    return {
        "method": "Heuristic Raster Sweep",
        "coverage_points": coverage,
        "trajectory_length_m": total_traj_len,
        "cumulative_rotation_rad": total_cum_rot,
        "efficiency_pts_per_m": coverage / (total_traj_len + 1e-6)
    }


def run_rl_sweep(env, policy, max_steps=200):
    """
    Autonomous RL Sweep driven by learned policy optimizing Task 2 submodular gain.
    """
    obs, info = env.reset()
    total_traj_len = 0.0
    total_cum_rot = 0.0
    prev_tip = env.home_pos[:3].copy()
    prev_orn = obs["pose"][3:7].copy()

    for step in range(max_steps):
        if policy is not None:
            action, _ = policy.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        ee_pos = obs["pose"][:3]
        ee_orn = obs["pose"][3:7]

        step_dist = float(np.linalg.norm(ee_pos - prev_tip))
        total_traj_len += step_dist
        prev_tip = ee_pos.copy()

        rot_delta = float(np.linalg.norm(ee_orn - prev_orn))
        total_cum_rot += rot_delta
        prev_orn = ee_orn.copy()

        if terminated or truncated:
            break

    coverage = info.get("coverage_points", len(env.visited_voxels))
    return {
        "method": "Autonomous RL Policy" if policy is not None else "Random Baseline",
        "coverage_points": coverage,
        "trajectory_length_m": total_traj_len,
        "cumulative_rotation_rad": total_cum_rot,
        "efficiency_pts_per_m": coverage / (total_traj_len + 1e-6)
    }


def main():
    parser = argparse.ArgumentParser(description="SonoGym Task 2: Reconstruction Sweep Benchmark")
    parser.add_argument("--model-path", type=str, default="", help="Path to trained RL checkpoint")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Patient directory")
    parser.add_argument("--max-steps", type=int, default=150, help="Steps per sweep")
    parser.add_argument("--output-json", type=str, default="reconstruction_sweep_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  SonoGym Task 2: Surface Reconstruction Sweep Comparison")
    print("=" * 70)
    print(f"  Patient:        {args.subject}")
    print(f"  Max Steps:      {args.max_steps}")

    # Load RL policy if provided
    policy = None
    if args.model_path and Path(args.model_path).exists():
        from eval_sonogym_benchmark import load_policy
        policy = load_policy(args.model_path)
        print(f"  Loaded RL Policy: {args.model_path}")

    # 1. Evaluate Heuristic Sweep
    print("\n  [1/2] Running Traditional Heuristic Raster (Zig-Zag) Sweep ...")
    env_raster = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        task="reconstruction",
        anatomy="spine",
        sim_mode="conv",
        max_episode_steps=args.max_steps,
        render_mode="rgb_array"
    )
    res_heuristic = run_heuristic_raster_sweep(env_raster, max_steps=args.max_steps)
    env_raster.close()

    # 2. Evaluate RL Sweep
    print("\n  [2/2] Running Autonomous Policy Sweep ...")
    env_rl = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        task="reconstruction",
        anatomy="spine",
        sim_mode="conv",
        max_episode_steps=args.max_steps,
        render_mode="rgb_array"
    )
    res_rl = run_rl_sweep(env_rl, policy, max_steps=args.max_steps)
    env_rl.close()

    # 3. Print Comparison Table
    print("\n" + "=" * 70)
    print("  RECONSTRUCTION BENCHMARK COMPARISON (SonoGym Fig. 7)")
    print("=" * 70)
    print(f"{'Method':<26} | {'Coverage (pts)':<14} | {'Traj Len (m)':<13} | {'Rot (rad)':<10} | {'Efficiency (pts/m)'}")
    print("-" * 80)
    print(f"{res_heuristic['method']:<26} | {res_heuristic['coverage_points']:<14d} | {res_heuristic['trajectory_length_m']:<13.3f} | {res_heuristic['cumulative_rotation_rad']:<10.3f} | {res_heuristic['efficiency_pts_per_m']:.2f}")
    print(f"{res_rl['method']:<26} | {res_rl['coverage_points']:<14d} | {res_rl['trajectory_length_m']:<13.3f} | {res_rl['cumulative_rotation_rad']:<10.3f} | {res_rl['efficiency_pts_per_m']:.2f}")
    print("=" * 70)

    # Save JSON report
    with open(args.output_json, "w") as f:
        json.dump({"heuristic": res_heuristic, "autonomous": res_rl}, f, indent=2)
    print(f"  Saved comparison results to {args.output_json}\n")


if __name__ == "__main__":
    main()
