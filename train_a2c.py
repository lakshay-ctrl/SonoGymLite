"""
train_a2c.py — Advantage Actor-Critic (A2C) Training for Autonomous Robotic Ultrasound.
========================================================================================

Dedicated high-throughput script for A2C training across:
  1. Spine Model-Based (MB Conv)
  2. Spine Learning-Based (LB U-Net)
  3. Liver Model-Based (MB Conv)
  4. Aorta Model-Based (MB Conv)

Key Features:
  - Vectorized Multi-Worker Rollouts (SubprocVecEnv, 8 parallel envs): Eliminates CPU
    bottleneck and matches PPO's >200 FPS throughput on NVIDIA RTX PRO 5000 Blackwell.
  - CUDA GPU Accelerated: Fast batched forward and backward passes directly on GPU.
  - Goal-Conditioned: Uses GoalConditionedExtractor fusing visual CNN features with 
    relative goal vector [dp / 0.05] and contact force.
  - Checkpoint and TensorBoard auto-naming per anatomy and simulation mode.

Usage:
  # 1. Spine MB (Model-Based Conv)
  python train_a2c.py --anatomy spine --sim-mode conv --timesteps 100000 --n-envs 8 --device cuda

  # 2. Spine LB (Learning-Based U-Net)
  python train_a2c.py --anatomy spine --sim-mode unet --timesteps 100000 --n-envs 4 --device cuda

  # 3. Liver MB (Model-Based Conv)
  python train_a2c.py --anatomy liver --sim-mode conv --timesteps 100000 --n-envs 8 --device cuda

  # 4. Aorta MB (Model-Based Conv)
  python train_a2c.py --anatomy aorta --sim-mode conv --timesteps 100000 --n-envs 8 --device cuda
"""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from robotic_us_env import RoboticUltrasoundGymEnv, TRAIN_PATIENTS
from us_feature_extractor import GoalConditionedExtractor


def make_env_fn(rank: int, args, mode_tag: str):
    def _init():
        if args.multi_patient:
            train_pids = ["s0011", "s0058", "s0250"]
            assigned_subj = str(Path(__file__).parent / "totalseg_patients" / train_pids[rank % len(train_pids)])
        else:
            assigned_subj = args.subject

        env = RoboticUltrasoundGymEnv(
            subject_dir=assigned_subj,
            checkpoint_path="runs/cavalcanti_unet/best_model.pth",
            device=args.device,
            render_mode="rgb_array",
            max_episode_steps=150,
            size=args.size,
            task=args.task,
            anatomy=args.anatomy,
            sim_mode=args.sim_mode,
            multi_patient=args.multi_patient,
            goal_conditioned=True,
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train an A2C Agent for Autonomous Robotic Ultrasound Scanning")
    parser.add_argument("--anatomy", type=str, default="spine", choices=["spine", "liver", "aorta"], help="Target anatomy")
    parser.add_argument("--sim-mode", type=str, default="conv", choices=["conv", "unet", "ray"], help="Ultrasound simulator mode")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps to train (default: 100k)")
    parser.add_argument("--tb-log", type=str, default=None, help="TensorBoard log directory (default: auto per organ)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save checkpoints (default: auto per organ)")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Initial subject patient directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save a checkpoint every N steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--n-steps", type=int, default=20, help="Steps collected per env before update (default: 20)")
    parser.add_argument("--lr", type=float, default=7e-4, help="Learning rate (default: 7e-4)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor (default: 0.99)")
    parser.add_argument("--task", type=str, default="navigation", choices=["navigation", "reconstruction"], help="Task mode")
    parser.add_argument("--multi-patient", action="store_true", help="Enable multi-patient domain randomization")
    parser.add_argument("--size", type=int, default=128, help="Ultrasound image resolution (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--vec-env", type=str, default="subproc", choices=["subproc", "dummy"], help="Vector environment type")
    args = parser.parse_args()

    # Force CUDA acceleration — non-negotiable for fast training
    assert torch.cuda.is_available(), (
        "[FATAL ERROR] CUDA GPU acceleration is MANDATORY for training!\n"
        "No CUDA-capable GPU was detected by PyTorch."
    )
    device = "cuda"

    # Auto-resolve directories
    mode_tag = "mb" if args.sim_mode in ["conv", "ray"] else "lb"
    if args.save_dir is None:
        args.save_dir = f"./a2c_checkpoints/{args.anatomy}_{mode_tag}/"
    if args.tb_log is None:
        args.tb_log = f"./a2c_tensorboard/{args.anatomy}_{mode_tag}/"

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 70)
    print("  Training A2C Agent for Autonomous Robotic Ultrasound Scanning")
    print("=" * 70)
    print(f"  Target Anatomy:    {args.anatomy.upper()}")
    print(f"  Simulation Mode:   {args.sim_mode} ({'Model-Based Physics' if mode_tag == 'mb' else 'Learning-Based U-Net'})")
    print(f"  Total Timesteps:   {args.timesteps}")
    print(f"  Parallel Workers:  {args.n_envs} ({args.vec_env.upper()} vectorized envs)")
    print(f"  Training Device:   {device.upper()} (GPU Acceleration: {'ENABLED' if 'cuda' in device else 'DISABLED'})")
    if torch.cuda.is_available() and "cuda" in device:
        print(f"  Active GPU:        {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print(f"  Goal Conditioning: ENABLED (GoalConditionedExtractor with relative displacement)")
    print(f"  Save Directory:    {args.save_dir}")
    print(f"  TensorBoard Log:   {args.tb_log}")
    print("=" * 70)

    # 1. Environment creation
    env_fns = [make_env_fn(i, args, mode_tag) for i in range(args.n_envs)]
    if args.vec_env == "subproc" and args.n_envs > 1:
        train_env = SubprocVecEnv(env_fns)
    else:
        train_env = DummyVecEnv(env_fns)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=args.save_dir,
        name_prefix=f"a2c_{args.anatomy}_{mode_tag}_model",
        verbose=1
    )

    policy_kwargs = dict(
        features_extractor_class=GoalConditionedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128, vec_output_dim=128),
    )

    # 2. A2C Agent setup
    model = A2C(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        gamma=args.gamma,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tb_log,
        verbose=1,
        device=device
    )

    # 3. Training Loop
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb], progress_bar=True)
        elapsed = time.time() - t0
        print(f"\n  [OK] A2C training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Save model under both standard and specific names
        named_path = os.path.join(args.save_dir, f"a2c_{mode_tag}_{args.anatomy}_final_model.zip")
        general_path = os.path.join(args.save_dir, "a2c_final_model.zip")
        model.save(named_path)
        model.save(general_path)
        print(f"  Final model saved to: {named_path} and {general_path}")
    except KeyboardInterrupt:
        print("\n  [Warning] Training interrupted by user. Saving current checkpoint...")
        interrupted_path = os.path.join(args.save_dir, f"a2c_{args.anatomy}_{mode_tag}_interrupted_model.zip")
        model.save(interrupted_path)
        print(f"  Interrupted model saved to: {interrupted_path}")
    finally:
        train_env.close()
        print("  Environments closed cleanly.")

if __name__ == "__main__":
    main()
