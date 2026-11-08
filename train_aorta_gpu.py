"""
train_aorta_gpu.py
==================
Dedicated Turnkey GPU Training Script for Autonomous Aorta Scanning Agents.
Designed for execution on CUDA workstation / GPU server.

Key Fixes Integrated:
1. Anatomical Aorta Bifurcation:
   - Target relocated from thoracic T5/T6 to the true clinical abdominal bifurcation at L4/L5.
2. Supine Patient Registration:
   - The patient is placed face-up (supine, +90 deg X rotation) on the hospital bed mattress.
   - Lowers the probe onto the anterior abdominal wall.
   - Vertebral column sits behind the aorta (0% bone occlusion from anterior approach).
3. Precomputed Supine Surface Maps:
   - Uses surface_map_supine.npz for O(1) surface height and outward normal lookup.
   - Strict clinical acoustic coupling: 2 mm indentation, 2-8 N contact force.

Usage:
------
1. Train PPO on Aorta (Recommended baseline):
   python train_aorta_gpu.py --algo ppo --timesteps 100000 --multi-patient

2. Train SAC on Aorta (High sample-efficiency):
   python train_aorta_gpu.py --algo sac --timesteps 50000 --multi-patient

3. Train A2C on Aorta:
   python train_aorta_gpu.py --algo a2c --timesteps 100000 --multi-patient

4. Train with Neural Rendering (GPU U-Net simulation):
   python train_aorta_gpu.py --algo ppo --sim-mode unet --timesteps 50000

5. Run Evaluation only on an existing checkpoint:
   python train_aorta_gpu.py --eval-only --eval-model aorta_checkpoints_ppo/ppo_final_model.zip
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch

# Ensure current folder is on sys.path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robotic_us_env import RoboticUltrasoundGymEnv

TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
TEST_PATIENTS = ["s0344", "s0358"]
ALL_PATIENTS = TRAIN_PATIENTS + TEST_PATIENTS


def evaluate_policy(model, env, num_episodes: int = 5):
    """Evaluate a trained model across episodes and report metrics."""
    metrics = {
        "success": [],
        "final_pos_err_mm": [],
        "mean_reward": [],
        "mean_force_N": [],
        "steps": []
    }

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        total_rew = 0.0
        forces = []
        step_count = 0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done_arr, info_list = env.step(action)
            done = bool(done_arr[0])
            total_rew += float(rew[0])
            info = info_list[0]
            forces.append(float(info.get("force", 4.0)))
            step_count += 1
            final_info = info

        pos_err_mm = float(final_info.get("pos_error_mm", 100.0))
        success = bool(final_info.get("is_success", pos_err_mm <= 15.0))

        metrics["success"].append(success)
        metrics["final_pos_err_mm"].append(pos_err_mm)
        metrics["mean_reward"].append(total_rew)
        metrics["mean_force_N"].append(float(np.mean(forces)) if forces else 4.0)
        metrics["steps"].append(step_count)

    return {
        "success_rate": float(np.mean(metrics["success"])),
        "mean_pos_err_mm": float(np.mean(metrics["final_pos_err_mm"])),
        "std_pos_err_mm": float(np.std(metrics["final_pos_err_mm"])),
        "mean_reward": float(np.mean(metrics["mean_reward"])),
        "mean_force_N": float(np.mean(metrics["mean_force_N"])),
        "mean_steps": float(np.mean(metrics["steps"]))
    }


def run_full_cohort_eval(model, algo_name: str, sim_mode: str, output_json: Path):
    """Run standardized 5-patient evaluation protocol."""
    print("\n" + "=" * 65)
    print("  RUNNING STANDARDIZED AORTA BENCHMARK EVALUATION (5 EPISODES/PATIENT)")
    print("=" * 65)
    results = {}

    for pat in ALL_PATIENTS:
        pat_dir = ROOT_DIR / "totalseg_patients" / pat
        if not (pat_dir / "ct.nii.gz").exists():
            print(f"  [Skip] Patient {pat} not found on disk.")
            continue

        cohort_type = "In-Domain Train" if pat in TRAIN_PATIENTS else "Held-Out Test"
        print(f"\n  Evaluating Patient {pat} ({cohort_type}) ...")

        def make_eval_env():
            env = RoboticUltrasoundGymEnv(
                subject_dir=str(pat_dir),
                device="auto",
                render_mode="rgb_array",
                max_episode_steps=200,
                size=128,
                task="navigation",
                anatomy="aorta",
                sim_mode=sim_mode,
                multi_patient=False,
                skip_unet=(sim_mode in ["conv", "ray"])
            )
            return Monitor(env)

        eval_env = DummyVecEnv([make_eval_env])
        try:
            res = evaluate_policy(model, eval_env, num_episodes=5)
            results[pat] = {
                "cohort": cohort_type,
                **res
            }
            print(f"    Success Rate:      {res['success_rate']*100:.1f} %")
            print(f"    Mean Pos Error:    {res['mean_pos_err_mm']:.1f} ± {res['std_pos_err_mm']:.1f} mm")
            print(f"    Mean Contact Force:{res['mean_force_N']:.2f} N")
            print(f"    Mean Episode Rew:  {res['mean_reward']:+.2f}")
        finally:
            eval_env.close()

    # Summary table
    print("\n" + "-" * 65)
    print(f"  {'Patient':<8} | {'Cohort':<16} | {'Success':<9} | {'Pos Err (mm)':<14} | {'Force (N)':<10}")
    print("-" * 65)
    for pat, res in results.items():
        print(f"  {pat:<8} | {res['cohort']:<16} | {res['success_rate']*100:5.1f} %   | {res['mean_pos_err_mm']:5.1f} ± {res['std_pos_err_mm']:4.1f} mm | {res['mean_force_N']:5.2f} N")
    print("-" * 65)

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Detailed evaluation metrics saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Robotic Aorta Ultrasound Training & Evaluation")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac", "a2c"], help="RL Algorithm")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps (default: 100k)")
    parser.add_argument("--sim-mode", type=str, default="conv", choices=["conv", "ray", "unet", "pix2pix"], help="US simulator mode")
    parser.add_argument("--multi-patient", action="store_true", default=True, help="Enable multi-patient domain randomization (s0011, s0058, s0250)")
    parser.add_argument("--no-multi-patient", dest="multi_patient", action="store_false")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Default subject if single patient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size (default: 64)")
    parser.add_argument("--save-freq", type=int, default=10000, help="Checkpoint frequency in steps")
    parser.add_argument("--eval-after", action="store_true", default=True, help="Run standardized 5-patient evaluation after training")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run evaluation only")
    parser.add_argument("--eval-model", type=str, default="", help="Path to checkpoint model for eval-only mode")
    parser.add_argument("--save-dir", type=str, default="", help="Custom checkpoint save directory")
    parser.add_argument("--tb-log", type=str, default="", help="Custom tensorboard log directory")
    args = parser.parse_args()

    # Determine directories
    save_dir = Path(args.save_dir if args.save_dir else f"./aorta_checkpoints_{args.algo}")
    tb_log = Path(args.tb_log if args.tb_log else f"./aorta_tensorboard_{args.algo}")
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_log.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  AUTONOMOUS ROBOTIC AORTA SCANNING — GPU RL SYSTEM")
    print("=" * 65)
    print(f"  PyTorch Version:      {torch.__version__}")
    print(f"  CUDA Available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU Device:           {torch.cuda.get_device_name(0)}")
    print(f"  Algorithm:            {args.algo.upper()}")
    print(f"  Target Anatomy:       AORTA BIFURCATION (L4/L5, Supine Anterior)")
    print(f"  Simulator Mode:       {args.sim_mode.upper()}")
    print(f"  Multi-Patient Mode:   {args.multi_patient} ({TRAIN_PATIENTS})")
    print(f"  Total Timesteps:      {args.timesteps}")
    print(f"  Checkpoints Directory:{save_dir}")
    print(f"  Tensorboard Directory:{tb_log}")
    print("=" * 65)

    # Eval-only mode
    if args.eval_only:
        model_path = Path(args.eval_model)
        if not model_path.exists():
            raise FileNotFoundError(f"Evaluation model checkpoint not found: {model_path}")
        print(f"\n  Loading checkpoint: {model_path}")
        if args.algo == "ppo":
            model = PPO.load(model_path)
        elif args.algo == "sac":
            model = SAC.load(model_path)
        elif args.algo == "a2c":
            model = A2C.load(model_path)
        else:
            raise ValueError(f"Unknown algo: {args.algo}")

        output_json = save_dir / "aorta_eval_benchmark.json"
        run_full_cohort_eval(model, args.algo, args.sim_mode, output_json)
        return

    # 1. Environment factory
    def make_train_env():
        env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            device="auto",
            render_mode="rgb_array",
            max_episode_steps=300,
            size=128,
            task="navigation",
            anatomy="aorta",
            sim_mode=args.sim_mode,
            multi_patient=args.multi_patient,
            skip_unet=(args.sim_mode in ["conv", "ray"])
        )
        return Monitor(env)

    train_env = DummyVecEnv([make_train_env])

    # 2. Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(save_dir),
        name_prefix=f"{args.algo}_aorta",
        verbose=1
    )

    # 3. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "auto"
    if args.algo == "ppo":
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=256,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(tb_log),
            verbose=1,
            device=device
        )
    elif args.algo == "sac":
        model = SAC(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.lr,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            tensorboard_log=str(tb_log),
            verbose=1,
            device=device
        )
    elif args.algo == "a2c":
        model = A2C(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=20,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=str(tb_log),
            verbose=1,
            device=device
        )

    # 4. Training loop
    print("\n  Starting training ...")
    final_model_path = save_dir / f"{args.algo}_final_model.zip"
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )
        print("\n  [OK] Training completed successfully!")
        model.save(str(final_model_path))
        print(f"  Final model saved to: {final_model_path}")
    except KeyboardInterrupt:
        print("\n  [Warning] Training interrupted by user. Saving checkpoint...")
        interrupted_path = save_dir / f"{args.algo}_interrupted_model.zip"
        model.save(str(interrupted_path))
        print(f"  Interrupted model saved to: {interrupted_path}")
        final_model_path = interrupted_path
    finally:
        train_env.close()

    # 5. Post-training evaluation
    if args.eval_after and final_model_path.exists():
        eval_output_json = save_dir / "aorta_eval_benchmark.json"
        run_full_cohort_eval(model, args.algo, args.sim_mode, eval_output_json)


if __name__ == "__main__":
    main()
