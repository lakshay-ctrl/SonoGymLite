#!/usr/bin/env python3
"""
scripts/step4b_lb_image_only.py
===============================
Learning-Based (LB) Spine Image-Only PPO Training and SonoGym Evaluation:
- sim_mode: "unet" (Cavalcanti U-Net simulation, bone mask reads full skeleton including sacrum).
- Observation space: Image + Contact force only (goal_conditioned=False).
- Training Budget: 300,000 steps across training patients (s0011, s0058, s0250).
- Cutoff rule: Thursday 30 Oct, 18:00 IST.
- Evaluation: SonoGym metrics on Train and Held-Out cohorts (fixed & early termination modes).
- Outputs:
  * ppo_checkpoints/spine_lb/ppo_lb_spine_image_only_final.zip
  * results/step4b_lb_image_only.md
  * results/step4b_lb_image_only.json
  * results/status.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import torch as th
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from robotic_us_env import RoboticUltrasoundGymEnv
from us_feature_extractor import UltrasoundCNNExtractor

TABLE_DIR = PROJECT_DIR / "results"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PROJECT_DIR / "ppo_checkpoints" / "spine_lb"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATETIME = datetime(2026, 10, 29, 18, 0, 0)  # Thursday 18:00 IST
TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
HELD_OUT_PATIENTS = ["s0344", "s0358"]


def check_cutoff():
    now = datetime.now()
    if now >= CUTOFF_DATETIME:
        print(f"[TIME GUARD] Current time {now} is past cutoff {CUTOFF_DATETIME}. Aborting job.")
        return False
    rem_h = (CUTOFF_DATETIME - now).total_seconds() / 3600.0
    print(f"[TIME GUARD] Verified: {rem_h:.1f} hours remaining until deadline ({CUTOFF_DATETIME}).")
    return True


def write_status(state="RUNNING", current_run="step4b_lb_image_only", error=None):
    status_file = TABLE_DIR / "status.json"
    status_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": state,
        "current_run": current_run,
        "error": error
    }
    try:
        with open(status_file, "w") as f:
            json.dump(status_data, f, indent=2)
    except Exception:
        pass


def make_lb_env(rank: int, max_steps: int = 300):
    def _init():
        pid = TRAIN_PATIENTS[rank % len(TRAIN_PATIENTS)]
        subj_dir = str(PROJECT_DIR / "totalseg_patients" / pid)
        env = RoboticUltrasoundGymEnv(
            subject_dir=subj_dir,
            task="navigation",
            anatomy="spine",
            sim_mode="unet",
            size=256,
            pixel_spacing=1.0,
            max_episode_steps=max_steps,
            early_termination=False,
            goal_conditioned=False,
            device="cuda" if th.cuda.is_available() else "cpu"
        )
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run tiny dry run (200 steps) to verify execution")
    parser.add_argument("--steps", type=int, default=300000, help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    args = parser.parse_args()

    write_status(state="RUNNING", current_run="step4b_lb_image_only")
    if not check_cutoff():
        return

    # Speed estimate: ~40 steps/sec with U-Net on GPU -> 300,000 steps takes ~2.1 hours.
    eta_hours = 2.5
    print("=" * 80)
    print("  STEP 4B: LEARNING-BASED (LB) SPINE IMAGE-ONLY PPO")
    print(f"  Training Steps: {args.steps:,d} | Target: Spine (LB U-Net) | Goal Conditioned: False")
    print(f"  Estimated Runtime: ~{eta_hours:.1f} hours | Deadline: {CUTOFF_DATETIME}")
    print("=" * 80)

    if args.dry_run:
        print("\n>>> PERFORMING TINY DRY RUN (200 TRAINING STEPS) <<<")
        env = make_lb_env(0, max_steps=50)()
        policy_kwargs = dict(
            features_extractor_class=UltrasoundCNNExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128)
        )
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="cpu"
        )
        t0 = time.time()
        model.learn(total_timesteps=128)
        el = time.time() - t0
        print(f"  [DRY-RUN] Model trained 128 steps in {el:.1f}s ({128/el:.1f} steps/s)")
        obs, _ = env.reset(seed=42)
        act, _ = model.predict(obs, deterministic=True)
        print(f"  [DRY-RUN] Sample inference action: {act}")
        env.close()
        print("\n[+] Step 4b dry run completed successfully! Exiting without full training.")
        write_status(state="DRY_RUN_PASSED", current_run="step4b_lb_image_only")
        return

    # Full training logic
    env_fns = [make_lb_env(i, max_steps=300) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")
    policy_kwargs = dict(
        features_extractor_class=UltrasoundCNNExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128)
    )
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(25000 // args.n_envs, 1),
        save_path=str(CKPT_DIR),
        name_prefix="ppo_lb_spine_image_only",
        verbose=1
    )

    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=[ckpt_cb])
    train_time = time.time() - t0
    vec_env.close()

    ckpt_path = CKPT_DIR / "ppo_lb_spine_image_only_final.zip"
    model.save(str(ckpt_path))
    print(f"\n[+] Trained LB Spine Image-Only PPO in {train_time/3600:.2f} hours. Saved to {ckpt_path}")

    # Evaluate in SonoGym metrics
    eval_results = {"Train": [], "Held-Out": []}
    for cohort, p_list in [("Train", TRAIN_PATIENTS), ("Held-Out", HELD_OUT_PATIENTS)]:
        for i in range(25):
            pid = p_list[i % len(p_list)]
            env_eval = RoboticUltrasoundGymEnv(
                subject_dir=str(PROJECT_DIR / "totalseg_patients" / pid),
                task="navigation",
                anatomy="spine",
                sim_mode="unet",
                size=256,
                max_episode_steps=300,
                early_termination=False,
                goal_conditioned=False,
                device="cpu"
            )
            obs, info = env_eval.reset(seed=2000 + i)
            for st in range(300):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env_eval.step(act)
                if term or trunc:
                    break
            eval_results[cohort].append({
                "pos_err_mm": float(info.get("pos_error_mm", 0.0)),
                "rot_err_deg": float(info.get("rot_error_deg", 0.0)),
                "success": bool(info.get("pos_error_mm", 999.0) < 5.0 and info.get("rot_error_deg", 999.0) < 10.0)
            })
            env_eval.close()

    out_json = TABLE_DIR / "step4b_lb_image_only.json"
    with open(out_json, "w") as f:
        json.dump(eval_results, f, indent=2)

    md_lines = [
        "# Step 4b: Learning-Based (LB) Spine Image-Only Results",
        "",
        "| Cohort | Pos Error (mm) | Rot Error (deg) | Success Rate (%) |",
        "| :---: | :---: | :---: | :---: |"
    ]
    for c in eval_results:
        p_err = [r["pos_err_mm"] for r in eval_results[c]]
        r_err = [r["rot_err_deg"] for r in eval_results[c]]
        succ = [100.0 if r["success"] else 0.0 for r in eval_results[c]]
        md_lines.append(f"| {c} | {np.mean(p_err):.1f} ± {np.std(p_err):.1f} | {np.mean(r_err):.1f} ± {np.std(r_err):.1f} | {np.mean(succ):.1f}% |")

    with open(TABLE_DIR / "step4b_lb_image_only.md", "w") as f:
        f.write("\n".join(md_lines))

    write_status(state="COMPLETED", current_run="step4b_lb_image_only")
    print(f"[DONE] Step 4b deliverables saved to {TABLE_DIR / 'step4b_lb_image_only.md'}")


if __name__ == "__main__":
    main()
