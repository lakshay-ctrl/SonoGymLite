#!/usr/bin/env python3
"""
scripts/fix_lb_image_only_eval.py
==================================
Runs clean evaluation for Spine LB Image-Only PPO:
1. Fixed-horizon: strictly runs all 300 steps (breaks only on trunc).
2. Early-term: breaks on term (success or out-of-bounds) and reports error AT terminating step.
"""

import os
import sys
import json
import time
import multiprocessing as mp
import numpy as np
import torch as th

os.environ["OMP_NUM_THREADS"] = "2"
PROJECT_DIR = Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from stable_baselines3 import PPO
from robotic_us_env import RoboticUltrasoundGymEnv

TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
TRAIN_COUNTS = [34, 33, 33]
HELD_OUT_PATIENTS = ["s0344", "s0358"]
HELD_OUT_COUNTS = [50, 50]
CKPT_PATH = os.path.join(PROJECT_DIR, "ppo_checkpoints/spine_lb/ppo_lb_spine_image_only_final.zip")

def run_trial(args):
    th.set_num_threads(1)
    patient, seed, early_term, max_steps = args
    subj_dir = os.path.join(PROJECT_DIR, "totalseg_patients", patient)

    model = PPO.load(CKPT_PATH, device="cpu")
    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="navigation",
        anatomy="spine",
        sim_mode="unet",
        size=256,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=early_term,
        goal_conditioned=False,
        device="cpu"
    )

    obs, info = env.reset(seed=seed)
    first_success = False
    step_count = 0

    for step in range(1, max_steps + 1):
        step_count = step
        c_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
        quat = np.array(env.unwrapped.current_probe_quat, dtype=np.float32)
        obs["contact"] = np.array([c_val], dtype=np.float32)
        obs["orientation"] = quat

        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)

        pos_err_mm = float(info.get("pos_error_mm", 0.0))
        rot_err_deg = float(info.get("rot_error_deg", 0.0))

        if pos_err_mm < 5.0 and rot_err_deg < 10.0:
            first_success = True

        if early_term:
            if term or trunc:
                break
        else:
            if trunc:
                break

    env.close()

    pos_err = float(info.get("pos_error_mm", 0.0))
    rot_err = float(info.get("rot_error_deg", 0.0))
    succ = bool(pos_err < 5.0 and rot_err < 10.0)

    return {
        "patient": patient,
        "seed": seed,
        "pos_err": pos_err,
        "rot_err": rot_err,
        "success": succ,
        "first_success": first_success,
        "steps": step_count
    }

def eval_cohort(patient_list, patient_counts, base_seed, early_term, workers=8):
    tasks = []
    idx = 0
    for p_idx, pid in enumerate(patient_list):
        for _ in range(patient_counts[p_idx]):
            tasks.append((pid, base_seed + idx, early_term, 300))
            idx += 1
    with mp.Pool(processes=workers) as pool:
        res = pool.map(run_trial, tasks)
    
    pos = [r["pos_err"] for r in res]
    rot = [r["rot_err"] for r in res]
    succ = [100.0 if r["success"] else 0.0 for r in res]
    first_succ = [100.0 if r["first_success"] else 0.0 for r in res]
    steps = [r["steps"] for r in res]

    return {
        "pos_mean": float(np.mean(pos)),
        "pos_std": float(np.std(pos)),
        "rot_mean": float(np.mean(rot)),
        "rot_std": float(np.std(rot)),
        "success_rate": float(np.mean(succ)),
        "first_success_rate": float(np.mean(first_succ)),
        "steps_mean": float(np.mean(steps)),
        "steps_std": float(np.std(steps))
    }

def main():
    print("Evaluating Fixed Horizon...")
    fixed_train = eval_cohort(TRAIN_PATIENTS, TRAIN_COUNTS, 1000, early_term=False)
    fixed_held = eval_cohort(HELD_OUT_PATIENTS, HELD_OUT_COUNTS, 2000, early_term=False)

    print("Evaluating Early Termination...")
    et_train = eval_cohort(TRAIN_PATIENTS, TRAIN_COUNTS, 1000, early_term=True)
    et_held = eval_cohort(HELD_OUT_PATIENTS, HELD_OUT_COUNTS, 2000, early_term=True)

    results = {
        "fixed_train": fixed_train,
        "fixed_held": fixed_held,
        "et_train": et_train,
        "et_held": et_held
    }

    with open("results/lb_image_only_corrected.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- RESULTS ---")
    print(f"Fixed Train:     Pos {fixed_train['pos_mean']:.2f} ± {fixed_train['pos_std']:.2f} mm | Rot {fixed_train['rot_mean']:.2f} ± {fixed_train['rot_std']:.2f}° | Succ {fixed_train['success_rate']:.1f}% | Steps {fixed_train['steps_mean']:.1f}")
    print(f"Fixed Held-Out:  Pos {fixed_held['pos_mean']:.2f} ± {fixed_held['pos_std']:.2f} mm | Rot {fixed_held['rot_mean']:.2f} ± {fixed_held['rot_std']:.2f}° | Succ {fixed_held['success_rate']:.1f}% | Steps {fixed_held['steps_mean']:.1f}")
    print(f"Early-Term Train:    Pos {et_train['pos_mean']:.2f} ± {et_train['pos_std']:.2f} mm | Rot {et_train['rot_mean']:.2f} ± {et_train['rot_std']:.2f}° | Succ {et_train['success_rate']:.1f}% | Steps {et_train['steps_mean']:.1f}")
    print(f"Early-Term Held-Out: Pos {et_held['pos_mean']:.2f} ± {et_held['pos_std']:.2f} mm | Rot {et_held['rot_mean']:.2f} ± {et_held['rot_std']:.2f}° | Succ {et_held['success_rate']:.1f}% | Steps {et_held['steps_mean']:.1f}")

if __name__ == "__main__":
    main()
