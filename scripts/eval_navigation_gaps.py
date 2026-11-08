#!/usr/bin/env python3
"""
scripts/eval_navigation_gaps.py
================================
Evaluates the requested gap models on 100 trials/cohort:
1. Goal-Conditioned Spine MB A2C (100k steps, a2c_spine_mb_model_100000_steps.zip)
2. Goal-Conditioned Spine LB PPO (100k steps, ppo_lb_spine_final_model.zip)
3. Goal-Conditioned Spine LB SAC (100k steps, sac_lb_spine_final_model.zip)
4. Image-Only Spine LB PPO Run 4b (300k steps, ppo_lb_spine_image_only_final.zip)
   - Evaluated in both Fixed-Horizon (300 steps) and Early-Termination ON.

Protocol:
- Train cohort: s0011 (34), s0058 (33), s0250 (33), seeds 1000-1099
- Held-out cohort: s0344 (50), s0358 (50), seeds 2000-2099
- Max steps: 300
- Success criterion: pos_error < 5.0 mm and rot_error < 10.0 deg
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch as th

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from stable_baselines3 import PPO, SAC, A2C
from robotic_us_env import RoboticUltrasoundGymEnv

TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
TRAIN_COUNTS = [34, 33, 33]
HELD_OUT_PATIENTS = ["s0344", "s0358"]
HELD_OUT_COUNTS = [50, 50]

MODELS_CONFIG = [
    {
        "name": "Spine MB Goal-A2C (100k)",
        "algo": "a2c",
        "ckpt": "a2c_checkpoints/spine_mb/a2c_spine_mb_model_100000_steps.zip",
        "sim_mode": "conv",
        "is_gc": True,
        "early_term": False,
        "eval_early_term": False
    },
    {
        "name": "Spine LB Goal-PPO (100k)",
        "algo": "ppo",
        "ckpt": "ppo_checkpoints/spine_lb/ppo_lb_spine_final_model.zip",
        "sim_mode": "unet",
        "is_gc": True,
        "early_term": False,
        "eval_early_term": False
    },
    {
        "name": "Spine LB Goal-SAC (100k)",
        "algo": "sac",
        "ckpt": "sac_checkpoints/spine_lb/sac_lb_spine_final_model.zip",
        "sim_mode": "unet",
        "is_gc": True,
        "early_term": False,
        "eval_early_term": False
    },
    {
        "name": "Spine LB Image-Only PPO (Run 4b, 300k)",
        "algo": "ppo",
        "ckpt": "ppo_checkpoints/spine_lb/ppo_lb_spine_image_only_final.zip",
        "sim_mode": "unet",
        "is_gc": False,
        "early_term": False,
        "eval_early_term": True
    }
]


def run_single_trial(args):
    th.set_num_threads(1)
    algo, ckpt_path, sim_mode, is_gc, early_term, patient, seed, max_steps = args
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient)

    if algo == "ppo":
        model = PPO.load(ckpt_path, device="cpu")
    elif algo == "sac":
        model = SAC.load(ckpt_path, device="cpu")
    elif algo == "a2c":
        model = A2C.load(ckpt_path, device="cpu")
    else:
        raise ValueError(f"Unknown algo {algo}")

    img_size = 128
    if hasattr(model.observation_space, "spaces") and "image" in model.observation_space.spaces:
        img_size = int(model.observation_space.spaces["image"].shape[0])

    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="navigation",
        anatomy="spine",
        sim_mode=sim_mode,
        size=img_size,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=early_term,
        goal_conditioned=is_gc,
        device="cpu"
    )

    obs, info = env.reset(seed=seed)
    first_success = False
    first_success_step = None
    term_pos_err = None
    term_rot_err = None
    term_step = max_steps

    final_pos_err = float(info.get("pos_error_mm", 0.0))
    final_rot_err = float(info.get("rot_error_deg", 0.0))

    for step in range(1, max_steps + 1):
        # Format obs keys if image-only
        if not is_gc:
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
            if not first_success:
                first_success = True
                first_success_step = step

        final_pos_err = pos_err_mm
        final_rot_err = rot_err_deg

        if early_term and (term or trunc):
            term_pos_err = pos_err_mm
            term_rot_err = rot_err_deg
            term_step = step
            break

        if term or trunc:
            break

    env.close()

    if term_pos_err is None:
        term_pos_err = final_pos_err
        term_rot_err = final_rot_err
        term_step = step

    return {
        "patient": patient,
        "seed": seed,
        "final_pos_mm": final_pos_err,
        "final_rot_deg": final_rot_err,
        "final_success": bool(final_pos_err < 5.0 and final_rot_err < 10.0),
        "first_success": bool(first_success),
        "first_success_step": first_success_step,
        "term_pos_mm": term_pos_err,
        "term_rot_deg": term_rot_err,
        "term_step": term_step,
        "term_success": bool(term_pos_err < 5.0 and term_rot_err < 10.0)
    }


def evaluate_cohort(cfg, patient_list, patient_counts, base_seed, early_term=False, workers=8):
    tasks = []
    trial_idx = 0
    for p_idx, pid in enumerate(patient_list):
        for _ in range(patient_counts[p_idx]):
            seed = base_seed + trial_idx
            tasks.append((
                cfg["algo"],
                str(PROJECT_DIR / cfg["ckpt"]),
                cfg["sim_mode"],
                cfg["is_gc"],
                early_term,
                pid,
                seed,
                300
            ))
            trial_idx += 1

    with mp.Pool(processes=workers) as pool:
        results = pool.map(run_single_trial, tasks)

    final_pos = [r["final_pos_mm"] for r in results]
    final_rot = [r["final_rot_deg"] for r in results]
    final_succ = [100.0 if r["final_success"] else 0.0 for r in results]
    first_succ = [100.0 if r["first_success"] else 0.0 for r in results]

    term_pos = [r["term_pos_mm"] for r in results]
    term_rot = [r["term_rot_deg"] for r in results]
    term_succ = [100.0 if r["term_success"] else 0.0 for r in results]
    term_steps = [r["term_step"] for r in results]

    return {
        "n_trials": len(results),
        "pos_mean": float(np.mean(final_pos)),
        "pos_std": float(np.std(final_pos)),
        "rot_mean": float(np.mean(final_rot)),
        "rot_std": float(np.std(final_rot)),
        "final_success_rate": float(np.mean(final_succ)),
        "first_success_rate": float(np.mean(first_succ)),
        "early_term": {
            "pos_mean": float(np.mean(term_pos)),
            "pos_std": float(np.std(term_pos)),
            "rot_mean": float(np.mean(term_rot)),
            "rot_std": float(np.std(term_rot)),
            "success_rate": float(np.mean(term_succ)),
            "steps_mean": float(np.mean(term_steps)),
            "steps_std": float(np.std(term_steps))
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_json = PROJECT_DIR / "results" / "navigation_gaps_report.json"
    out_md = PROJECT_DIR / "results" / "navigation_gaps_report.md"

    master_results = {}

    print("=" * 80)
    print("  EVALUATING NAVIGATION GAPS BENCHMARK SUITE")
    print("=" * 80)

    for cfg in MODELS_CONFIG:
        name = cfg["name"]
        print(f"\n[+] Evaluating Model: {name}")
        print(f"    Checkpoint: {cfg['ckpt']}")
        print(f"    Backend:    {cfg['sim_mode']}")

        # 1. Fixed Horizon (300 steps)
        print("  --> Running Train Cohort (Seeds 1000-1099, Fixed Horizon)...")
        t0 = time.time()
        train_res = evaluate_cohort(cfg, TRAIN_PATIENTS, TRAIN_COUNTS, base_seed=1000, early_term=False, workers=args.workers)
        print(f"      Train done in {time.time()-t0:.1f}s | Pos: {train_res['pos_mean']:.2f} mm | Succ: {train_res['final_success_rate']:.1f}% | First-Succ: {train_res['first_success_rate']:.1f}%")

        print("  --> Running Held-Out Cohort (Seeds 2000-2099, Fixed Horizon)...")
        t0 = time.time()
        held_out_res = evaluate_cohort(cfg, HELD_OUT_PATIENTS, HELD_OUT_COUNTS, base_seed=2000, early_term=False, workers=args.workers)
        print(f"      Held-Out done in {time.time()-t0:.1f}s | Pos: {held_out_res['pos_mean']:.2f} mm | Succ: {held_out_res['final_success_rate']:.1f}% | First-Succ: {held_out_res['first_success_rate']:.1f}%")

        model_entry = {
            "name": name,
            "ckpt": cfg["ckpt"],
            "sim_mode": cfg["sim_mode"],
            "is_gc": cfg["is_gc"],
            "train": train_res,
            "held_out": held_out_res
        }

        # 2. Early Termination Mode (if requested)
        if cfg["eval_early_term"]:
            print("  --> Running Train Cohort (Early Termination ON)...")
            train_et = evaluate_cohort(cfg, TRAIN_PATIENTS, TRAIN_COUNTS, base_seed=1000, early_term=True, workers=args.workers)
            print("  --> Running Held-Out Cohort (Early Termination ON)...")
            held_out_et = evaluate_cohort(cfg, HELD_OUT_PATIENTS, HELD_OUT_COUNTS, base_seed=2000, early_term=True, workers=args.workers)
            model_entry["early_term_train"] = train_et["early_term"]
            model_entry["early_term_held_out"] = held_out_et["early_term"]

        master_results[name] = model_entry

    with open(out_json, "w") as f:
        json.dump(master_results, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Navigation Gaps Benchmark Report",
        "",
        "Evaluated on 100 trials per cohort (Train: seeds 1000-1099, Held-Out: seeds 2000-2099, Horizon: 300 steps).",
        "Success criterion: `pos_error < 5.0 mm` and `rot_error < 10.0 deg`.",
        "",
        "| Model / Baseline | Mode | Cohort | Pos Error (mm) | Rot Error (deg) | Final Succ (%) | First-Succ (%) | Steps |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for name, res in master_results.items():
        tr = res["train"]
        ho = res["held_out"]
        md_lines.append(f"| **{name}** | Fixed | Train | {tr['pos_mean']:.2f} ± {tr['pos_std']:.2f} | {tr['rot_mean']:.2f} ± {tr['rot_std']:.2f}° | {tr['final_success_rate']:.1f}% | {tr['first_success_rate']:.1f}% | 300.0 |")
        md_lines.append(f"| **{name}** | Fixed | Held-Out | {ho['pos_mean']:.2f} ± {ho['pos_std']:.2f} | {ho['rot_mean']:.2f} ± {ho['rot_std']:.2f}° | {ho['final_success_rate']:.1f}% | {ho['first_success_rate']:.1f}% | 300.0 |")

        if "early_term_train" in res:
            ett = res["early_term_train"]
            eth = res["early_term_held_out"]
            md_lines.append(f"| **{name}** | Early Term | Train | {ett['pos_mean']:.2f} ± {ett['pos_std']:.2f} | {ett['rot_mean']:.2f} ± {ett['rot_std']:.2f}° | {ett['success_rate']:.1f}% | -- | {ett['steps_mean']:.1f} ± {ett['steps_std']:.1f} |")
            md_lines.append(f"| **{name}** | Early Term | Held-Out | {eth['pos_mean']:.2f} ± {eth['pos_std']:.2f} | {eth['rot_mean']:.2f} ± {eth['rot_std']:.2f}° | {eth['success_rate']:.1f}% | -- | {eth['steps_mean']:.1f} ± {eth['steps_std']:.1f} |")

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Saved JSON to {out_json}")
    print(f"[DONE] Saved Markdown to {out_md}")


if __name__ == "__main__":
    main()
