#!/usr/bin/env python3
"""
scripts/eval_image_only_ppo_fixed_horizon.py
============================================
Evaluates existing Image-Only PPO checkpoints in fixed-horizon mode (300 steps):
1. Aorta, MB, image-only PPO: 3 seeds (300,032 steps each)
   - Seed 0: ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip
   - Seed 1: ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip
   - Seed 2: ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip
2. Liver, MB, image-only PPO: 300,000 steps
   - ppo_checkpoints/liver_centre/ppo_scale_liver_300000_steps.zip

Protocol:
- Identical to existing spine image-only PPO fixed-horizon evaluation:
  * 100 trials Train cohort (seeds 1000-1099): s0011 (34), s0058 (33), s0250 (33)
  * 100 trials Unseen/Held-Out cohort (seeds 2000-2099): s0344 (50), s0358 (50)
- Horizon: 300 steps with NO early termination on goal success (early_termination=False).
- Out-of-bounds termination (probe leaves the skin boundary, current_is_valid=False) kept identical to spine.
- Success criterion at final step: pos_error < 5.0 mm and rot_error < 10.0 deg.

Reports:
- For each model and cohort: Succ (%), mean Pos (mm), mean Rot (deg), mean steps.
- For Aorta: reports each seed separately AND mean ± std over the 3 seeds.
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from stable_baselines3 import PPO
from robotic_us_env import RoboticUltrasoundGymEnv
from train_image_only_scale import ImageContactOrientationWrapper

TRAIN_PATIENTS = ["s0011"] * 34 + ["s0058"] * 33 + ["s0250"] * 33
TRAIN_SEEDS = [1000 + i for i in range(100)]

HELD_OUT_PATIENTS = ["s0344"] * 50 + ["s0358"] * 50
HELD_OUT_SEEDS = [2000 + i for i in range(100)]

MODELS = [
    {
        "target": "aorta",
        "name": "Aorta Image-Only PPO Seed 0",
        "seed_tag": 0,
        "ckpt": str(PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip"),
        "wrapper": "image_contact_orientation"
    },
    {
        "target": "aorta",
        "name": "Aorta Image-Only PPO Seed 1",
        "seed_tag": 1,
        "ckpt": str(PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip"),
        "wrapper": "image_contact_orientation"
    },
    {
        "target": "aorta",
        "name": "Aorta Image-Only PPO Seed 2",
        "seed_tag": 2,
        "ckpt": str(PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip"),
        "wrapper": "image_contact_orientation"
    },
    {
        "target": "liver",
        "name": "Liver Image-Only PPO (300k)",
        "seed_tag": 0,
        "ckpt": str(PROJECT_DIR / "ppo_checkpoints/liver_centre/ppo_scale_liver_300000_steps.zip"),
        "wrapper": "raw_env"
    }
]


def run_single_episode_worker(args):
    th.set_num_threads(1)
    ckpt_path, target, wrapper_type, patient, seed, max_steps = args
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient)

    model = PPO.load(ckpt_path, device="cpu")

    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="navigation",
        anatomy=target,
        sim_mode="conv",
        size=128,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=False,
        multi_patient=False,
        goal_conditioned=False,
        device="cpu"
    )

    if wrapper_type == "image_contact_orientation":
        w_env = ImageContactOrientationWrapper(env)
    else:
        w_env = env

    obs, info = w_env.reset(seed=seed)
    final_pos_err = float(info.get("pos_error_mm", 0.0))
    final_rot_err = float(info.get("rot_error_deg", 0.0))
    steps_taken = max_steps

    for step in range(1, max_steps + 1):
        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = w_env.step(action)

        final_pos_err = float(info.get("pos_error_mm", 0.0))
        final_rot_err = float(info.get("rot_error_deg", 0.0))

        if term or trunc:
            steps_taken = step
            break

    w_env.close()

    is_success = (final_pos_err < 5.0 and final_rot_err < 10.0)

    return {
        "patient": patient,
        "seed": seed,
        "success": is_success,
        "pos_err_mm": final_pos_err,
        "rot_err_deg": final_rot_err,
        "steps": steps_taken
    }


def evaluate_model(model_info, n_workers=8):
    name = model_info["name"]
    ckpt = model_info["ckpt"]
    target = model_info["target"]
    wrapper_type = model_info["wrapper"]

    print("\n" + "=" * 80)
    print(f"  EVALUATING: {name}")
    print(f"  Target: {target} | Checkpoint: {ckpt}")
    print(f"  Protocol: Fixed Horizon 300 steps (100 Train trials, 100 Held-Out trials)")
    print("=" * 80)

    results = {}

    for cohort, p_map, seeds in [("Train", TRAIN_PATIENTS, TRAIN_SEEDS), ("Held-Out", HELD_OUT_PATIENTS, HELD_OUT_SEEDS)]:
        tasks = [
            (ckpt, target, wrapper_type, p_map[i], seeds[i], 300)
            for i in range(100)
        ]

        t0 = time.time()
        with mp.Pool(processes=n_workers) as pool:
            ep_results = list(pool.imap_unordered(run_single_episode_worker, tasks))
        el = time.time() - t0

        succ_list = [1.0 if r["success"] else 0.0 for r in ep_results]
        pos_list = [r["pos_err_mm"] for r in ep_results]
        rot_list = [r["rot_err_deg"] for r in ep_results]
        steps_list = [r["steps"] for r in ep_results]

        succ_rate = float(np.mean(succ_list) * 100.0)
        pos_mean, pos_std = float(np.mean(pos_list)), float(np.std(pos_list))
        rot_mean, rot_std = float(np.mean(rot_list)), float(np.std(rot_list))
        steps_mean, steps_std = float(np.mean(steps_list)), float(np.std(steps_list))

        results[cohort] = {
            "success_rate": succ_rate,
            "pos_err_mean": pos_mean,
            "pos_err_std": pos_std,
            "rot_err_mean": rot_mean,
            "rot_err_std": rot_std,
            "steps_mean": steps_mean,
            "steps_std": steps_std,
            "eval_time_sec": el,
            "raw_episodes": ep_results
        }

        print(f"  [{cohort:8s}] Succ={succ_rate:5.1f}% | Pos={pos_mean:6.2f} ± {pos_std:5.2f} mm | Rot={rot_mean:5.2f} ± {rot_std:5.2f}° | Steps={steps_mean:5.1f} ± {steps_std:4.1f} ({el:.1f}s)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="Number of CPU worker processes")
    args = parser.parse_args()

    all_results = {}
    for m in MODELS:
        all_results[m["name"]] = {
            "model_info": m,
            "cohorts": evaluate_model(m, n_workers=args.workers)
        }

    # Aggregate Aorta over the 3 seeds
    aorta_keys = [m["name"] for m in MODELS if m["target"] == "aorta"]
    aorta_agg = {}
    for cohort in ["Train", "Held-Out"]:
        succ_seeds = [all_results[k]["cohorts"][cohort]["success_rate"] for k in aorta_keys]
        pos_seeds = [all_results[k]["cohorts"][cohort]["pos_err_mean"] for k in aorta_keys]
        rot_seeds = [all_results[k]["cohorts"][cohort]["rot_err_mean"] for k in aorta_keys]
        steps_seeds = [all_results[k]["cohorts"][cohort]["steps_mean"] for k in aorta_keys]

        aorta_agg[cohort] = {
            "succ_mean": float(np.mean(succ_seeds)),
            "succ_std": float(np.std(succ_seeds)),
            "pos_mean": float(np.mean(pos_seeds)),
            "pos_std": float(np.std(pos_seeds)),
            "rot_mean": float(np.mean(rot_seeds)),
            "rot_std": float(np.std(rot_seeds)),
            "steps_mean": float(np.mean(steps_seeds)),
            "steps_std": float(np.std(steps_seeds)),
        }
    all_results["Aorta Image-Only PPO (Mean ± Std over 3 Seeds)"] = aorta_agg

    # Save JSON and Markdown
    out_dir = PROJECT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "image_only_ppo_fixed_horizon_eval.json"
    md_path = out_dir / "image_only_ppo_fixed_horizon_eval.md"

    # Clean raw episodes for compact json
    compact_results = {}
    for k, v in all_results.items():
        if k == "Aorta Image-Only PPO (Mean ± Std over 3 Seeds)":
            compact_results[k] = v
        else:
            compact_results[k] = {
                "model_info": v["model_info"],
                "cohorts": {
                    c: {k2: v2 for k2, v2 in cdata.items() if k2 != "raw_episodes"}
                    for c, cdata in v["cohorts"].items()
                }
            }

    with open(json_path, "w") as f:
        json.dump(compact_results, f, indent=2)

    # Format Markdown Table
    lines = [
        "# Image-Only PPO Fixed-Horizon Evaluation Report",
        "",
        "Evaluation Protocol: Fixed-Horizon mode (300 steps with NO early termination on goal success).",
        "Out-of-bounds safety termination: probe leaves skin boundary (identical to spine evaluation).",
        "100 trials per cohort: Train seeds 1000-1099, Unseen/Held-Out seeds 2000-2099.",
        "Success criterion at final step: `pos_error < 5.0 mm` and `rot_error < 10.0 deg`.",
        "",
        "| Target | Model | Cohort | Succ (%) | Pos Error (mm) | Rot Error (deg) | Steps | Checkpoint |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for m in MODELS:
        name = m["name"]
        target = m["target"].capitalize()
        ckpt = m["ckpt"]
        for cohort in ["Train", "Held-Out"]:
            cdata = all_results[name]["cohorts"][cohort]
            succ_str = f"{cdata['success_rate']:.1f}%"
            pos_str = f"{cdata['pos_err_mean']:.2f} ± {cdata['pos_err_std']:.2f}"
            rot_str = f"{cdata['rot_err_mean']:.2f} ± {cdata['rot_err_std']:.2f}°"
            steps_str = f"{cdata['steps_mean']:.1f} ± {cdata['steps_std']:.1f}"
            lines.append(f"| {target} | {name} | {cohort} | {succ_str} | {pos_str} | {rot_str} | {steps_str} | `{ckpt}` |")

    lines.append("")
    lines.append("### Summary: Aorta (Mean ± Std across 3 Seeds)")
    lines.append("")
    lines.append("| Target | Summary Metric | Cohort | Succ (%) | Pos Error (mm) | Rot Error (deg) | Steps |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for cohort in ["Train", "Held-Out"]:
        agg = aorta_agg[cohort]
        lines.append(f"| Aorta | **Mean ± Std (3 Seeds)** | {cohort} | **{agg['succ_mean']:.1f} ± {agg['succ_std']:.1f}%** | **{agg['pos_mean']:.2f} ± {agg['pos_std']:.2f}** | **{agg['rot_mean']:.2f} ± {agg['rot_std']:.2f}°** | **{agg['steps_mean']:.1f} ± {agg['steps_std']:.1f}** |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n[DONE] Saved report to {md_path} and {json_path}")


if __name__ == "__main__":
    main()
