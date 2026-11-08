#!/usr/bin/env python3
"""
scripts/re_evaluate_aorta_liver_rotation.py
===========================================
Re-evaluates (NOT retrains) existing reconstruction checkpoints for Aorta and Liver
with total_rotation_deg logged (sum of absolute yaw changes per episode).

Methods: PPO+Memory, PPO (No Memory)
Targets: Aorta, Liver
Protocol:
- 10 random starts per patient: seed = 42 + seed_idx * 17 for seed_idx in range(10)
- Training patients: s0011, s0058, s0250 (30 episodes per method)
- Held-out patients: s0344, s0358 (20 episodes per method)
- Horizon: 300 steps
- Exact same environment settings as step2_recon_eval.py (sim_mode="conv", size=128, roi_radius=inf)
"""

import os
import sys
import json
import time
import multiprocessing as mp
import numpy as np
import torch as th

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

PROJECT_DIR = Path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from scripts.step2_recon_eval import run_single_episode_worker, METHOD_NAMES

TARGETS = ["aorta", "liver"]
METHODS = ["ppo_memory", "ppo_nomem"]
TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
HELD_OUT_PATIENTS = ["s0344", "s0358"]
SEEDS = [42 + i * 17 for i in range(10)]

def main():
    tasks = []
    for target in TARGETS:
        for method in METHODS:
            for patient in TRAIN_PATIENTS + HELD_OUT_PATIENTS:
                for seed in SEEDS:
                    tasks.append((target, patient, method, seed, 300))

    print(f"Total tasks to run: {len(tasks)} (using 8 parallel workers)...")
    t0 = time.time()

    with mp.Pool(processes=8) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(run_single_episode_worker, tasks), 1):
            results.append(res)
            if i % 20 == 0 or i == len(tasks):
                print(f"  Progress: [{i}/{len(tasks)}] finished in {time.time()-t0:.1f}s | {res['target']} {res['patient']} {res['method']} (rot: {res['total_rotation_deg']:.1f}°)")

    total_time = time.time() - t0
    print(f"\nAll {len(tasks)} episodes finished in {total_time:.1f}s ({total_time/60:.2f} min).")

    # Aggregate by Target, Method, and Cohort
    summary = {}
    md_lines = [
        "# Aorta and Liver Reconstruction Re-Evaluation (with Total Rotation Logged)",
        "",
        "Protocol: 10 random starts per patient (seeds: `42 + 17*i`), 300 steps.",
        "Checkpoints evaluated: identical existing checkpoints from `ppo_checkpoints/block_b/`.",
        "",
        "| Target | Method | Cohort | Upper Coverage (%) | Path Length (m) | Total Rotation (deg) | Existing Table Coverage (%) | Reproduces? |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    # Reference coverage from existing table to check reproducibility
    REF_TABLE = {
        ("aorta", "ppo_memory", "Train"): (11.9, 0.40),
        ("aorta", "ppo_memory", "Held-Out"): (23.7, 0.12),
        ("aorta", "ppo_nomem", "Train"): (14.8, 0.42),
        ("aorta", "ppo_nomem", "Held-Out"): (33.5, 0.33),
        ("liver", "ppo_memory", "Train"): (42.1, 1.29),
        ("liver", "ppo_memory", "Held-Out"): (54.2, 1.59),
        ("liver", "ppo_nomem", "Train"): (47.7, 1.38),
        ("liver", "ppo_nomem", "Held-Out"): (66.0, 0.79),
    }

    for target in TARGETS:
        summary[target] = {}
        for method in METHODS:
            summary[target][method] = {}
            for cohort, p_list in [("Train", TRAIN_PATIENTS), ("Held-Out", HELD_OUT_PATIENTS)]:
                records = [r for r in results if r["target"] == target and r["method"] == method and r["patient"] in p_list]
                u_cov = [r["final_upper_cov"] for r in records]
                p_len = [r["path_len_m"] for r in records]
                rot = [r["total_rotation_deg"] for r in records]

                u_mean, u_std = float(np.mean(u_cov)), float(np.std(u_cov))
                p_mean, p_std = float(np.mean(p_len)), float(np.std(p_len))
                rot_mean, rot_std = float(np.mean(rot)), float(np.std(rot))

                ref_cov, ref_path = REF_TABLE.get((target, method, cohort), (None, None))
                # Consider reproduced if mean matches within 0.1%
                reproduced = (abs(u_mean - ref_cov) <= 0.1) if ref_cov is not None else False

                summary[target][method][cohort] = {
                    "upper_coverage_mean": u_mean,
                    "upper_coverage_std": u_std,
                    "path_length_mean": p_mean,
                    "path_length_std": p_std,
                    "total_rotation_mean": rot_mean,
                    "total_rotation_std": rot_std,
                    "ref_upper_coverage": ref_cov,
                    "ref_path_length": ref_path,
                    "reproduced": reproduced
                }

                m_name = METHOD_NAMES[method]
                repro_str = "YES" if reproduced else f"Diff ({u_mean:.1f}% vs {ref_cov:.1f}%)"
                md_lines.append(f"| {target.capitalize()} | {m_name} | {cohort} | {u_mean:.1f} ± {u_std:.1f} | {p_mean:.2f} ± {p_std:.2f} | **{rot_mean:.1f} ± {rot_std:.1f}** | {ref_cov:.1f}% | {repro_str} |")

    out_json = os.path.join(PROJECT_DIR, "results/aorta_liver_rotation_reeval.json")
    out_md = os.path.join(PROJECT_DIR, "results/aorta_liver_rotation_reeval.md")

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Saved summary to {out_md} and {out_json}")

if __name__ == "__main__":
    main()
