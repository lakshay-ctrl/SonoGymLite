#!/usr/bin/env python3
"""
scripts/step3_nav_eval.py
=========================
Navigation Benchmark Evaluation in SonoGym Metrics (NeurIPS 2025):
- Models:
  * Goal-conditioned PPO (spine, aorta, liver)
  * Goal-conditioned SAC (spine, aorta, liver)
  * Image-only PPO (spine, aorta, liver)
- Cohorts:
  * Train: s0011, s0058, s0250 (100 trials total = 34 + 33 + 33)
  * Held-Out: s0344, s0358 (100 trials total = 50 + 50)
- Episode Horizon: 300 steps, deterministic policy.
- Two termination modes evaluated for each setting:
  (a) Fixed horizon: early_termination = False
  (b) Early termination: early_termination = True (stops on success)
- Metrics (Mean ± Std at final step):
  * Skin-plane position error (mm)
  * Rotation error (deg) = env.rot_error_deg (beam-to-skin-normal angle)
  * Success rate (%) = position < 5 mm and rotation < 10 deg
  * 3D error (mm)
  * Steps to first success
- Outputs:
  * results/navigation_benchmark_table.md
  * results/navigation_benchmark_table.json
  * results/status.json
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import torch as th
import gymnasium

# Safe globals for PyTorch 2.6+
try:
    th.serialization.add_safe_globals([
        gymnasium.spaces.dict.Dict,
        gymnasium.spaces.box.Box,
        gymnasium.spaces.discrete.Discrete,
        np.dtype,
        np.core.multiarray.scalar,
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.UInt8DType,
    ])
except Exception:
    pass

from stable_baselines3 import PPO, SAC
from robotic_us_env import RoboticUltrasoundGymEnv
from us_feature_extractor import GoalConditionedExtractor, UltrasoundCNNExtractor

TABLE_DIR = PROJECT_DIR / "results"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["spine", "aorta", "liver"]
TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
HELD_OUT_PATIENTS = ["s0344", "s0358"]

CHECKPOINTS = {
    "spine": {
        "gc_ppo": PROJECT_DIR / "ppo_checkpoints/spine_mb_goal_100k/ppo_mb_spine_final_model.zip",
        "gc_sac": PROJECT_DIR / "sac_checkpoints/spine_mb/sac_mb_spine_final_model.zip",
        "image_only_ppo": PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_spine_s0_final.zip"
    },
    "aorta": {
        "gc_ppo": PROJECT_DIR / "ppo_checkpoints/aorta_mb/ppo_mb_aorta_final_model.zip",
        "gc_sac": PROJECT_DIR / "sac_checkpoints/aorta_mb/sac_mb_aorta_final_model.zip",
        "image_only_ppo": PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip",
        "image_only_ppo_s0": PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip",
        "image_only_ppo_s2": PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip"
    },
    "liver": {
        "gc_ppo": PROJECT_DIR / "ppo_checkpoints/liver_mb/ppo_mb_liver_final_model.zip",
        "gc_sac": PROJECT_DIR / "sac_checkpoints/liver_mb/sac_mb_liver_final_model.zip"
        # Old 100k liver model excluded; 300k model scheduled in Step 4c
    }
}


def run_single_nav_trial(args):
    th.set_num_threads(1)
    target, patient, model_key, early_term, seed, max_steps = args
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient)
    
    if model_key not in CHECKPOINTS[target]:
        return None
    ckpt_path = CHECKPOINTS[target][model_key]
    if not Path(ckpt_path).exists():
        # Fallback search if exact file differs
        alt = list(Path(ckpt_path).parent.glob("*.zip"))
        if alt:
            ckpt_path = alt[0]
        else:
            return None

    if "sac" in model_key:
        model = SAC.load(str(ckpt_path), device="cpu")
    else:
        model = PPO.load(str(ckpt_path), device="cpu")

    img_size = 256
    if hasattr(model.observation_space, "spaces") and "image" in model.observation_space.spaces:
        img_size = int(model.observation_space.spaces["image"].shape[0])

    is_image_only = model_key.startswith("image_only")
    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="navigation",
        anatomy=target,
        sim_mode="conv",
        size=img_size,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=early_term,
        goal_conditioned=(not is_image_only),
        device="cpu"
    )

    obs, info = env.reset(seed=seed)
    if is_image_only:
        c_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
        quat = np.array(env.unwrapped.current_probe_quat, dtype=np.float32)
        obs["contact"] = np.array([c_val], dtype=np.float32)
        obs["orientation"] = quat

    first_success_step = max_steps
    success_achieved = False

    final_pos_err = 0.0
    final_rot_err = 0.0
    final_3d_err = 0.0

    for step in range(1, max_steps + 1):
        if is_image_only:
            c_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
            quat = np.array(env.unwrapped.current_probe_quat, dtype=np.float32)
            obs["contact"] = np.array([c_val], dtype=np.float32)
            obs["orientation"] = quat
        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if is_image_only:
            c_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
            quat = np.array(env.unwrapped.current_probe_quat, dtype=np.float32)
            obs["contact"] = np.array([c_val], dtype=np.float32)
            obs["orientation"] = quat

        pos_err_mm = float(info.get("pos_error_mm", 0.0))
        rot_err_deg = float(info.get("rot_error_deg", 0.0))
        err_3d_mm = float(info.get("error_3d_mm", 0.0))

        if pos_err_mm < 5.0 and rot_err_deg < 10.0:
            if not success_achieved:
                success_achieved = True
                first_success_step = step

        final_pos_err = pos_err_mm
        final_rot_err = rot_err_deg
        final_3d_err = err_3d_mm

        if term or trunc:
            break

    env.close()

    return {
        "target": target,
        "patient": patient,
        "model_key": model_key,
        "early_term": early_term,
        "seed": seed,
        "pos_err_mm": final_pos_err,
        "rot_err_deg": final_rot_err,
        "err_3d_mm": final_3d_err,
        "success": bool(final_pos_err < 5.0 and final_rot_err < 10.0),
        "steps_to_first_success": first_success_step,
        "steps_taken": step
    }


def write_status(state="RUNNING", current_run="navigation_evaluation", error=None):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run tiny dry run (2 trials) to verify script runs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU worker processes")
    parser.add_argument("--steps", type=int, default=300, help="Episode horizon")
    args = parser.parse_args()

    write_status(state="RUNNING", current_run="navigation_evaluation")

    if args.dry_run:
        print("\n>>> PERFORMING TINY DRY RUN FOR NAVIGATION EVALUATION (2 TRIALS) <<<")
        dry_tasks = [
            ("spine", "s0011", "gc_ppo", False, 1001, 50),
            ("spine", "s0344", "gc_sac", True, 1002, 50)
        ]
        for task in dry_tasks:
            t0 = time.time()
            res = run_single_nav_trial(task)
            el = time.time() - t0
            print(f"  [DRY-RUN] {res['target']} | {res['patient']} | {res['model_key']} | EarlyTerm={res['early_term']} -> PosErr: {res['pos_err_mm']:.1f}mm | RotErr: {res['rot_err_deg']:.1f}° | 3DErr: {res['err_3d_mm']:.1f}mm | Success: {res['success']} ({el:.1f}s)")
        print("\n[+] Navigation dry run completed successfully! Exiting without launching full 100-trial sweep.")
        write_status(state="DRY_RUN_PASSED", current_run="navigation_evaluation")
        return

    # Full Evaluation Suite:
    # 100 trials per cohort (Train: 3 patients -> 34, 33, 33; Held-out: 2 patients -> 50, 50)
    # 2 modes: fixed horizon (early_term=False), early termination (early_term=True)
    modes = [False, True]

    tasks = []
    for target in TARGETS:
        for model_key in CHECKPOINTS[target].keys():
            if not Path(CHECKPOINTS[target][model_key]).exists():
                print(f"[WARN] Checkpoint {CHECKPOINTS[target][model_key]} not found. Skipping {target} {model_key}.")
                continue
            for early_term in modes:
                # Train cohort (100 trials)
                for i in range(100):
                    pid = TRAIN_PATIENTS[i % len(TRAIN_PATIENTS)]
                    seed = 1000 + i * 7
                    tasks.append((target, pid, model_key, early_term, seed, args.steps))
                # Held-out cohort (100 trials)
                for i in range(100):
                    pid = HELD_OUT_PATIENTS[i % len(HELD_OUT_PATIENTS)]
                    seed = 5000 + i * 7
                    tasks.append((target, pid, model_key, early_term, seed, args.steps))

    cache_file = TABLE_DIR / "navigation_evaluation_raw.jsonl"
    existing_trials = []
    existing_keys = set()
    if cache_file.exists():
        with open(cache_file) as f:
            for l in f:
                if l.strip():
                    try:
                        rec = json.loads(l)
                        existing_trials.append(rec)
                        existing_keys.add((rec["target"], rec["patient"], rec["model_key"], rec["early_term"], rec["seed"]))
                    except Exception:
                        pass

    pending_tasks = [t for t in tasks if (t[0], t[1], t[2], t[3], t[4]) not in existing_keys]
    print(f"\n[SWEEP] Total trials: {len(tasks)} ({len(existing_trials)} cached, {len(pending_tasks)} remaining). Launching using {args.workers} workers...")
    results = list(existing_trials)

    if len(pending_tasks) > 0:
        with mp.Pool(processes=args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_single_nav_trial, pending_tasks), 1):
                if res is not None:
                    results.append(res)
                    with open(cache_file, "a") as f:
                        f.write(json.dumps(res) + "\n")
                if i % 50 == 0 or i == len(pending_tasks):
                    print(f"  Progress: [{i}/{len(pending_tasks)}] new trials completed ({len(results)}/{len(tasks)} total)")

    # Aggregate Metrics into Markdown and JSON
    table_dict = {}
    md_lines = [
        "# SonoGym Navigation Benchmark Evaluation Table",
        "",
        "| Target | Method | Early Term | Cohort | Pos Error (mm) | Rot Error (deg) | Success Rate (%) | 3D Error (mm) | Steps to Success |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for target in TARGETS:
        table_dict[target] = {}
        for model_key in CHECKPOINTS[target].keys():
            table_dict[target][model_key] = {}
            for early_term in modes:
                et_str = "ON" if early_term else "OFF"
                table_dict[target][model_key][et_str] = {}
                for cohort, p_list in [("Train", TRAIN_PATIENTS), ("Held-Out", HELD_OUT_PATIENTS)]:
                    subset = [r for r in results if r["target"] == target and r["model_key"] == model_key and r["early_term"] == early_term and r["patient"] in p_list]
                    if len(subset) == 0:
                        continue
                    p_err = [r["pos_err_mm"] for r in subset]
                    r_err = [r["rot_err_deg"] for r in subset]
                    succ = [100.0 if r["success"] else 0.0 for r in subset]
                    err_3d = [r["err_3d_mm"] for r in subset]
                    s_first = [r["steps_to_first_success"] for r in subset]

                    p_str = f"{np.mean(p_err):.1f} ± {np.std(p_err):.1f}"
                    r_str = f"{np.mean(r_err):.1f} ± {np.std(r_err):.1f}"
                    s_str = f"{np.mean(succ):.1f}%"
                    e3_str = f"{np.mean(err_3d):.1f} ± {np.std(err_3d):.1f}"
                    step_str = f"{np.mean(s_first):.1f} ± {np.std(s_first):.1f}"

                    table_dict[target][model_key][et_str][cohort] = {
                        "pos_error_mean": float(np.mean(p_err)),
                        "pos_error_std": float(np.std(p_err)),
                        "rot_error_mean": float(np.mean(r_err)),
                        "rot_error_std": float(np.std(r_err)),
                        "success_rate": float(np.mean(succ)),
                        "error_3d_mean": float(np.mean(err_3d)),
                        "error_3d_std": float(np.std(err_3d)),
                        "steps_to_success_mean": float(np.mean(s_first)),
                        "steps_to_success_std": float(np.std(s_first)),
                        "n_trials": len(subset)
                    }

                    label_map = {
                        "gc_ppo": "Goal-PPO",
                        "gc_sac": "Goal-SAC",
                        "image_only_ppo": "Image-Only PPO (300k, Seed 0)" if target == "spine" else "Image-Only PPO (300k, Seed 1)",
                        "image_only_ppo_s0": "Image-Only PPO (300k, Seed 0)",
                        "image_only_ppo_s2": "Image-Only PPO (300k, Seed 2)"
                    }
                    m_label = label_map.get(model_key, model_key)
                    md_lines.append(f"| {target.capitalize()} | {m_label} | {et_str} | {cohort} | {p_str} | {r_str} | {s_str} | {e3_str} | {step_str} |")

    table_md = "\n".join(md_lines)
    with open(TABLE_DIR / "navigation_benchmark_table.md", "w") as f:
        f.write(table_md)
    with open(TABLE_DIR / "navigation_benchmark_table.json", "w") as f:
        json.dump(table_dict, f, indent=2)
    print(f"\n[DONE] Navigation tables written to {TABLE_DIR / 'navigation_benchmark_table.md'}")

    write_status(state="COMPLETED", current_run="navigation_evaluation")


if __name__ == "__main__":
    main()
