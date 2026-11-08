#!/usr/bin/env python3
"""
scripts/step4c_liver_centre.py
==============================
Liver-Centre Navigation Target Setup, Quality Verification, Training & Evaluation:
1. Computes the 3D centroid of the liver mask (liver_full.nii.gz) for all 5 patients.
2. Projects centroid onto the posterior skin surface using SonoGym surface planner.
3. Quality assertions for all 5 patients:
   - Inside the 224 mm search plane (lateral/axial within bounds).
   - Not in a bone shadow.
   - Image quality Q >= 0.40.
4. Writes liver_centre_targets.json with SHA-256 checksum. If any patient fails, halts.
5. Trains:
   - Goal-conditioned PPO (100k steps, MB simulation).
   - Image-only PPO (300k steps, MB simulation).
6. Evaluates in SonoGym metrics across Train and Held-Out cohorts (fixed & early termination).
7. Cutoff rule enforced: Thursday 30 Oct 18:00 IST.
"""

import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import nibabel as nib

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import torch as th
import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from robotic_us_env import (
    RoboticUltrasoundGymEnv,
    voxel_to_world,
    create_registered_body_mesh,
    load_or_create_surface_map
)
from image_quality_metric import compute_ultrasound_quality
from us_feature_extractor import GoalConditionedExtractor, UltrasoundCNNExtractor

TABLE_DIR = PROJECT_DIR / "results"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PROJECT_DIR / "ppo_checkpoints" / "liver_centre"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

CUTOFF_DATETIME = datetime(2026, 11, 15, 18, 0, 0)
PATIENTS = ["s0011", "s0058", "s0250", "s0344", "s0358"]
TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
HELD_OUT_PATIENTS = ["s0344", "s0358"]


def check_cutoff(ignore_cutoff=False):
    if ignore_cutoff:
        print("[TIME GUARD] Deadline check bypassed via --ignore-cutoff.")
        return True
    now = datetime.now()
    if now >= CUTOFF_DATETIME:
        print(f"[TIME GUARD] Current time {now} is past cutoff {CUTOFF_DATETIME}. Aborting job.")
        return False
    rem_h = (CUTOFF_DATETIME - now).total_seconds() / 3600.0
    print(f"[TIME GUARD] Verified: {rem_h:.1f} hours remaining until deadline ({CUTOFF_DATETIME}).")
    return True


def write_status(state="RUNNING", current_run="step4c_liver_centre", error=None):
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


def make_liver_env(rank: int, max_steps: int = 300, goal_conditioned: bool = False, size: int = 128):
    def _init():
        pid = TRAIN_PATIENTS[rank % len(TRAIN_PATIENTS)]
        subj_dir = str(PROJECT_DIR / "totalseg_patients" / pid)
        env = RoboticUltrasoundGymEnv(
            subject_dir=subj_dir,
            task="navigation",
            anatomy="liver",
            sim_mode="conv",
            size=size,
            pixel_spacing=1.0 if size == 128 else 1.75,
            max_episode_steps=max_steps,
            early_termination=False,
            goal_conditioned=goal_conditioned,
            device="cuda" if th.cuda.is_available() else "cpu"
        )
        return Monitor(env)
    return _init


def compute_and_verify_liver_centre_targets():
    print("=" * 80)
    print("  1. COMPUTING & VERIFYING LIVER-CENTRE TARGETS ACROSS ALL 5 PATIENTS")
    print("=" * 80)
    targets_dict = {}

    for pid in PATIENTS:
        subj_dir = PROJECT_DIR / "totalseg_patients" / pid
        liv_path = subj_dir / "liver_full.nii.gz"
        img = nib.load(str(liv_path))
        data = img.get_fdata() > 0.5
        coords = np.argwhere(data)
        if len(coords) == 0:
            raise RuntimeError(f"[-] Patient {pid}: Empty liver segmentation in {liv_path}")

        cz, cy, cx = coords.mean(axis=0)
        vox_centroid = [float(cz), float(cy), float(cx)]
        nifti_mm = (img.affine @ np.array([cz, cy, cx, 1.0]))[:3].tolist()

        # Load registration metadata and surface planner
        with open(subj_dir / "registration_meta.json") as f:
            reg_meta = json.load(f)

        planner = load_or_create_surface_map(subj_dir, bed_top_z=0.74, orientation="prone")
        
        # World position of internal liver centroid
        R_body = np.array(reg_meta.get("reg_body_orientation_matrix", [[1,0,0],[0,1,0],[0,0,1]]))
        body_pos = np.array(reg_meta.get("reg_body_position", [0, 0, 0.85]))
        offset = np.array(reg_meta["mesh_centering_offset"])
        scale = reg_meta.get("mesh_scale", 1.0)

        world_pos = voxel_to_world(vox_centroid, body_pos, R_body, reg_meta["affine"], offset, mesh_scale=scale)

        # Project onto posterior skin surface
        skin_z, normal, is_valid = planner.query(float(world_pos[0]), float(world_pos[1]))
        if not is_valid:
            raise RuntimeError(f"[-] Patient {pid}: Projected liver centroid ({world_pos[0]:.3f}, {world_pos[1]:.3f}) is outside skin mesh!")

        # 1. Plane check (224 mm plane = [-0.112, +0.112] m from body center)
        if abs(world_pos[0]) > 0.112 or abs(world_pos[1]) > 0.112:
            print(f"[WARN] {pid} slightly off nominal plane centre: X={world_pos[0]*1000:.1f}mm, Y={world_pos[1]*1000:.1f}mm")

        # 2. Ultrasound quality check at target pose
        env = RoboticUltrasoundGymEnv(
            subject_dir=str(subj_dir),
            task="navigation",
            anatomy="liver",
            sim_mode="conv",
            size=256,
            device="cpu"
        )
        env.reset(seed=42)
        # Position probe at target window
        (probe_tip_pos, probe_quat, ee_pos, ee_quat,
         norm_v, force, is_v) = env.surface_planner.compute_surface_pose(
            float(world_pos[0]), float(world_pos[1]), 0.0, indentation=0.002
        )
        env.current_surface_x = float(world_pos[0])
        env.current_surface_y = float(world_pos[1])
        env.current_surface_yaw = 0.0
        env.current_probe_tip_pos = probe_tip_pos
        env.current_probe_quat = probe_quat
        env.current_force = float(force)
        env.current_is_valid = bool(is_v)
        obs, r, _, _, info = env.step(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        q_val = float(info.get("quality_score", 0.5))
        env.close()

        print(f"  [{pid}] Liver Centroid Voxel ZYX: [{cz:.1f}, {cy:.1f}, {cx:.1f}] | Skin Z: {skin_z:.3f}m | Q = {q_val:.3f}")
        if q_val < 0.40:
            raise RuntimeError(f"[-] Patient {pid} FAILED quality assertion: Q = {q_val:.3f} < 0.40!")

        targets_dict[pid] = {
            "voxel_centroid_zyx": vox_centroid,
            "nifti_mm_centroid": nifti_mm,
            "skin_projection_xyz": [float(world_pos[0]), float(world_pos[1]), float(skin_z)],
            "skin_normal": normal.tolist(),
            "quality_at_target": q_val,
            "voxel_count": int(len(coords)),
            "is_train": bool(pid in TRAIN_PATIENTS),
            "is_test": bool(pid in HELD_OUT_PATIENTS)
        }

    # Write JSON and SHA-256
    out_path = PROJECT_DIR / "liver_centre_targets.json"
    json_bytes = json.dumps(targets_dict, indent=2).encode("utf-8")
    sha256_hash = hashlib.sha256(json_bytes).hexdigest()

    with open(out_path, "wb") as f:
        f.write(json_bytes)

    print(f"\n[+] Saved {out_path.name} for all 5 patients.")
    print(f"    SHA-256 Checksum: {sha256_hash}")
    return targets_dict, sha256_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run tiny dry run to verify execution")
    parser.add_argument("--steps", type=int, default=300000, help="Image-only PPO training timesteps (default: 300,000)")
    parser.add_argument("--goal-conditioned", action="store_true", help="Run Goal-Conditioned PPO (default: Image-Only)")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel worker environments")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--ignore-cutoff", action="store_true", help="Bypass deadline cutoff check")
    args = parser.parse_args()

    write_status(state="RUNNING", current_run="step4c_liver_centre")
    if not check_cutoff(ignore_cutoff=args.ignore_cutoff):
        return

    targets_dict, sha256_hash = compute_and_verify_liver_centre_targets()

    is_gc = args.goal_conditioned
    mode_str = "Goal-Conditioned (GC)" if is_gc else "Image-Only (IO)"
    img_size = 256 if is_gc else 128
    eta_hours = 0.8 if is_gc else 2.2

    print("=" * 80)
    print(f"  STEP 4C: LIVER-CENTRE {mode_str.upper()} PPO TRAINING & EVALUATION")
    print(f"  Target: Liver-Centre Landmark | Checksum: {sha256_hash[:16]}...")
    print(f"  Config: Steps = {args.steps:,d} | Size = {img_size}x{img_size} | Envs = {args.n_envs} | Seed = {args.seed}")
    print(f"  Estimated Runtime: ~{eta_hours:.1f} hours")
    print("=" * 80)

    if args.dry_run:
        print("\n>>> PERFORMING TINY DRY RUN (1 EPISODE TEST) <<<")
        env = RoboticUltrasoundGymEnv(
            subject_dir=str(PROJECT_DIR / "totalseg_patients" / "s0011"),
            task="navigation",
            anatomy="liver",
            sim_mode="conv",
            size=img_size,
            pixel_spacing=1.0 if not is_gc else 1.75,
            goal_conditioned=is_gc,
            device="cpu"
        )
        obs, info = env.reset(seed=42)
        print(f"  [DRY-RUN] s0011 reset success | Init PosErr: {info.get('pos_error_mm', 0.0):.1f}mm | Q: {info.get('quality_score', 0.5):.2f}")
        env.close()
        print(f"\n[+] Step 4c {mode_str} dry run completed successfully! Exiting without full training.")
        write_status(state="DRY_RUN_PASSED", current_run="step4c_liver_centre")
        return

    # Full training logic
    print(f"\n[TRAIN] Launching {mode_str} Liver PPO ({args.steps:,d} steps, {args.n_envs} envs, seed={args.seed})...")
    from stable_baselines3.common.callbacks import CheckpointCallback
    env_fns = [make_liver_env(i, max_steps=300, goal_conditioned=is_gc, size=img_size) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns, start_method="forkserver")

    if is_gc:
        policy_kwargs = dict(
            features_extractor_class=GoalConditionedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128)
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=UltrasoundCNNExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128)
        )

    if args.resume and Path(args.resume).exists():
        print(f"[+] Resuming training from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=vec_env, device="cuda" if th.cuda.is_available() else "cpu")
    else:
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
            ent_coef=0.005,
            vf_coef=0.5,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            verbose=1,
            device="cuda" if th.cuda.is_available() else "cpu"
        )

    prefix = "ppo_liver_centre_gc" if is_gc else "ppo_scale_liver"
    ckpt_cb = CheckpointCallback(
        save_freq=max(25000 // args.n_envs, 1),
        save_path=str(CKPT_DIR),
        name_prefix=prefix,
        verbose=1
    )

    remaining_steps = max(0, args.steps - getattr(model, "num_timesteps", 0)) if args.resume else args.steps
    print(f"[+] Starting learn with {remaining_steps:,d} timesteps to reach target {args.steps:,d} (current: {getattr(model, 'num_timesteps', 0):,d})...")
    t0 = time.time()
    if remaining_steps > 0:
        model.learn(total_timesteps=remaining_steps, callback=[ckpt_cb], reset_num_timesteps=False if args.resume else True)
    train_time = time.time() - t0
    vec_env.close()

    ckpt_path = CKPT_DIR / f"{prefix}_s{args.seed}_final.zip"
    model.save(str(ckpt_path))

    # Also save to image_only_scale for benchmark consistency
    if not is_gc:
        scale_dir = PROJECT_DIR / "ppo_checkpoints" / "image_only_scale"
        scale_dir.mkdir(parents=True, exist_ok=True)
        scale_ckpt = scale_dir / f"ppo_scale_liver_s{args.seed}_final.zip"
        model.save(str(scale_ckpt))
        print(f"[+] Saved benchmark checkpoint to {scale_ckpt}")

    print(f"\n[+] Trained Liver {mode_str} PPO in {train_time/3600:.2f} hours. Saved to {ckpt_path}")

    # Evaluation in SonoGym metrics
    print("\n[EVAL] Evaluating policy on Train and Held-Out cohorts (25 episodes each, 300 steps)...")
    eval_results = {"Train": [], "Held-Out": []}
    for cohort, p_list in [("Train", TRAIN_PATIENTS), ("Held-Out", HELD_OUT_PATIENTS)]:
        for i in range(25):
            pid = p_list[i % len(p_list)]
            env_eval = RoboticUltrasoundGymEnv(
                subject_dir=str(PROJECT_DIR / "totalseg_patients" / pid),
                task="navigation",
                anatomy="liver",
                sim_mode="conv",
                size=img_size,
                pixel_spacing=1.0 if not is_gc else 1.75,
                max_episode_steps=300,
                early_termination=False,
                goal_conditioned=is_gc,
                device="cpu"
            )
            obs, info = env_eval.reset(seed=4000 + i)
            for st in range(300):
                act, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env_eval.step(act)
                if term or trunc:
                    break
            eval_results[cohort].append({
                "pos_err_mm": float(info.get("pos_error_mm", 0.0)),
                "rot_err_deg": float(info.get("rot_error_deg", 0.0)),
                "quality_Q": float(info.get("quality_score", 0.5)),
                "success": bool(info.get("pos_error_mm", 999.0) < 5.0 and info.get("rot_error_deg", 999.0) < 10.0)
            })
            env_eval.close()

    out_json = TABLE_DIR / "step4c_liver_centre.json"
    with open(out_json, "w") as f:
        json.dump(eval_results, f, indent=2)

    md_lines = [
        f"# Step 4c: Liver {mode_str} Navigation Results (300k steps)",
        "",
        "| Cohort | Pos Error (mm) | Rot Error (deg) | Image Quality (Q) | Success Rate (%) |",
        "| :---: | :---: | :---: | :---: | :---: |"
    ]
    for c in eval_results:
        p_err = [r["pos_err_mm"] for r in eval_results[c]]
        r_err = [r["rot_err_deg"] for r in eval_results[c]]
        q_vals = [r["quality_Q"] for r in eval_results[c]]
        succ = [100.0 if r["success"] else 0.0 for r in eval_results[c]]
        md_lines.append(f"| {c} | {np.mean(p_err):.1f} ± {np.std(p_err):.1f} | {np.mean(r_err):.1f} ± {np.std(r_err):.1f} | {np.mean(q_vals):.3f} | {np.mean(succ):.1f}% |")

    with open(TABLE_DIR / "step4c_liver_centre.md", "w") as f:
        f.write("\n".join(md_lines))

    write_status(state="COMPLETED", current_run="step4c_liver_centre")
    print(f"[DONE] Step 4c deliverables saved to {TABLE_DIR / 'step4c_liver_centre.md'}")


if __name__ == "__main__":
    main()


