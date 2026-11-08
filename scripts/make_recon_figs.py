#!/usr/bin/env python3
"""
scripts/make_recon_figs.py
==========================
SonoGym-Style Multi-Organ 3D Reconstruction Visualization & Statistical Benchmark:
1. Pure L1-L5 Spine verification & rebuilding (strictly excludes sacrum, hips, ribs).
2. 10-episode statistical evaluation across 5 patients (3 train, 2 held-out) with random starts.
3. Organ-aligned Footprint Raster policy + short raster baseline with footprint reach fraction.
4. High-impact publication 3-panel figures (cropped to organ, large text, no colorbars, visible surface layer).
5. Comprehensive JSON and Markdown deliverables.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
from skimage import measure
import pyvista as pv
import nibabel as nib

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from stable_baselines3 import PPO
from robotic_us_env import RoboticUltrasoundGymEnv
from us_feature_extractor import OccupancyConditionedExtractor, GoalConditionedExtractor

OUTPUT_DIR = PROJECT_DIR / "results" / "recon_vis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["spine", "aorta", "liver"]
TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
HELD_OUT_PATIENTS = ["s0344", "s0358"]
ALL_PATIENTS = TRAIN_PATIENTS + HELD_OUT_PATIENTS

METHODS = ["ppo_memory", "ppo_nomem", "raster_footprint", "raster_short"]
METHOD_NAMES = {
    "ppo_memory": "PPO+Memory",
    "ppo_nomem": "PPO (No Memory)",
    "raster_footprint": "Footprint Raster",
    "raster_short": "Short Raster"
}

CHECKPOINTS = {
    "spine": {
        "ppo_memory": PROJECT_DIR / "ppo_checkpoints/block_b/spine_ppo_memory_whole_s0/final_model.zip",
        "ppo_nomem": PROJECT_DIR / "ppo_checkpoints/block_b/spine_ppo_nomem_whole_s0/final_model.zip"
    },
    "aorta": {
        "ppo_memory": PROJECT_DIR / "ppo_checkpoints/block_b/aorta_ppo_memory_whole_s0/final_model.zip",
        "ppo_nomem": PROJECT_DIR / "ppo_checkpoints/block_b/aorta_ppo_nomem_whole_s0/final_model.zip"
    },
    "liver": {
        "ppo_memory": PROJECT_DIR / "ppo_checkpoints/block_b/liver_ppo_memory_whole_s0/final_model.zip",
        "ppo_nomem": PROJECT_DIR / "ppo_checkpoints/block_b/liver_ppo_nomem_whole_s0/final_model.zip"
    }
}


def verify_and_rebuild_spine_labels():
    """
    Checks TotalSegmentator labels for spine across all 5 patients.
    Rebuilds spine_full.nii.gz to strictly contain vertebrae L1-L5.
    """
    print("=" * 80)
    print("  1. TOTALEPISODE SPINE MESH AUDIT & L1-L5 REBUILD")
    print("=" * 80)
    audit_results = {}

    for pid in ALL_PATIENTS:
        p_dir = PROJECT_DIR / "totalseg_patients" / pid
        s_full_path = p_dir / "spine_full.nii.gz"
        backup_path = p_dir / "spine_full_with_sacrum.nii.gz"
        scratch_dir = PROJECT_DIR / "scratch" / "totalseg_official" / pid

        if not backup_path.exists() and s_full_path.exists():
            import shutil
            shutil.copy2(str(s_full_path), str(backup_path))

        ref_img = nib.load(str(scratch_dir / "vertebrae_L1.nii.gz"))
        affine = ref_img.affine
        header = ref_img.header

        l1 = nib.load(str(scratch_dir / "vertebrae_L1.nii.gz")).get_fdata() > 0.5
        l2 = nib.load(str(scratch_dir / "vertebrae_L2.nii.gz")).get_fdata() > 0.5
        l3 = nib.load(str(scratch_dir / "vertebrae_L3.nii.gz")).get_fdata() > 0.5
        l4 = nib.load(str(scratch_dir / "vertebrae_L4.nii.gz")).get_fdata() > 0.5
        l5 = nib.load(str(scratch_dir / "vertebrae_L5.nii.gz")).get_fdata() > 0.5
        sac = nib.load(str(scratch_dir / "sacrum.nii.gz")).get_fdata() > 0.5
        t12 = nib.load(str(scratch_dir / "vertebrae_T12.nii.gz")).get_fdata() > 0.5

        pure_l15 = (l1 | l2 | l3 | l4 | l5).astype(np.uint8)

        # Write pure L1-L5 mask
        out_img = nib.Nifti1Image(pure_l15, affine=affine, header=header)
        nib.save(out_img, str(s_full_path))

        audit_results[pid] = {
            "l1_vox": int(np.sum(l1)),
            "l2_vox": int(np.sum(l2)),
            "l3_vox": int(np.sum(l3)),
            "l4_vox": int(np.sum(l4)),
            "l5_vox": int(np.sum(l5)),
            "sacrum_vox": int(np.sum(sac)),
            "t12_vox": int(np.sum(t12)),
            "pure_l15_vox": int(np.sum(pure_l15)),
            "sacrum_excluded": True
        }

        print(f"[{pid}] Verified L1-L5: {np.sum(pure_l15):,} voxels | Excluded Sacrum: {np.sum(sac):,} voxels | T12: 0")

    return audit_results


def extract_organ_mesh_and_ref_voxels(env, anatomy):
    """
    Extracts:
    1. Full triangle mesh via Marching Cubes in world coordinates in mm.
    2. Reference surface voxels (ref_voxels_mm) in world coordinates in mm.
    3. Upper-surface subset (transducer ceiling) in world coordinates in mm.
    """
    lbl = env.label_volume.cpu().numpy() if hasattr(env.label_volume, "cpu") else np.asarray(env.label_volume)
    affine = env.reg_meta['affine']
    offset = env.reg_meta['mesh_centering_offset']
    scale = env.mesh_scale
    R_body = env.reg_body_orientation_matrix
    body_pos = env.reg_body_position

    # 1. Marching Cubes Mesh
    verts_vox, faces, _, _ = measure.marching_cubes(lbl, level=0.5)
    v_hom = np.hstack([verts_vox, np.ones((len(verts_vox), 1), dtype=np.float64)])
    p_mm = (affine @ v_hom.T).T[:, :3]
    p_world = (R_body @ ((p_mm / 1000.0 - offset) * scale).T).T + body_pos
    mesh_verts_mm = p_world * 1000.0

    n_faces = len(faces)
    faces_pv = np.empty((n_faces, 4), dtype=np.int64)
    faces_pv[:, 0] = 3
    faces_pv[:, 1:] = faces
    pv_mesh = pv.PolyData(mesh_verts_mm, faces_pv.ravel())

    # 2. Reference surface voxels (boundary edge)
    edge = (lbl > 0.5) & ~binary_erosion(lbl > 0.5)
    edge_vox = np.argwhere(edge).astype(np.float64)
    vox_hom = np.hstack([edge_vox, np.ones((len(edge_vox), 1), dtype=np.float64)])
    p_ref_mm = (affine @ vox_hom.T).T[:, :3]
    ref_voxels_m = (R_body @ ((p_ref_mm / 1000.0 - offset) * scale).T).T + body_pos
    ref_voxels_mm = ref_voxels_m * 1000.0

    # 3. Upper-surface subset: maximum Z per (X, Y) column at 1.5mm resolution
    res_xy = 1.5  # mm
    xy_keys = np.round(ref_voxels_mm[:, :2] / res_xy).astype(int)
    top_vox_dict = {}
    for idx, (kx, ky) in enumerate(xy_keys):
        z_val = ref_voxels_mm[idx, 2]
        if (kx, ky) not in top_vox_dict or z_val > top_vox_dict[(kx, ky)][0]:
            top_vox_dict[(kx, ky)] = (z_val, idx)
    upper_indices = np.array([val[1] for val in top_vox_dict.values()], dtype=int)
    upper_surface_voxels_mm = ref_voxels_mm[upper_indices]

    return pv_mesh, ref_voxels_mm, upper_surface_voxels_mm


class FootprintRasterPolicy:
    """
    Organ-aligned Footprint Raster Policy:
    Sweeps the full projected bounding box of the organ on the patient skin.
    Advances along Y to maximize footprint coverage within the step budget.
    """
    def __init__(self, x_min_mm, x_max_mm, y_min_mm, y_max_mm, y_pitch_mm=15.0):
        # Margin of 4mm inside footprint boundaries
        self.x_min_m = (x_min_mm + 4.0) / 1000.0
        self.x_max_m = (x_max_mm - 4.0) / 1000.0
        self.y_min_m = y_min_mm / 1000.0
        self.y_max_m = y_max_mm / 1000.0
        self.y_pitch_m = y_pitch_mm / 1000.0
        self.dir_x = 1.0
        self.dir_y = 1.0
        self.target_y = None
        self.advancing_y = False

    def predict(self, current_x_m, current_y_m):
        if self.target_y is None:
            self.target_y = current_y_m

        if self.advancing_y:
            dy_err = self.target_y - current_y_m
            if abs(dy_err) < 0.003:
                self.advancing_y = False
                return np.array([self.dir_x, 0.0, 0.0], dtype=np.float32)
            else:
                act_y = float(np.clip(dy_err / 0.005, -1.0, 1.0))
                return np.array([0.0, act_y, 0.0], dtype=np.float32)
        else:
            if self.dir_x > 0 and current_x_m >= self.x_max_m:
                self.dir_x = -1.0
                self.advancing_y = True
                next_y = current_y_m + self.dir_y * self.y_pitch_m
                if next_y > self.y_max_m:
                    self.dir_y = -1.0
                    next_y = current_y_m + self.dir_y * self.y_pitch_m
                elif next_y < self.y_min_m:
                    self.dir_y = 1.0
                    next_y = current_y_m + self.dir_y * self.y_pitch_m
                self.target_y = next_y
                return np.array([0.0, float(self.dir_y), 0.0], dtype=np.float32)
            elif self.dir_x < 0 and current_x_m <= self.x_min_m:
                self.dir_x = 1.0
                self.advancing_y = True
                next_y = current_y_m + self.dir_y * self.y_pitch_m
                if next_y > self.y_max_m:
                    self.dir_y = -1.0
                    next_y = current_y_m + self.dir_y * self.y_pitch_m
                elif next_y < self.y_min_m:
                    self.dir_y = 1.0
                    next_y = current_y_m + self.dir_y * self.y_pitch_m
                self.target_y = next_y
                return np.array([0.0, float(self.dir_y), 0.0], dtype=np.float32)
            else:
                return np.array([self.dir_x, 0.0, 0.0], dtype=np.float32)


def run_single_episode_worker(args):
    """Worker function for running one episode in an isolated process."""
    import torch
    torch.set_num_threads(1)
    target, patient, method, seed, max_steps = args
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient)
    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="reconstruction",
        anatomy=target,
        sim_mode="conv",
        size=128,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=False,
        roi_radius=float("inf"),
        device="cpu"
    )

    obs, _ = env.reset(seed=seed)
    pv_mesh, ref_voxels_mm, upper_surface_voxels_mm = extract_organ_mesh_and_ref_voxels(env, target)
    b = pv_mesh.bounds
    tree_upper = cKDTree(upper_surface_voxels_mm)
    tree_ref = cKDTree(ref_voxels_mm)

    model = None
    raster_footprint = None
    if method in ("ppo_memory", "ppo_nomem"):
        ckpt = CHECKPOINTS[target][method]
        model = PPO.load(str(ckpt), device="cpu")
    elif method == "raster_footprint":
        raster_footprint = FootprintRasterPolicy(b[0], b[1], b[2], b[3], y_pitch_mm=16.0)

    probe_xyz = [env.current_probe_tip_pos.copy() * 1000.0]
    coverage_curve = [0.0]
    covered_upper_mask = np.zeros(len(upper_surface_voxels_mm), dtype=bool)
    seen_voxels = set()
    y_visited_mm = [env.current_probe_tip_pos[1] * 1000.0]

    for step in range(max_steps):
        if method == "raster_short":
            dx = 1.0 if (step // 20) % 2 == 0 else -1.0
            dy = 0.6 if step % 20 == 0 else 0.0
            action = np.array([dx, dy, 0.0], dtype=np.float32)
        elif method == "raster_footprint":
            action = raster_footprint.predict(env.current_surface_x, env.current_surface_y)
        else:
            m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
            action, _ = model.predict(m_obs, deterministic=True)

        obs, r, term, trunc, info = env.step(action)
        curr_p_mm = env.current_probe_tip_pos.copy() * 1000.0
        probe_xyz.append(curr_p_mm)
        y_visited_mm.append(curr_p_mm[1])

        if hasattr(env, "all_reconstructed_voxels_world"):
            new_voxels = [v for v in env.all_reconstructed_voxels_world if v not in seen_voxels]
            if len(new_voxels) > 0:
                for v in new_voxels:
                    seen_voxels.add(v)
                new_voxels_mm = np.array(new_voxels, dtype=np.float64) * 1.5
                idx_lists = tree_upper.query_ball_point(new_voxels_mm, r=1.5)
                for indices in idx_lists:
                    if len(indices) > 0:
                        covered_upper_mask[indices] = True

        cov_pct = float(np.mean(covered_upper_mask) * 100.0)
        coverage_curve.append(cov_pct)

        if term or trunc:
            break

    if hasattr(env, "all_reconstructed_voxels_world") and len(env.all_reconstructed_voxels_world) > 0:
        discovered_voxels_mm = np.array(list(env.all_reconstructed_voxels_world), dtype=np.float64) * 1.5
    else:
        discovered_voxels_mm = np.empty((0, 3), dtype=np.float64)

    if len(discovered_voxels_mm) > 0:
        dists_to_ref, _ = tree_ref.query(discovered_voxels_mm)
        mean_dist_to_ref = float(np.mean(dists_to_ref))
        dists_all_ref, _ = cKDTree(discovered_voxels_mm).query(ref_voxels_mm)
        final_whole_cov = float(np.mean(dists_all_ref <= 1.5) * 100.0)
    else:
        mean_dist_to_ref = 0.0
        final_whole_cov = 0.0

    probe_xyz = np.array(probe_xyz)
    path_len_m = float(np.sum(np.linalg.norm(np.diff(probe_xyz, axis=0), axis=1)) / 1000.0)

    # Footprint fraction reached
    y_span_organ = b[3] - b[2]
    y_reach_span = max(y_visited_mm) - min(y_visited_mm)
    footprint_fraction = float(min(1.0, y_reach_span / max(1.0, y_span_organ)) * 100.0)

    env.close()

    return {
        "target": target,
        "patient": patient,
        "method": method,
        "seed": seed,
        "discovered_voxels": discovered_voxels_mm,
        "probe_xyz": probe_xyz,
        "coverage_curve": np.array(coverage_curve),
        "ref_voxels": ref_voxels_mm,
        "upper_surface_voxels": upper_surface_voxels_mm,
        "mean_dist_to_ref": mean_dist_to_ref,
        "final_upper_cov": coverage_curve[-1],
        "final_whole_cov": final_whole_cov,
        "path_len_m": path_len_m,
        "footprint_fraction": footprint_fraction,
        "mesh_bounds": b
    }


def render_panel(pv_mesh, discovered_voxels, probe_xyz, camera_mode="top", resolution=(1600, 1200)):
    """
    Renders single panel:
    - GT organ: light grey layer (opacity 0.40) showing visible ceiling
    - Discovered voxels: small spheres on top
    - Probe path: thin line (green start, red end)
    - Tight bounding box framing with 10% margin
    """
    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background("white")

    # 1. Ground Truth Visible Surface Mesh
    pl.add_mesh(pv_mesh, color="#c8d1d3", opacity=0.40, smooth_shading=True)

    # 2. Discovered points
    if len(discovered_voxels) > 0:
        p_cloud = pv.PolyData(discovered_voxels)
        p_cloud["height"] = discovered_voxels[:, 2]
        pl.add_mesh(p_cloud, scalars="height", cmap="viridis", point_size=7, render_points_as_spheres=True, show_scalar_bar=False)

    # 3. Probe path
    if len(probe_xyz) > 1:
        line = pv.lines_from_points(probe_xyz)
        pl.add_mesh(line, color="#1a252f", line_width=2.5)
        start_pt = pv.PolyData(probe_xyz[0:1])
        end_pt = pv.PolyData(probe_xyz[-1:])
        pl.add_mesh(start_pt, color="#27ae60", point_size=18, render_points_as_spheres=True)
        pl.add_mesh(end_pt, color="#e74c3c", point_size=18, render_points_as_spheres=True)

    center = pv_mesh.center
    bounds = pv_mesh.bounds
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])

    if camera_mode == "top":
        cam_pos = [center[0], center[1], center[2] + max_dim * 1.8]
        view_up = [0, 1, 0]
    else:
        cam_pos = [center[0] + max_dim * 2.2, center[1], center[2]]
        view_up = [0, 0, 1]

    pl.camera_position = [cam_pos, center, view_up]
    pl.reset_camera()
    pl.camera.zoom(0.90)

    img = pl.screenshot()
    pl.close()
    return img


def main():
    print("=" * 80)
    print("  SONOGYM WHOLE-ORGAN RECONSTRUCTION BENCHMARK & EVALUATION")
    print("=" * 80)

    # Step 1: Rebuild Spine to pure L1-L5
    spine_audit = verify_and_rebuild_spine_labels()

    # Step 2: Statistical Run (10 episodes per patient x 5 patients x 4 methods x 3 targets)
    # Total combinations = 3 targets x 5 patients x 4 methods x 10 seeds = 600 episodes
    cache_jsonl = OUTPUT_DIR / "statistical_evaluation_raw.jsonl"
    cache_json = OUTPUT_DIR / "statistical_evaluation_raw.json"
    raw_results = []
    if cache_jsonl.exists():
        print(f"[CACHE] Loading cached statistical evaluations from {cache_jsonl}")
        with open(cache_jsonl) as f:
            for line in f:
                if line.strip():
                    try:
                        raw_results.append(json.loads(line))
                    except Exception:
                        pass
    elif cache_json.exists():
        print(f"[CACHE] Loading cached statistical evaluations from {cache_json}")
        with open(cache_json) as f:
            raw_results = json.load(f)

    existing_keys = {(r["target"], r["patient"], r["method"], r["seed"]) for r in raw_results}
    tasks_to_run = []

    for target in TARGETS:
        for patient in ALL_PATIENTS:
            for method in METHODS:
                for seed_idx in range(10):
                    seed = 42 + seed_idx * 17  # 10 diverse random start seeds
                    if (target, patient, method, seed) not in existing_keys:
                        tasks_to_run.append((target, patient, method, seed, 300))

    if len(tasks_to_run) > 0:
        print(f"[EVALUATION] Running {len(tasks_to_run)} episodes across multiprocessing pool (12 CPU workers)...", flush=True)
        t0 = time.time()
        with mp.Pool(processes=12) as pool:
            for idx, res in enumerate(pool.imap_unordered(run_single_episode_worker, tasks_to_run)):
                # Save essential metrics to raw_results
                rec = {
                    "target": res["target"],
                    "patient": res["patient"],
                    "method": res["method"],
                    "seed": res["seed"],
                    "final_upper_cov": float(res["final_upper_cov"]),
                    "final_whole_cov": float(res["final_whole_cov"]),
                    "path_len_m": float(res["path_len_m"]),
                    "mean_dist_to_ref": float(res["mean_dist_to_ref"]),
                    "footprint_fraction": float(res["footprint_fraction"]),
                    "n_discovered_voxels": int(len(res["discovered_voxels"]))
                }
                raw_results.append(rec)

                # Incremental append to JSONL so progress is never lost
                with open(cache_jsonl, "a") as f_out:
                    f_out.write(json.dumps(rec) + "\n")
                    f_out.flush()

                # Save seed 42 (seed_idx 0) NPZ for canonical figure rendering
                if res["seed"] == 42:
                    npz_name = f"{res['target']}_{res['patient']}_{res['method']}.npz"
                    np.savez_compressed(
                        OUTPUT_DIR / npz_name,
                        discovered_voxels=res["discovered_voxels"],
                        probe_xyz=res["probe_xyz"],
                        coverage_curve=res["coverage_curve"],
                        ref_voxels=res["ref_voxels"],
                        upper_surface_voxels=res["upper_surface_voxels"]
                    )

                print(f"  --> Progress: [{idx + 1}/{len(tasks_to_run)}] episodes completed ({time.time() - t0:.1f}s) | {res['target']} {res['patient']} {res['method']} ({res['final_upper_cov']:.1f}%)", flush=True)

        with open(cache_json, "w") as f:
            json.dump(raw_results, f, indent=2)
        print(f"[SAVED] Saved all statistical results to {cache_json}")

    # Step 3: Compute Summary Statistics split by Train vs Held-Out
    summary_stats = {}
    for target in TARGETS:
        summary_stats[target] = {}
        for method in METHODS:
            summary_stats[target][method] = {
                "train": {"upper_cov": [], "whole_cov": [], "path_len": [], "dist": [], "footprint": []},
                "held_out": {"upper_cov": [], "whole_cov": [], "path_len": [], "dist": [], "footprint": []}
            }

    for r in raw_results:
        tgt = r["target"]
        mth = r["method"]
        split = "train" if r["patient"] in TRAIN_PATIENTS else "held_out"
        summary_stats[tgt][mth][split]["upper_cov"].append(r["final_upper_cov"])
        summary_stats[tgt][mth][split]["whole_cov"].append(r["final_whole_cov"])
        summary_stats[tgt][mth][split]["path_len"].append(r["path_len_m"])
        summary_stats[tgt][mth][split]["dist"].append(r["mean_dist_to_ref"])
        summary_stats[tgt][mth][split]["footprint"].append(r["footprint_fraction"])

    # Build Markdown & JSON tables
    md_lines = [
        "# Statistical Evaluation Summary (10 Episodes / Patient, 300-step Checkpoints)",
        "",
        "### Training Patients (s0011, s0058, s0250) vs Held-Out Patients (s0344, s0358)",
        "",
        "| Target Anatomy | Method | Split | Upper-Surface Coverage (%) | Whole-Organ Coverage (%) | Path Length (m) | Footprint Reached (%) | Mean Dist to Mesh (mm) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    table_data = []

    for target in TARGETS:
        for method in METHODS:
            for split in ["train", "held_out"]:
                vals = summary_stats[target][method][split]
                u_m, u_s = np.mean(vals["upper_cov"]), np.std(vals["upper_cov"])
                w_m, w_s = np.mean(vals["whole_cov"]), np.std(vals["whole_cov"])
                p_m, p_s = np.mean(vals["path_len"]), np.std(vals["path_len"])
                f_m, f_s = np.mean(vals["footprint"]), np.std(vals["footprint"])
                d_m, d_s = np.mean(vals["dist"]), np.std(vals["dist"])

                split_label = "Train (3 Pts)" if split == "train" else "Held-Out (2 Pts)"
                md_lines.append(
                    f"| **{target.upper()}** | {METHOD_NAMES[method]} | {split_label} | "
                    f"**{u_m:.1f} ± {u_s:.1f}%** | {w_m:.1f} ± {w_s:.1f}% | {p_m:.2f} ± {p_s:.2f} m | "
                    f"{f_m:.1f} ± {f_s:.1f}% | {d_m:.2f} ± {d_s:.2f} mm |"
                )

                table_data.append({
                    "target": target,
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "split": split,
                    "upper_coverage_mean": float(u_m),
                    "upper_coverage_std": float(u_s),
                    "whole_coverage_mean": float(w_m),
                    "whole_coverage_std": float(w_s),
                    "path_length_mean": float(p_m),
                    "path_length_std": float(p_s),
                    "footprint_reached_mean": float(f_m),
                    "footprint_reached_std": float(f_s),
                    "distance_to_mesh_mean": float(d_m),
                    "distance_to_mesh_std": float(d_s)
                })

    md_path = OUTPUT_DIR / "reconstruction_statistics.md"
    json_path = OUTPUT_DIR / "reconstruction_statistics.json"

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(json_path, "w") as f:
        json.dump(table_data, f, indent=2)

    print(f"[SAVED] Markdown table to {md_path}")
    print(f"[SAVED] JSON table to {json_path}")

    # Step 4: Render High-Quality 3-Panel Publication Figures
    print("\n" + "=" * 80)
    print("  4. RENDERING 3-PANEL PUBLICATION FIGURES (LARGE TEXT, NO COLORBAR)")
    print("=" * 80)

    # For held-out patient s0344
    patient_fig = "s0344"
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient_fig)

    for target in TARGETS:
        env = RoboticUltrasoundGymEnv(
            subject_dir=subj_dir, task="reconstruction", anatomy=target,
            sim_mode="conv", size=128, pixel_spacing=1.0, max_episode_steps=1, roi_radius=float("inf")
        )
        env.reset(seed=42)
        pv_mesh, _, _ = extract_organ_mesh_and_ref_voxels(env, target)
        env.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), dpi=300)
        plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.06)

        display_methods = ["ppo_memory", "ppo_nomem", "raster_footprint"]

        for col_idx, method in enumerate(display_methods):
            npz_path = OUTPUT_DIR / f"{target}_{patient_fig}_{method}.npz"
            data = np.load(npz_path)
            disc = data["discovered_voxels"]
            pxyz = data["probe_xyz"]
            cov_curve = data["coverage_curve"]
            final_cov = cov_curve[-1]

            img = render_panel(pv_mesh, disc, pxyz, camera_mode="top")
            ax = axes[col_idx]
            ax.imshow(img)
            ax.set_axis_off()

            # Prominent, large text label with no colorbar
            title_text = f"{METHOD_NAMES[method]}\nUpper Surface Coverage: {final_cov:.1f}%"
            ax.set_title(title_text, fontsize=16, fontweight="bold", pad=12, color="#1a252f")

        fig_png = OUTPUT_DIR / f"fig_pub_recon_{target}_{patient_fig}.png"
        fig_pdf = OUTPUT_DIR / f"fig_pub_recon_{target}_{patient_fig}.pdf"
        fig.savefig(str(fig_png), dpi=300, bbox_inches="tight")
        fig.savefig(str(fig_pdf), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[RENDERED PUBLICATION FIGURE] {fig_png.name} & {fig_pdf.name}")

    print("\n[COMPLETE] All deliverables generated successfully!")


if __name__ == "__main__":
    main()
