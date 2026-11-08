#!/usr/bin/env python3
"""
scripts/step2_recon_eval.py
===========================
Whole-Organ 3D Reconstruction Statistical Benchmark & Visualizer:
- Evaluates Spine (L1-L5 + sacrum), Aorta (aorta + common iliacs + IVC), and Liver (parenchyma).
- Evaluates 4 methods: PPO+Memory, PPO (No Memory), Footprint Raster, Short Raster.
- 10 random-start episodes per patient across 5 patients (3 train: s0011, s0058, s0250; 2 held-out: s0344, s0358).
- 300 steps per episode.
- Metrics (Mean ± Std, split Train / Held-Out):
  1. Upper-surface coverage (%)
  2. Whole-organ coverage (%)
  3. Path length (m)
  4. Total rotation (sum of |yaw change| over episode, deg)
  5. Footprint fraction reached (%) for footprint raster.
- Outputs:
  * results/reconstruction_benchmark_table.md
  * results/reconstruction_benchmark_table.json
  * results/recon_vis/fig_recon_{target}_s0344.png & .pdf (cropped, large text, no colorbar)
  * results/recon_vis/fig_coverage_curve_{target}_s0344.png & .pdf
  * results/recon_vis/README.md
  * results/status.json update
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
TABLE_DIR = PROJECT_DIR / "results"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

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


def extract_organ_mesh_and_ref_voxels(env, anatomy):
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

    # 3. Upper-surface subset: top-most voxel per (X, Y) column at 1.5mm resolution
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
    def __init__(self, x_min_mm, x_max_mm, y_min_mm, y_max_mm, y_pitch_mm=16.0):
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
    import pybullet as p
    prev_yaw = float(p.getEulerFromQuaternion(env.current_probe_quat.tolist())[2])
    total_rotation_deg = 0.0

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

        # Track probe yaw change
        curr_yaw = float(p.getEulerFromQuaternion(env.current_probe_quat.tolist())[2])
        d_yaw = abs((curr_yaw - prev_yaw + np.pi) % (2.0 * np.pi) - np.pi)
        total_rotation_deg += float(np.degrees(d_yaw))
        prev_yaw = curr_yaw

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
        "discovered_voxels": discovered_voxels_mm.tolist(),
        "probe_xyz": probe_xyz.tolist(),
        "coverage_curve": coverage_curve,
        "ref_voxels": ref_voxels_mm.tolist(),
        "upper_surface_voxels": upper_surface_voxels_mm.tolist(),
        "mean_dist_to_ref": mean_dist_to_ref,
        "final_upper_cov": coverage_curve[-1],
        "final_whole_cov": final_whole_cov,
        "path_len_m": path_len_m,
        "total_rotation_deg": total_rotation_deg,
        "footprint_fraction": footprint_fraction,
        "mesh_bounds": list(b)
    }


def render_panel(pv_mesh, discovered_voxels, probe_xyz, camera_mode="top", resolution=(1600, 1200)):
    pl = pv.Plotter(off_screen=True, window_size=list(resolution))
    pl.set_background("white")

    # 1. Ground Truth Visible Surface Mesh (light grey layer)
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


def write_status(state="RUNNING", current_run="reconstruction_evaluation", error=None):
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


def write_readme():
    readme_path = OUTPUT_DIR / "README.md"
    content = r"""# 3D Ultrasound Anatomy Reconstruction Benchmark

## Upper-Surface Coverage Definition
Upper-surface coverage measures the fraction of an organ's transducer-visible dorsal/anterior surface successfully mapped by the ultrasound probe within the episode step budget:
1. Ground-truth surface voxels are extracted from the segmentation volume via Marching Cubes and 3D boundary edge detection.
2. The visible surface subset is formed by grouping voxels into vertical columns of grid resolution 1.5 mm and selecting the single top-most voxel per column along the vertical scanning axis.
3. An upper-surface voxel is declared **covered** if at least one reconstructed ultrasound surface voxel falls within a Euclidean ball of radius **r = 1.5 mm** ($r \le 1.5\text{ mm}$).
4. Upper-surface coverage is reported as $\\frac{N_{\\text{covered}}}{N_{\\text{upper}}} \\times 100\\%$.

## Whole-Organ Coverage Definition
Whole-organ coverage evaluates map completeness over all 3D surface boundary voxels of the organ:
$$\\text{Coverage}_{\\text{whole}} = \\frac{\\sum_{i} \\mathbb{I}(\\min_j \\|v_{\\text{gt}, i} - v_{\\text{recon}, j}\\| \\le 1.5\\text{ mm})}{N_{\\text{total}}} \\times 100\\%$$

## Footprint Raster Policy Definition
The organ-aligned footprint raster sweeps the projected 2D footprint of the anatomical target on the patient's skin:
1. The bounding box of the organ in world coordinates $[X_{\\min}, X_{\\max}] \\times [Y_{\\min}, Y_{\\max}]$ defines the lateral and cranial-caudal limits with a 4 mm internal margin.
2. Lateral sweep speed $\\Delta x$ traverses the width of the organ until the skin-plane boundary is reached.
3. Cranial-caudal step advance: at each turning point, the probe advances along $Y$ by a fixed pitch of **16.0 mm** (matched to transducer aperture and 300-step horizon).
4. **Footprint fraction reached** is defined as the span of $Y$ coordinates traversed divided by the total organ longitudinal length:
   $$\\text{Footprint Fraction} = \\min\\left(1.0, \\frac{Y_{\\text{visited, max}} - Y_{\\text{visited, min}}}{Y_{\\text{organ, max}} - Y_{\\text{organ, min}}}\\right) \\times 100\\%$$

## Target Anatomy Specifications
- **Spine**: Lumbar vertebrae L1-L5 plus Sacrum (`spine_full.nii.gz` with sacrum preserved).
- **Aorta**: Full abdominal vascular tree including abdominal aorta, bilateral common iliac arteries, and inferior vena cava (`aorta_full.nii.gz`).
- **Liver**: Liver parenchyma (`liver_full.nii.gz`).
"""
    with open(readme_path, "w") as f:
        f.write(content)
    print(f"[README] Documentation written to {readme_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run tiny dry run (2 episodes) to verify script runs")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel CPU worker processes")
    parser.add_argument("--steps", type=int, default=300, help="Episode horizon")
    args = parser.parse_args()

    write_status(state="RUNNING", current_run="reconstruction_evaluation")
    write_readme()

    cache_jsonl = OUTPUT_DIR / "statistical_evaluation_raw.jsonl"
    existing_records = []
    if cache_jsonl.exists():
        with open(cache_jsonl) as f:
            for l in f:
                if l.strip():
                    try:
                        existing_records.append(json.loads(l))
                    except Exception:
                        pass

    if args.dry_run:
        print("\n>>> PERFORMING TINY DRY RUN (2 EPISODES) <<<")
        dry_tasks = [
            ("spine", "s0344", "ppo_memory", 42, 50),
            ("spine", "s0011", "raster_footprint", 42, 50)
        ]
        for task in dry_tasks:
            t0 = time.time()
            res = run_single_episode_worker(task)
            el = time.time() - t0
            print(f"  [DRY-RUN] {res['target']} | {res['patient']} | {res['method']} -> UpperCov: {res['final_upper_cov']:.1f}% | WholeCov: {res['final_whole_cov']:.1f}% | Path: {res['path_len_m']:.2f}m | Rot: {res['total_rotation_deg']:.1f}° | Footprint: {res['footprint_fraction']:.1f}% ({el:.1f}s)")
        print("\n[+] Dry run completed successfully! Exiting without launching full sweep.")
        write_status(state="DRY_RUN_PASSED", current_run="reconstruction_evaluation")
        return

    # Full execution
    # Deduplicate records by keeping the latest record per (target, patient, method, seed)
    dedup_dict = {}
    for r in existing_records:
        k = (r.get("target"), r.get("patient"), r.get("method"), r.get("seed"))
        dedup_dict[k] = r
    all_results = list(dedup_dict.values())
    existing_keys = set(dedup_dict.keys())

    tasks = []
    for target in TARGETS:
        for patient in ALL_PATIENTS:
            for method in METHODS:
                for seed_idx in range(10):
                    seed = 42 + seed_idx * 17
                    if (target, patient, method, seed) not in existing_keys:
                        tasks.append((target, patient, method, seed, args.steps))

    print(f"\n[SWEEP] Running {len(tasks)} tasks using {args.workers} CPU workers...")

    if len(tasks) > 0:
        with mp.Pool(processes=args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_single_episode_worker, tasks), 1):
                all_results.append(res)
                with open(cache_jsonl, "a") as f:
                    f.write(json.dumps(res) + "\n")
                if i % 10 == 0 or i == len(tasks):
                    print(f"  Progress: [{i}/{len(tasks)}] finished | {res['target']} {res['patient']} {res['method']} ({res['final_upper_cov']:.1f}%)")

    # Aggregate Metrics into Markdown and JSON
    table_dict = {}
    md_lines = [
        "# SonoGym Whole-Organ 3D Reconstruction Benchmark Table",
        "",
        "| Target | Cohort | Method | Upper Coverage (%) | Whole Coverage (%) | Path Length (m) | Total Rotation (deg) | Footprint Reach (%) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for target in TARGETS:
        table_dict[target] = {}
        for cohort, p_list in [("Train", TRAIN_PATIENTS), ("Held-Out", HELD_OUT_PATIENTS)]:
            table_dict[target][cohort] = {}
            for method in METHODS:
                records = [r for r in all_results if r["target"] == target and r["patient"] in p_list and r["method"] == method]
                if len(records) == 0:
                    continue
                u_cov = [r["final_upper_cov"] for r in records]
                w_cov = [r["final_whole_cov"] for r in records]
                p_len = [r["path_len_m"] for r in records]
                r_deg = [r.get("total_rotation_deg", 0.0) for r in records]
                f_frc = [r["footprint_fraction"] for r in records]

                u_str = f"{np.mean(u_cov):.1f} ± {np.std(u_cov):.1f}"
                w_str = f"{np.mean(w_cov):.1f} ± {np.std(w_cov):.1f}"
                p_str = f"{np.mean(p_len):.2f} ± {np.std(p_len):.2f}"
                r_str = f"{np.mean(r_deg):.1f} ± {np.std(r_deg):.1f}"
                f_str = f"{np.mean(f_frc):.1f} ± {np.std(f_frc):.1f}" if method == "raster_footprint" else "-"

                table_dict[target][cohort][method] = {
                    "upper_coverage_mean": float(np.mean(u_cov)),
                    "upper_coverage_std": float(np.std(u_cov)),
                    "whole_coverage_mean": float(np.mean(w_cov)),
                    "whole_coverage_std": float(np.std(w_cov)),
                    "path_length_mean": float(np.mean(p_len)),
                    "path_length_std": float(np.std(p_len)),
                    "total_rotation_mean": float(np.mean(r_deg)),
                    "total_rotation_std": float(np.std(r_deg)),
                    "footprint_reach_mean": float(np.mean(f_frc)),
                    "footprint_reach_std": float(np.std(f_frc))
                }

                m_name = METHOD_NAMES[method]
                md_lines.append(f"| {target.capitalize()} | {cohort} | {m_name} | {u_str} | {w_str} | {p_str} | {r_str} | {f_str} |")

    table_md = "\n".join(md_lines)
    with open(TABLE_DIR / "reconstruction_benchmark_table.md", "w") as f:
        f.write(table_md)
    with open(TABLE_DIR / "reconstruction_benchmark_table.json", "w") as f:
        json.dump(table_dict, f, indent=2)
    print(f"\n[DONE] Tables written to {TABLE_DIR / 'reconstruction_benchmark_table.md'}")

    # Render High-Quality 3-Panel Publication Figures for held-out patient s0344
    print("\n" + "=" * 80)
    print("  RENDERING 3-PANEL RECONSTRUCTION FIGURES & COVERAGE CURVES (s0344)")
    print("=" * 80)
    patient_fig = "s0344"
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient_fig)
    display_methods = ["ppo_memory", "ppo_nomem", "raster_footprint"]

    for target in TARGETS:
        env = RoboticUltrasoundGymEnv(
            subject_dir=subj_dir, task="reconstruction", anatomy=target,
            sim_mode="conv", size=128, pixel_spacing=1.0, max_episode_steps=1, roi_radius=float("inf"),
            device="cpu"
        )
        env.reset(seed=42)
        pv_mesh, _, _ = extract_organ_mesh_and_ref_voxels(env, target)
        env.close()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), dpi=300)
        plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99, top=0.88, bottom=0.06)

        curves_dict = {}

        for col_idx, method in enumerate(display_methods):
            # Load representative episode for s0344 from npz or all_results
            npz_file = OUTPUT_DIR / f"{target}_{patient_fig}_{method}.npz"
            if npz_file.exists():
                npz_data = np.load(npz_file)
                disc = npz_data["discovered_voxels"]
                pxyz = npz_data["probe_xyz"]
                cov_curve = npz_data["coverage_curve"]
            else:
                ep_records = [r for r in all_results if r["target"] == target and r["patient"] == patient_fig and r["method"] == method and "discovered_voxels" in r]
                if len(ep_records) > 0:
                    rec = ep_records[0]
                    disc = np.array(rec["discovered_voxels"])
                    pxyz = np.array(rec["probe_xyz"])
                    cov_curve = np.array(rec["coverage_curve"])
                else:
                    rec = run_single_episode_worker((target, patient_fig, method, 42, 300))
                    disc = np.array(rec["discovered_voxels"])
                    pxyz = np.array(rec["probe_xyz"])
                    cov_curve = np.array(rec["coverage_curve"])

            curves_dict[method] = cov_curve
            final_cov = cov_curve[-1]

            img = render_panel(pv_mesh, disc, pxyz, camera_mode="top")
            ax = axes[col_idx]
            ax.imshow(img)
            ax.set_axis_off()

            title_text = f"{METHOD_NAMES[method]}\nUpper Surface Coverage: {final_cov:.1f}%"
            ax.set_title(title_text, fontsize=16, fontweight="bold", pad=12, color="#1a252f")

        fig_png = OUTPUT_DIR / f"fig_recon_{target}_{patient_fig}.png"
        fig_pdf = OUTPUT_DIR / f"fig_recon_{target}_{patient_fig}.pdf"
        fig.savefig(str(fig_png), dpi=300, bbox_inches="tight")
        fig.savefig(str(fig_pdf), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[FIGURE] Rendered 3D panel: {fig_png.name} & {fig_pdf.name}")

        # Coverage curve vs step plot
        fig_c, ax_c = plt.subplots(figsize=(8, 5.5), dpi=300)
        colors = {"ppo_memory": "#27ae60", "ppo_nomem": "#2980b9", "raster_footprint": "#e67e22"}
        for method in display_methods:
            if method in curves_dict:
                curve = curves_dict[method]
                ax_c.plot(curve, label=METHOD_NAMES[method], color=colors[method], linewidth=2.5)

        ax_c.set_title(f"{target.capitalize()} Coverage vs Step (Patient {patient_fig})", fontsize=14, fontweight="bold", pad=10)
        ax_c.set_xlabel("Episode Step", fontsize=12)
        ax_c.set_ylabel("Upper-Surface Coverage (%)", fontsize=12)
        ax_c.set_ylim(-2, 102)
        ax_c.grid(True, linestyle="--", alpha=0.5)
        ax_c.legend(fontsize=11, loc="lower right")

        curve_png = OUTPUT_DIR / f"fig_coverage_curve_{target}_{patient_fig}.png"
        curve_pdf = OUTPUT_DIR / f"fig_coverage_curve_{target}_{patient_fig}.pdf"
        fig_c.savefig(str(curve_png), dpi=300, bbox_inches="tight")
        fig_c.savefig(str(curve_pdf), dpi=300, bbox_inches="tight")
        plt.close(fig_c)
        print(f"[CURVE] Rendered coverage curve: {curve_png.name} & {curve_pdf.name}")

    write_status(state="COMPLETED", current_run="reconstruction_evaluation")


if __name__ == "__main__":
    main()
