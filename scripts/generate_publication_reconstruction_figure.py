#!/usr/bin/env python3
"""
generate_master_reconstruction_figures.py
=========================================
Generates the publication-grade FULL-ORGAN Reconstruction Benchmark Figures:
- Uses the NEW master models (*_whole_s0) trained overnight on whole-organ geometry.
- Extracts the FULL CT Ground Truth surface (from spine_full.nii.gz, aorta_full.nii.gz, liver_full.nii.gz)
  WITHOUT any artificial ROI spherical clipping.
- Evaluates:
    Columns: Heuristic Raster, PPO (No Memory), PPO + Memory (Our Method)
    Rows: Full Lumbar Spine (L1-L5), Full Abdominal Aorta, Full Whole Liver
- Outputs:
    1. figures/figure_reconstruction_3d_overlay.pdf & .png (3x3 Full-Organ 3D Overlay)
    2. figures/figure_reconstruction_ct_slice_overlay.pdf & .png (CT Volume Slice Registration)
    3. figures/full_organ_ct_vs_ultrasound_comparison.png (Direct Full CT vs Ultrasound 3D Shape)
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion
import nibabel as nib

PROJECT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import torch
from stable_baselines3 import PPO
from robotic_us_env import RoboticUltrasoundGymEnv
from train_reconstruction_mb_occupancy import OccupancyConditionedExtractor, GoalConditionedExtractor

FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = PROJECT_DIR / "results" / "figure_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Master New Models (Whole Organ)
CHECKPOINTS = {
    "spine": {
        "PPO (No Mem)": PROJECT_DIR / "ppo_checkpoints/block_b/spine_ppo_nomem_whole_s0/final_model.zip",
        "PPO + Memory": PROJECT_DIR / "ppo_checkpoints/block_b/spine_ppo_memory_whole_s0/final_model.zip",
    },
    "aorta": {
        "PPO (No Mem)": PROJECT_DIR / "ppo_checkpoints/block_b/aorta_ppo_nomem_whole_s0/final_model.zip",
        "PPO + Memory": PROJECT_DIR / "ppo_checkpoints/block_b/aorta_ppo_memory_whole_s0/final_model.zip",
    },
    "liver": {
        "PPO (No Mem)": PROJECT_DIR / "ppo_checkpoints/block_b/liver_ppo_nomem_whole_s0/final_model.zip",
        "PPO + Memory": PROJECT_DIR / "ppo_checkpoints/block_b/liver_ppo_memory_whole_s0/final_model.zip",
    }
}

ORGANS = ["spine", "aorta", "liver"]
METHODS = ["Heuristic Raster", "PPO (No Mem)", "PPO + Memory"]


def extract_full_organ_gt(env, anatomy):
    """Extract complete, unclipped 3D surface points of the full organ from CT label volume."""
    lbl = env.label_volume.cpu().numpy() if hasattr(env.label_volume, "cpu") else np.asarray(env.label_volume)
    edge = (lbl > 0.5) & ~binary_erosion(lbl > 0.5)
    edge_vox = np.argwhere(edge).astype(np.float64)

    affine = env.reg_meta['affine']
    offset = env.reg_meta['mesh_centering_offset']
    scale = env.mesh_scale
    R_body = env.reg_body_orientation_matrix
    body_pos = env.reg_body_position

    vox_hom = np.hstack([edge_vox, np.ones((len(edge_vox), 1), dtype=np.float64)])
    p_mm = (affine @ vox_hom.T).T[:, :3]
    gt_all = (R_body @ ((p_mm / 1000.0 - offset) * scale).T).T + body_pos

    # Compute visible probe-facing ceiling mask
    res_c = 0.002
    xy_keys = np.round(gt_all[:, :2] / res_c).astype(int)
    col_dict = {}
    for idx, (kx, ky) in enumerate(xy_keys):
        z_val = gt_all[idx, 2]
        if (kx, ky) not in col_dict or z_val > col_dict[(kx, ky)][0]:
            col_dict[(kx, ky)] = (z_val, idx)
    ceiling_indices = set(val[1] for val in col_dict.values())
    visible_mask = np.array([i in ceiling_indices for i in range(len(gt_all))], dtype=bool)

    return gt_all, visible_mask


def run_full_organ_episode(anatomy, method, pid="s0011", ep_seed=42, max_steps=150):
    """Run evaluation on whole-organ geometry with early_termination=False."""
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / pid)
    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="reconstruction",
        anatomy=anatomy,
        sim_mode="conv",
        size=128,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=False,
        roi_radius=float("inf")  # FULL WHOLE ORGAN
    )

    obs, _ = env.reset(seed=ep_seed)
    gt_all, visible_mask = extract_full_organ_gt(env, anatomy)

    model = None
    if method in ("PPO (No Mem)", "PPO + Memory"):
        ckpt = CHECKPOINTS[anatomy][method]
        if ckpt.exists():
            model = PPO.load(str(ckpt), device="cpu")
        else:
            print(f"[WARN] Checkpoint missing: {ckpt}")

    probe_path = [env.current_probe_tip_pos.copy()]
    total_len = 0.0
    prev_tip = env.current_probe_tip_pos.copy()

    for step in range(max_steps):
        if method == "Heuristic Raster":
            # Heuristic raster sweeping scanning pattern across skin
            dx = 1.0 if (step // 20) % 2 == 0 else -1.0
            dy = 0.6 if step % 20 == 0 else 0.0
            action = np.array([dx, dy, 0.0], dtype=np.float32)
        else:
            if model is not None:
                m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
                action, _ = model.predict(m_obs, deterministic=True)
            else:
                action = env.action_space.sample()

        obs, r, term, trunc, info = env.step(action)
        curr_tip = env.current_probe_tip_pos.copy()
        probe_path.append(curr_tip)
        total_len += float(np.linalg.norm(curr_tip - prev_tip))
        prev_tip = curr_tip.copy()
        if term or trunc:
            break

    # Gather full reconstructed points in world coordinates
    if hasattr(env, "all_reconstructed_voxels_world") and len(env.all_reconstructed_voxels_world) > 0:
        pts_all = np.array([list(k) for k in env.all_reconstructed_voxels_world], dtype=np.float64) * 0.0015
    elif hasattr(env, "reconstructed_voxels_world") and len(env.reconstructed_voxels_world) > 0:
        pts_all = np.array([list(k) for k in env.reconstructed_voxels_world], dtype=np.float64) * 0.0015
    else:
        pts_all = np.empty((0, 3))

    # Evaluate against Ground Truth CT Surface
    if len(gt_all) > 0 and len(pts_all) > 0:
        tree_recon = cKDTree(pts_all)
        dists_gt_to_recon, _ = tree_recon.query(gt_all)
        covered_mask = (dists_gt_to_recon <= 0.0015)
        cov_pct = float(np.mean(covered_mask) * 100.0)
        vis_cov_pct = float(np.mean(dists_gt_to_recon[visible_mask] <= 0.0015) * 100.0) if np.any(visible_mask) else 0.0

        # Mean distance to nearest GT point (Point accuracy)
        tree_gt = cKDTree(gt_all)
        dists_recon_to_gt, _ = tree_gt.query(pts_all)
        acc_mm = float(np.mean(dists_recon_to_gt) * 1000.0)
    else:
        covered_mask = np.zeros(len(gt_all), dtype=bool)
        cov_pct = 0.0
        vis_cov_pct = 0.0
        acc_mm = float("nan")

    print(f"[{anatomy.upper()} - {method}] Full Cov: {cov_pct:.2f}% | Visible Cov: {vis_cov_pct:.2f}% | Acc: {acc_mm:.2f}mm | Path: {total_len:.2f}m | Pts: {len(pts_all):,}")

    t_internal = env.target_internal_world_pos.copy()
    env.close()

    return {
        "gt_all": gt_all,
        "visible_mask": visible_mask,
        "pts_all": pts_all,
        "probe_path": np.array(probe_path),
        "covered_mask": covered_mask,
        "cov_pct": cov_pct,
        "vis_cov_pct": vis_cov_pct,
        "acc_mm": acc_mm,
        "path_len": total_len,
        "t_internal": t_internal
    }


def main():
    print("=" * 80)
    print("  GENERATING MASTER FULL-ORGAN 3D RECONSTRUCTION FIGURES (NEW WHOLE MODELS)")
    print("=" * 80)

    # 1. Run / extract data for all 3 organs x 3 methods
    full_data = {}
    for organ in ORGANS:
        full_data[organ] = {}
        for method in METHODS:
            print(f"--> Extracting: {organ.upper()} | {method} ...")
            full_data[organ][method] = run_full_organ_episode(organ, method, pid="s0011", ep_seed=42)

    # Cache data
    with open(DATA_DIR / "master_full_organ_reconstruction_data.pkl", "wb") as f:
        pickle.dump(full_data, f)
    print(f"[OK] Cached data to {DATA_DIR / 'master_full_organ_reconstruction_data.pkl'}")

    # Load patient CT volume for s0011 for slice registration
    ct_img = nib.load(str(PROJECT_DIR / "totalseg_patients/s0011/ct.nii.gz"))
    ct_vol = ct_img.get_fdata()
    with open(PROJECT_DIR / "totalseg_patients/s0011/registration_meta.json") as f:
        reg_meta = json.load(f)

    affine = np.array(reg_meta["affine"], dtype=np.float64)
    inv_affine = np.array(reg_meta["inv_affine"], dtype=np.float64)
    offset = np.array(reg_meta["mesh_centering_offset"], dtype=np.float64)
    body_pos = np.array([0.0, 0.0, 0.876437], dtype=np.float64)
    R_body = np.eye(3, dtype=np.float64)
    ct_shape = ct_vol.shape

    def world_to_voxel(p_world):
        p_w = np.asarray(p_world, dtype=np.float64)
        p_centered = R_body.T @ (p_w - body_pos)
        p_meters = p_centered + offset
        p_mm = p_meters * 1000.0
        p_hom = np.array([p_mm[0], p_mm[1], p_mm[2], 1.0], dtype=np.float64)
        vox_hom = inv_affine @ p_hom
        return vox_hom[:3]

    view_angles = {
        "spine": (25, -65),
        "aorta": (20, -70),
        "liver": (20, -55)
    }

    # ==============================================================================
    # FIGURE 1: 3D RECONSTRUCTION OVERLAY (3 rows x 3 columns)
    # Full Ground Truth CT Organ Shape vs Ultrasound Points vs Probe Path
    # ==============================================================================
    plt.style.use('default')
    fig = plt.figure(figsize=(18, 17), dpi=300)

    for row_idx, organ in enumerate(ORGANS):
        elev, azim = view_angles[organ]
        dat_ref = full_data[organ]["Heuristic Raster"]
        gt_mm = dat_ref["gt_all"] * 1000.0
        
        # Bounding box spanning the entire organ
        margin = 25.0  # mm
        xlim = (np.min(gt_mm[:, 0]) - margin, np.max(gt_mm[:, 0]) + margin)
        ylim = (np.min(gt_mm[:, 1]) - margin, np.max(gt_mm[:, 1]) + margin)
        zlim = (np.min(gt_mm[:, 2]) - margin, np.max(gt_mm[:, 2]) + margin)

        for col_idx, method in enumerate(METHODS):
            ax_idx = row_idx * 3 + col_idx + 1
            ax = fig.add_subplot(3, 3, ax_idx, projection='3d')

            dat = full_data[organ][method]
            gt_mm_inst = dat["gt_all"] * 1000.0
            cov_mask = dat["covered_mask"]
            pts_mm = dat["pts_all"] * 1000.0
            probe_mm = dat["probe_path"] * 1000.0

            # Subsample full CT surface mesh for crisp rendering
            n_gt = len(gt_mm_inst)
            sub_gt = np.random.choice(n_gt, min(n_gt, 3500), replace=False) if n_gt > 3500 else slice(None)

            # Uncovered/Shadowed CT surface in soft clinical grey
            uncovered = ~cov_mask[sub_gt]
            if np.any(uncovered):
                ax.scatter(
                    gt_mm_inst[sub_gt][uncovered, 0],
                    gt_mm_inst[sub_gt][uncovered, 1],
                    gt_mm_inst[sub_gt][uncovered, 2],
                    c='#bdc3c7', s=4, alpha=0.25, edgecolors='none', label='Ground Truth CT Surface'
                )

            # Covered CT Surface (<= 1.5mm) in Emerald Green
            covered = cov_mask[sub_gt]
            if np.any(covered):
                ax.scatter(
                    gt_mm_inst[sub_gt][covered, 0],
                    gt_mm_inst[sub_gt][covered, 1],
                    gt_mm_inst[sub_gt][covered, 2],
                    c='#27ae60', s=10, alpha=0.85, edgecolors='none', label='US Reconstructed (<=1.5mm)'
                )

            # Ultrasound Discovered Points in bright vibrant Cyan
            if len(pts_mm) > 0:
                n_pts = len(pts_mm)
                sub_pts = np.random.choice(n_pts, min(n_pts, 2000), replace=False) if n_pts > 2000 else slice(None)
                ax.scatter(
                    pts_mm[sub_pts, 0], pts_mm[sub_pts, 1], pts_mm[sub_pts, 2],
                    c='#00e5ff', s=8, alpha=0.7, edgecolors='none', label='US Point Cloud'
                )

            # Probe Path on Skin
            if len(probe_mm) > 0:
                ax.plot(probe_mm[:, 0], probe_mm[:, 1], probe_mm[:, 2],
                        color='#e67e22', linestyle='-', linewidth=2.5, alpha=0.95, label='Robot Probe Path')
                ax.scatter(probe_mm[0, 0], probe_mm[0, 1], probe_mm[0, 2],
                           color='#d35400', s=50, marker='o', edgecolors='black')

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X (mm)', fontsize=9)
            ax.set_ylabel('Y (mm)', fontsize=9)
            ax.set_zlabel('Z (mm)', fontsize=9)
            ax.tick_params(labelsize=8)

            acc_str = f"Acc: {dat['acc_mm']:.2f}mm" if not np.isnan(dat['acc_mm']) else ""
            title = f"{method}\nCov: {dat['cov_pct']:.1f}% | {acc_str} | Path: {dat['path_len']:.2f}m"
            if col_idx == 0:
                organ_title = {
                    "spine": "Full Lumbar Spine (L1-L5)",
                    "aorta": "Full Abdominal Aorta",
                    "liver": "Full Whole Liver"
                }[organ]
                title = f"[{organ_title}]\n" + title
            ax.set_title(title, fontsize=11, fontweight='bold', pad=4)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    pdf_3d_path = FIG_DIR / "figure_reconstruction_3d_overlay.pdf"
    png_3d_path = FIG_DIR / "figure_reconstruction_3d_overlay.png"
    fig.savefig(str(pdf_3d_path), dpi=300, bbox_inches='tight')
    fig.savefig(str(png_3d_path), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SUCCESS] Wrote Full 3D Reconstruction Overlay: {png_3d_path} & {pdf_3d_path}")

    # ==============================================================================
    # FIGURE 2: CT SLICE ANATOMICAL OVERLAY (3 rows x 3 columns)
    # Complete CT Cross-section + Full Organ Contour + In-plane Ultrasound Hits
    # ==============================================================================
    fig_ct, axes_ct = plt.subplots(3, 3, figsize=(16, 15), dpi=300)

    for row_idx, organ in enumerate(ORGANS):
        dat_ref = full_data[organ]["Heuristic Raster"]
        t_internal = dat_ref["t_internal"]
        vox_center = world_to_voxel(t_internal)

        slice_axis = 1 if organ == "liver" else 0
        slice_idx = int(np.clip(round(vox_center[slice_axis]), 0, ct_shape[slice_axis] - 1))

        if slice_axis == 0:
            horiz_axis = 1
            vert_axis = 2
            plane_name = "Sagittal"
        else:
            horiz_axis = 0
            vert_axis = 2
            plane_name = "Coronal"

        gt_vox = np.array([world_to_voxel(p) for p in dat_ref["gt_all"]])
        in_slice_mask = np.abs(gt_vox[:, slice_axis] - slice_idx) <= 3.0
        gt_in_slice = gt_vox[in_slice_mask]

        # Crop extent around organ in CT voxel space
        h_min = max(0, int(np.min(gt_vox[:, horiz_axis])) - 20)
        h_max = min(ct_shape[horiz_axis], int(np.max(gt_vox[:, horiz_axis])) + 20)
        v_min = max(0, int(np.min(gt_vox[:, vert_axis])) - 20)
        v_max = min(ct_shape[vert_axis], int(np.max(gt_vox[:, vert_axis])) + 20)

        ct_disp = np.clip((ct_vol - (-160)) / (240 - (-160)), 0, 1)

        for col_idx, method in enumerate(METHODS):
            ax = axes_ct[row_idx, col_idx]
            dat = full_data[organ][method]

            if slice_axis == 0:
                slice_disp = ct_disp[slice_idx, h_min:h_max, v_min:v_max].T
            else:
                slice_disp = ct_disp[h_min:h_max, slice_idx, v_min:v_max].T

            ax.imshow(slice_disp, cmap='gray', origin='lower', extent=[h_min, h_max, v_min, v_max], aspect='equal')

            # Full Ground Truth Organ Contour in soft white
            if len(gt_in_slice) > 0:
                ax.scatter(gt_in_slice[:, horiz_axis], gt_in_slice[:, vert_axis],
                           c='#ecf0f1', s=8, alpha=0.6, edgecolors='none', label='Full CT Contour')

            # Discovered Ultrasound Points within 3mm of plane
            if len(dat["pts_all"]) > 0:
                pts_vox = np.array([world_to_voxel(p) for p in dat["pts_all"]])
                pts_in_slice = pts_vox[np.abs(pts_vox[:, slice_axis] - slice_idx) <= 3.0]
                if len(pts_in_slice) > 0:
                    ax.scatter(pts_in_slice[:, horiz_axis], pts_in_slice[:, vert_axis],
                               c='#00e5ff', s=14, alpha=0.9, edgecolors='black', linewidth=0.3, label='US Hits (<3mm)')

            # Probe trajectory projected onto slice
            probe_vox = np.array([world_to_voxel(p) for p in dat["probe_path"]])
            ax.plot(probe_vox[:, horiz_axis], probe_vox[:, vert_axis],
                    color='#e67e22', linestyle='-', linewidth=2.2, label='Probe Trajectory')
            ax.scatter(probe_vox[0, horiz_axis], probe_vox[0, vert_axis],
                       color='#d35400', s=45, marker='o', edgecolors='black')

            ax.set_xlim(h_min, h_max)
            ax.set_ylim(v_min, v_max)
            ax.set_axis_off()

            title = f"{method}\nCov: {dat['cov_pct']:.1f}% | Path: {dat['path_len']:.2f} m"
            if col_idx == 0:
                title = f"[{organ.upper()} - {plane_name}]\n" + title
            ax.set_title(title, fontsize=11, fontweight='bold', pad=4)

            if row_idx == 0 and col_idx == 0:
                ax.legend(loc='lower right', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    pdf_ct_path = FIG_DIR / "figure_reconstruction_ct_slice_overlay.pdf"
    png_ct_path = FIG_DIR / "figure_reconstruction_ct_slice_overlay.png"
    fig_ct.savefig(str(pdf_ct_path), dpi=300, bbox_inches='tight')
    fig_ct.savefig(str(png_ct_path), dpi=300, bbox_inches='tight')
    plt.close(fig_ct)
    print(f"[SUCCESS] Wrote CT Slice Overlay: {png_ct_path} & {pdf_ct_path}")

    # ==============================================================================
    # FIGURE 3: DIRECT SIDE-BY-SIDE (Full CT Ground Truth vs Ultrasound Reconstructed)
    # 3 organs (Spine, Aorta, Liver)
    # ==============================================================================
    fig_comp = plt.figure(figsize=(16, 18), dpi=300)

    for row_idx, organ in enumerate(ORGANS):
        elev, azim = view_angles[organ]
        dat_best = full_data[organ]["PPO + Memory"]
        gt_mm = dat_best["gt_all"] * 1000.0
        pts_mm = dat_best["pts_all"] * 1000.0

        margin = 25.0
        xlim = (np.min(gt_mm[:, 0]) - margin, np.max(gt_mm[:, 0]) + margin)
        ylim = (np.min(gt_mm[:, 1]) - margin, np.max(gt_mm[:, 1]) + margin)
        zlim = (np.min(gt_mm[:, 2]) - margin, np.max(gt_mm[:, 2]) + margin)

        # Panel 1: Full Organ CT Ground Truth
        ax1 = fig_comp.add_subplot(3, 2, row_idx * 2 + 1, projection='3d')
        n_gt = len(gt_mm)
        sub_gt = np.random.choice(n_gt, min(n_gt, 4000), replace=False) if n_gt > 4000 else slice(None)
        ax1.scatter(gt_mm[sub_gt, 0], gt_mm[sub_gt, 1], gt_mm[sub_gt, 2],
                    c='#34495e', s=6, alpha=0.5, edgecolors='none')
        ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.set_zlim(zlim)
        ax1.view_init(elev=elev, azim=azim)
        organ_name = {"spine": "Lumbar Spine L1-L5", "aorta": "Abdominal Aorta", "liver": "Whole Liver"}[organ]
        ax1.set_title(f"Full Ground Truth CT Shape\n[{organ_name}] ({len(gt_mm):,} surface points)", fontsize=12, fontweight='bold')

        # Panel 2: Full Ultrasound 3D Reconstruction Overlay
        ax2 = fig_comp.add_subplot(3, 2, row_idx * 2 + 2, projection='3d')
        # Background faint CT
        ax2.scatter(gt_mm[sub_gt, 0], gt_mm[sub_gt, 1], gt_mm[sub_gt, 2],
                    c='#bdc3c7', s=4, alpha=0.2, edgecolors='none')
        # Reconstructed Points colored by distance to CT
        if len(pts_mm) > 0:
            tree_gt = cKDTree(gt_mm)
            dists_mm, _ = tree_gt.query(pts_mm)
            n_pts = len(pts_mm)
            sub_pts = np.random.choice(n_pts, min(n_pts, 3000), replace=False) if n_pts > 3000 else slice(None)
            sc = ax2.scatter(pts_mm[sub_pts, 0], pts_mm[sub_pts, 1], pts_mm[sub_pts, 2],
                             c=dists_mm[sub_pts], cmap='viridis_r', vmin=0.0, vmax=5.0, s=10, alpha=0.9, edgecolors='none')
            cbar = plt.colorbar(sc, ax=ax2, shrink=0.6, pad=0.1)
            cbar.set_label('Error to CT Surface (mm)', fontsize=9)

        ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_zlim(zlim)
        ax2.view_init(elev=elev, azim=azim)
        acc_val = dat_best['acc_mm']
        ax2.set_title(f"PPO+Memory US Reconstruction Overlay\nAccuracy: {acc_val:.2f} mm | Cov: {dat_best['cov_pct']:.1f}%", fontsize=12, fontweight='bold')

    plt.tight_layout()
    comp_path = FIG_DIR / "full_organ_ct_vs_ultrasound_comparison.png"
    fig_comp.savefig(str(comp_path), dpi=300, bbox_inches='tight')
    plt.close(fig_comp)
    print(f"[SUCCESS] Wrote Direct CT vs US Comparison Gallery: {comp_path}")
    print("\n[ALL FIGURES SUCCESSFULLY GENERATED]")


if __name__ == "__main__":
    main()
