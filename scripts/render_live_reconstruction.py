#!/usr/bin/env python3
"""
scripts/render_live_reconstruction.py
=====================================
Renders a side-by-side live video/animation of the autonomous reconstruction episode:
- Panel 1 (Left): 3D scientific reconstruction (Ground truth spine mesh + real-time discovered voxels + probe tip position).
- Panel 2 (Top-Right): Live simulated B-mode ultrasound image captured by the probe at that exact step.
- Panel 3 (Bottom-Right): Live coverage trajectory (%) over time (Step 0 to 300).

Outputs:
- results/recon_vis/live_recon_spine_s0344.mp4
- results/recon_vis/live_recon_spine_s0344.gif
"""

import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.spatial import cKDTree
from stable_baselines3 import PPO

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from robotic_us_env import RoboticUltrasoundGymEnv
from scripts.step2_recon_eval import (
    CHECKPOINTS, OUTPUT_DIR, extract_organ_mesh_and_ref_voxels, render_panel
)

def main():
    parser = argparse.ArgumentParser(description="Live 3D Ultrasound Reconstruction Visualizer & Video Generator")
    parser.add_argument("--target", type=str, default="spine", choices=["spine", "aorta", "liver"], help="Target anatomy")
    parser.add_argument("--patient", type=str, default="s0344", help="Patient ID (e.g., s0344, s0058)")
    parser.add_argument("--method", type=str, default="ppo_memory", choices=["ppo_memory", "ppo_nomem"], help="Reconstruction method")
    parser.add_argument("--steps", type=int, default=300, help="Maximum episode steps")
    parser.add_argument("--interval", type=int, default=5, help="Frame capture interval in steps")
    parser.add_argument("--gui", action="store_true", help="Launch interactive PyBullet 3D GUI window")
    parser.add_argument("--skin-opacity", type=float, default=0.0, help="Patient skin opacity in GUI mode (0.0=invisible, 0.15=transparent)")
    parser.add_argument("--show-window", action="store_true", help="Display the 3-panel reconstruction figure live on screen")
    parser.add_argument("--sleep", type=float, default=0.02, help="Sleep duration per step in GUI mode (seconds)")
    args = parser.parse_args()

    patient = args.patient
    target = args.target
    method = args.method
    max_steps = args.steps
    step_interval = args.interval
    seed = 42
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient)

    print(f"[LIVE] Initializing environment for {target} ({patient}, method: {method}, seed: {seed}, GUI: {args.gui})...")
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
        render_mode="human" if args.gui else "direct",
        device="cpu"
    )

    obs, _ = env.reset(seed=seed)
    pv_mesh, ref_voxels_mm, upper_surface_voxels_mm = extract_organ_mesh_and_ref_voxels(env, target)
    tree_upper = cKDTree(upper_surface_voxels_mm)

    import pybullet as p
    if args.gui:
        # Make patient skin transparent/invisible so internal anatomy and point cloud are clearly visible
        p.changeVisualShape(env.body_id, -1, rgbaColor=[0.9, 0.8, 0.7, args.skin_opacity])
        print(f"[LIVE GUI] Set patient skin opacity to {args.skin_opacity} (interior organ visible).")

        # Focus PyBullet debug camera onto the target organ center
        mesh_center_m = [float(c) / 1000.0 for c in pv_mesh.center]
        p.resetDebugVisualizerCamera(cameraDistance=0.45, cameraYaw=55, cameraPitch=-30, cameraTargetPosition=mesh_center_m)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Draw reference organ surface points in faint gray for spatial context
        ref_pts_m = (ref_voxels_mm / 1000.0)[::6].tolist()
        p.addUserDebugPoints(ref_pts_m, [[0.7, 0.7, 0.7]] * len(ref_pts_m), pointSize=2)
        print(f"[LIVE GUI] Rendered {len(ref_pts_m)} reference anatomical points into PyBullet viewport.")

    ckpt = CHECKPOINTS[target][method]
    model = PPO.load(str(ckpt), device="cpu")

    probe_xyz = [env.current_probe_tip_pos.copy() * 1000.0]
    covered_upper_mask = np.zeros(len(upper_surface_voxels_mm), dtype=bool)
    seen_voxels = set()
    coverage_curve = [0.0]
    frames = []

    fig = plt.figure(figsize=(14, 7), dpi=150)
    
    print("[LIVE] Stepping episode, tracking probe, and rendering live point cloud...")
    for step in range(max_steps):
        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)

        curr_p_mm = env.current_probe_tip_pos.copy() * 1000.0
        probe_xyz.append(curr_p_mm)

        if args.gui and len(probe_xyz) > 1:
            p1 = (probe_xyz[-2] / 1000.0).tolist()
            p2 = (probe_xyz[-1] / 1000.0).tolist()
            p.addUserDebugLine(p1, p2, [0.9, 0.1, 0.1], lineWidth=2)

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

                if args.gui:
                    new_pts_m = (np.array(new_voxels, dtype=np.float64) * 0.0015).tolist()
                    p.addUserDebugPoints(new_pts_m, [[0.0, 0.95, 0.35]] * len(new_pts_m), pointSize=5)

        cov_pct = float(np.mean(covered_upper_mask) * 100.0)
        coverage_curve.append(cov_pct)

        if args.gui and args.sleep > 0:
            import time
            time.sleep(args.sleep)

        # Capture visual frame
        if step % step_interval == 0 or step == max_steps - 1:
            disc_voxels = np.array(list(seen_voxels), dtype=np.float64) * 1.5 if len(seen_voxels) > 0 else np.empty((0, 3))
            traj_xyz = np.array(probe_xyz)

            # Render 3D PyVista screenshot
            img_3d = render_panel(pv_mesh, disc_voxels, traj_xyz, camera_mode="top", resolution=(800, 800))

            # Compose multi-panel frame
            fig.clf()
            gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
            
            # Left: 3D Scientific Reconstruction
            ax_3d = fig.add_subplot(gs[:, 0])
            ax_3d.imshow(img_3d)
            ax_3d.set_title(f"Live 3D Ultrasound Reconstruction (Step {step}/{max_steps})\nTarget: {target.upper()} ({patient}) | {method} | Upper Coverage: {cov_pct:.1f}%", fontsize=12, fontweight="bold", pad=8)
            ax_3d.set_axis_off()

            # Top-Right: Current B-Mode Ultrasound Image
            ax_us = fig.add_subplot(gs[0, 1])
            b_mode_img = obs["image"]
            if b_mode_img.ndim == 3:
                b_mode_img = b_mode_img[0]
            ax_us.imshow(b_mode_img, cmap="gray", vmin=0, vmax=255)
            ax_us.set_title(f"Real-Time B-Mode Slice (Transducer View)", fontsize=11, fontweight="bold", pad=6)
            ax_us.set_axis_off()

            # Bottom-Right: Live Cumulative Coverage Curve
            ax_cov = fig.add_subplot(gs[1, 1])
            ax_cov.plot(coverage_curve, color="#27ae60", linewidth=2.5, label="Upper Surface Coverage")
            ax_cov.scatter([len(coverage_curve)-1], [cov_pct], color="#e74c3c", s=60, zorder=5)
            ax_cov.set_xlim(0, max_steps)
            ax_cov.set_ylim(-2, 102)
            ax_cov.set_xlabel("Episode Step", fontsize=10)
            ax_cov.set_ylabel("Coverage (%)", fontsize=10)
            ax_cov.set_title("Coverage Progress vs Step", fontsize=11, fontweight="bold", pad=6)
            ax_cov.grid(True, linestyle="--", alpha=0.5)
            ax_cov.legend(fontsize=9, loc="lower right")

            plt.tight_layout()
            fig.canvas.draw()
            frame_rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame_rgb = frame_rgb.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            frames.append(frame_rgb)

            if args.show_window:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow("SonoGym Scientific 3D Reconstruction", frame_bgr)
                cv2.waitKey(1)

        if term or trunc:
            break

    env.close()
    if args.show_window:
        cv2.destroyAllWindows()
    plt.close(fig)

    print(f"[LIVE] Captured {len(frames)} frames. Encoding MP4 and GIF...")
    h, w, _ = frames[0].shape
    mp4_path = OUTPUT_DIR / f"live_recon_{target}_{patient}_{method}.mp4"
    gif_path = OUTPUT_DIR / f"live_recon_{target}_{patient}_{method}.gif"

    # Write MP4 via OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(mp4_path), fourcc, 8, (w, h))
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"[DONE] Saved live reconstruction video: {mp4_path}")

    # Write GIF via imageio (subsampling to keep size compact)
    import imageio
    imageio.mimsave(str(gif_path), frames[::2], fps=6)
    print(f"[DONE] Saved live reconstruction animation: {gif_path}")

if __name__ == "__main__":
    main()
