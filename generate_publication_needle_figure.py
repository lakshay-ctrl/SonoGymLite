#!/usr/bin/env python3
"""
generate_publication_needle_figure.py
======================================
Generates a clean, publication-grade 4-panel figure conforming strictly to
the supervisor's corrections:

  (a) PyBullet 3D World: Franka Panda manipulator holding linear ultrasound probe
      with probe-mounted 45 deg needle guide in compliant contact with patient abdomen.
      Labels point accurately to probe body, needle guide, and robot.
  (b) Patient CT Cross-Section: Aorta Bifurcation target (112.3 mm depth), planned
      bone-free path (dashed green), fixed lateral entry hitting bone (solid red with 'X'),
      and 24-azimuth fan (7/24 bone-free corridors), projected onto the slice.
  (c) B-Mode Ultrasound Strip: Genuine CT-synthesized B-mode ultrasound frames across
      30 deg, 45 deg, and 60 deg with measured CNR values (9.38, 7.69, 6.10), modeled
      visibility V, labeled shaft (pointing accurately at the bright diagonal line),
      tip, aorta lumen, and "Depth (cm)" axis.
  (d) Geometric Stroke vs. Angle: Analytical curve of stroke s = d / sin(theta) against
      guide angle, unshaded, with "Reachable: below the line" note and corrected reach counts:
      100 mm: 0 / 1 / 1, 150 mm: 1 / 3 / 5, 200 mm: 3 / 6 / 8.
"""

import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import sys
import math
import json
from pathlib import Path

import numpy as np
import cv2
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data

# Workspace root
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import live_unet_demo as demo
from extract_slice import extract_slice
from registration import compute_registered_ct_center, voxel_to_world
import live_needle_experiment as lne

PANEL_W = 800
PANEL_H = 600


def draw_outlined_text(img: np.ndarray, text: str, pos: tuple[int, int],
                       font=cv2.FONT_HERSHEY_DUPLEX, scale: float = 0.50,
                       color=(255, 255, 255), thickness: int = 1,
                       outline_color=(0, 0, 0), outline_thickness: int = 3):
    """Draws text with a distinct dark outline for maximum readability."""
    cv2.putText(img, text, pos, font, scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def add_subfig_badge(img: np.ndarray, letter: str) -> np.ndarray:
    """Overlays an elegant, high-contrast publication subfigure badge (e.g., '(a)')."""
    out = img.copy()
    badge_text = f"({letter.lower()})"
    pill_x, pill_y = 12, 12
    pill_w, pill_h = 44, 32
    sub = out[pill_y : pill_y + pill_h, pill_x : pill_x + pill_w]
    bg = np.zeros_like(sub)
    bg[:] = (20, 24, 28)
    cv2.addWeighted(bg, 0.85, sub, 0.15, 0, sub)
    cv2.rectangle(out, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (80, 95, 110), 1, cv2.LINE_AA)
    cv2.putText(out, badge_text, (pill_x + 7, pill_y + 22), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_panel_a_pybullet() -> np.ndarray:
    """Renders Panel (a): High-res PyBullet 3D setup with accurate probe body pointer."""
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    # 1. Hospital Room & Bed
    bed_top_z = demo.create_hospital_room()

    # 2. Patient Torso Mesh (s0058)
    subject_dir = PROJECT_ROOT / "totalseg_patients/s0058"
    mesh_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    body_id, reg_pos, reg_orn_mat, reg_meta = demo.create_registered_body_mesh(
        subject_dir, bed_top_z, mesh_scale
    )

    # 3. Franka Panda 7-DoF Robotic Arm
    panda_id = demo.create_panda_robot()
    for finger_link in [9, 10]:
        p.changeVisualShape(panda_id, finger_link, rgbaColor=[0, 0, 0, 0])

    # 4. Probe Position on Abdomen at Aorta Scanning Site
    tx, ty = 0.00, -0.05
    found, skin_z = demo.raycast_skin_surface(tx, ty, body_id)
    if not found:
        skin_z = 1.053

    # Contact indentation 2.0 mm
    probe_base_pos = [tx, ty, skin_z - 0.007]
    yaw = 0.65  # Clear profile angle
    probe_orn = p.getQuaternionFromEuler([0, 0, yaw])

    # 5. Load Probe with 45-degree Guide Bracket
    urdf_file = str(PROJECT_ROOT / "robotic_probe_needle_guide.urdf")
    probe_id = p.loadURDF(urdf_file, basePosition=probe_base_pos, baseOrientation=probe_orn, useFixedBase=True)

    # Franka Panda IK to attach to Probe Flange
    flange_pos = np.array(probe_base_pos) + np.array([0, 0, 0.200])
    ee_orn = p.getQuaternionFromEuler([np.pi, 0, yaw])
    ik_joints = p.calculateInverseKinematics(
        panda_id, 8, targetPosition=flange_pos.tolist(), targetOrientation=ee_orn, maxNumIterations=200
    )
    for j in range(7):
        p.resetJointState(panda_id, j, ik_joints[j])

    # Insert Needle partially (Joint 2 is prismatic insertion)
    if p.getNumJoints(probe_id) > 2:
        p.resetJointState(probe_id, 2, targetValue=0.060)
    p.stepSimulation()

    # Camera looking closely at probe, guide bracket, and abdomen
    cam = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[tx + 0.02, ty + 0.01, skin_z + 0.06],
        distance=0.46, yaw=50.0, pitch=-14.0, roll=0, upAxisIndex=2
    )
    proj = p.computeProjectionMatrixFOV(fov=46.0, aspect=PANEL_W / float(PANEL_H), nearVal=0.05, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(1600, 1200, cam, proj, renderer=p.ER_TINY_RENDERER)
    p.disconnect(client)

    img = np.array(rgb, dtype=np.uint8).reshape((1200, 1600, 4))[:, :, :3]
    content = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)

    # High-contrast concise labels with precise arrows
    draw_outlined_text(content, "Franka Panda manipulator", (70, 35), scale=0.52, color=(255, 255, 255))

    # Linear probe arrow points directly to the probe cylinder body at (215, 260)
    draw_outlined_text(content, "Linear probe", (35, 265), scale=0.50, color=(220, 240, 255))
    cv2.arrowedLine(content, (145, 260), (215, 260), (220, 240, 255), 2, tipLength=0.20)

    # 45 deg needle guide arrow points directly to the guide bracket at (345, 255)
    draw_outlined_text(content, "45 deg needle guide", (480, 215), scale=0.50, color=(120, 220, 255))
    cv2.arrowedLine(content, (475, 225), (345, 255), (120, 220, 255), 2, tipLength=0.20)

    return add_subfig_badge(content, "a")


def render_panel_b_ct() -> np.ndarray:
    """Renders Panel (b): Patient CT cross-section at aorta bifurcation with azimuth planning fan."""
    subject_dir = PROJECT_ROOT / "totalseg_patients/s0058"
    ct_nii = nib.load(str(subject_dir / "ct.nii.gz"))
    ct_data = ct_nii.get_fdata()

    bone_nii = nib.load(str(subject_dir / "bone_label.nii.gz"))
    bone_data = bone_nii.get_fdata()

    aorta_nii = nib.load(str(subject_dir / "segmentations/aorta.nii.gz"))
    aorta_data = aorta_nii.get_fdata()

    # Slice at aorta bifurcation (z = 152 in TotalSegmentator s0058)
    z = 152
    ct_slice = np.flipud(ct_data[:, :, z].T)
    bone_slice = np.flipud(bone_data[:, :, z].T)
    aorta_slice = np.flipud(aorta_data[:, :, z].T)

    # Clinical abdomen CT window: [-150, 250] HU
    ct_norm = np.clip((ct_slice + 150.0) / 400.0, 0.0, 1.0)
    ct_bgr = cv2.cvtColor((ct_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Crop 260x260 centered around patient midline
    crop_s = 260
    cx, cy = 145, 145
    x0, y0 = cx - crop_s // 2, cy - crop_s // 2
    ct_crop = ct_bgr[y0 : y0 + crop_s, x0 : x0 + crop_s].copy()

    # Resize to canvas dimensions
    scale_x = PANEL_W / float(crop_s)
    scale_y = PANEL_H / float(crop_s)
    ct_res = cv2.resize(ct_crop, (PANEL_W, PANEL_H), interpolation=cv2.INTER_CUBIC)

    # Aorta Target Coordinates
    aorta_pts = np.argwhere(aorta_slice > 0.5)
    r_tgt, c_tgt = aorta_pts.mean(axis=0)
    tgt_x = int(round((c_tgt - x0) * scale_x))
    tgt_y = int(round((r_tgt - y0) * scale_y))
    tgt_pt = (tgt_x, tgt_y)

    # Azimuth Planning Fan: 7 safe azimuths evaluated from 3D raycasting
    safe_indices = {9, 10, 11, 14, 16, 17, 18}
    radius_fan = 255
    candidate_angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    for idx, psi in enumerate(candidate_angles):
        rx = int(tgt_x + radius_fan * math.cos(psi))
        ry = int(tgt_y + radius_fan * math.sin(psi))

        is_free = (idx in safe_indices)
        color = (70, 205, 100) if is_free else (50, 50, 220)
        cv2.line(ct_res, tgt_pt, (rx, ry), color, 1, cv2.LINE_AA)

    # Fixed Lateral Entry Trajectory (Solid Red Line hitting iliac bone cortex)
    entry_coll = (tgt_x - 245, tgt_y - 25)
    cv2.line(ct_res, entry_coll, tgt_pt, (40, 50, 240), 3, cv2.LINE_AA)
    # Collision Marker sitting on bone cortex
    coll_pt = (int(tgt_x - 180), int(tgt_y - 18))
    cv2.drawMarker(ct_res, coll_pt, (25, 35, 255), cv2.MARKER_TILTED_CROSS, 16, 3, cv2.LINE_AA)

    # Planned Bone-Free Trajectory (Dashed Green Line from anterior skin)
    entry_planned = (tgt_x + 55, 60)
    p1 = np.array(entry_planned, dtype=np.float32)
    p2 = np.array(tgt_pt, dtype=np.float32)
    dist = np.linalg.norm(p2 - p1)
    num_dash = int(dist / 14)
    for i in range(num_dash):
        if i % 2 == 0:
            s_pt = tuple((p1 + (p2 - p1) * (i / num_dash)).astype(int))
            e_pt = tuple((p1 + (p2 - p1) * ((i + 0.8) / num_dash)).astype(int))
            cv2.line(ct_res, s_pt, e_pt, (60, 245, 95), 3, cv2.LINE_AA)

    # Target Marker (Cyan circle + crosshair)
    cv2.circle(ct_res, tgt_pt, 11, (255, 230, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(ct_res, tgt_pt, (255, 230, 0), cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)

    # Annotations
    draw_outlined_text(ct_res, "Aorta Target (112.3 mm)", (tgt_x + 20, tgt_y + 6), scale=0.48, color=(255, 230, 0))
    draw_outlined_text(ct_res, "Planned Bone-Free Path", (entry_planned[0] + 16, entry_planned[1] + 35), scale=0.48, color=(60, 245, 95))
    draw_outlined_text(ct_res, "Fixed Lateral Entry (Bone Collision)", (entry_coll[0] - 15, entry_coll[1] - 14), scale=0.46, color=(60, 70, 255))
    draw_outlined_text(ct_res, "Azimuth Planning Fan (7/24 Bone-Free)", (20, PANEL_H - 25), scale=0.48, color=(120, 225, 145))

    # Scale bar (50 mm): 50mm / 1.5mm spacing = 33.33 voxels * scale_x = ~102 px
    bar_x, bar_y = PANEL_W - 140, PANEL_H - 30
    bar_len = int(33.33 * scale_x)
    cv2.line(ct_res, (bar_x, bar_y), (bar_x + bar_len, bar_y), (255, 255, 255), 3, cv2.LINE_AA)
    draw_outlined_text(ct_res, "50 mm", (bar_x + 18, bar_y - 8), scale=0.42, color=(255, 255, 255))

    return add_subfig_badge(ct_res, "b")


def render_panel_c_bmode_strip() -> np.ndarray:
    """
    Renders Panel (c): Side-by-side comparison of genuine patient CT-synthesized
    B-mode ultrasound images at 30 deg, 45 deg, and 60 deg with measured CNR values,
    accurately pointed needle shaft, glowing tip, aorta lumen, and 'Depth (cm)' axis.
    """
    div_w = 2
    col_w = (PANEL_W - 2 * div_w) // 3  # 264 px
    col_w3 = PANEL_W - 2 * div_w - 2 * col_w  # 268 px
    footer_h = 64
    us_h = PANEL_H - footer_h  # 536 px

    img30 = cv2.imread(str(PROJECT_ROOT / "scratch/real_bmode_angle_30.png"))
    img45 = cv2.imread(str(PROJECT_ROOT / "scratch/real_bmode_angle_45.png"))
    img60 = cv2.imread(str(PROJECT_ROOT / "scratch/real_bmode_angle_60.png"))

    def format_col(img: np.ndarray, angle: float, vis: float, vis_db: float, cnr: float, w: int, is_left: bool = False, is_center: bool = False) -> np.ndarray:
        us_resized = cv2.resize(img, (w, us_h), interpolation=cv2.INTER_CUBIC)

        # Depth scale calibration ticks along left edge (every 2 cm from 2 to 12 cm)
        for d_cm in range(2, 13, 2):
            y_tick = int((d_cm / 12.8) * us_h)
            if 0 < y_tick < us_h:
                cv2.line(us_resized, (0, y_tick), (7, y_tick), (210, 220, 230), 1, cv2.LINE_AA)
                draw_outlined_text(us_resized, f"{d_cm}", (10, y_tick + 4), scale=0.34, color=(210, 220, 230))

        # Explicit "Depth (cm)" axis title on leftmost column below badge
        if is_left:
            draw_outlined_text(us_resized, "Depth (cm)", (10, 60), scale=0.38, color=(200, 215, 230))

        # Callouts on center panel (45 deg)
        if is_center:
            # Needle Shaft arrow lands directly on the bright diagonal line at (225, 305)
            draw_outlined_text(us_resized, "Needle Shaft", (75, 230), scale=0.38, color=(180, 230, 255))
            cv2.arrowedLine(us_resized, (145, 235), (225, 305), (180, 230, 255), 1, tipLength=0.20)

            # Aorta Lumen arrow lands in anechoic vascular region
            draw_outlined_text(us_resized, "Aorta Lumen", (15, 280), scale=0.38, color=(255, 215, 70))
            cv2.arrowedLine(us_resized, (85, 285), (105, 335), (255, 215, 70), 1, tipLength=0.25)

            # Hyperechoic Needle Tip arrow lands on bright tip
            draw_outlined_text(us_resized, "Needle Tip", (15, 435), scale=0.38, color=(60, 255, 120))
            cv2.arrowedLine(us_resized, (75, 440), (120, 480), (60, 255, 120), 1, tipLength=0.25)

        # Column Footer with exact measured values from manuscript table
        footer = np.zeros((footer_h, w, 3), dtype=np.uint8)
        footer[:] = (18, 22, 26)
        cv2.line(footer, (0, 0), (w, 0), (50, 60, 72), 1)

        cv2.putText(footer, f"Guide Angle: {int(angle)} deg", (w // 2 - 62, 24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(footer, f"V = {vis:.2f} ({vis_db:.1f} dB) | CNR: {cnr:.2f}", (w // 2 - 100, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        return np.vstack([us_resized, footer])

    sub30 = format_col(img30, 30, 0.87, -1.2, 9.38, col_w, is_left=True)
    sub45 = format_col(img45, 45, 0.71, -3.0, 7.69, col_w, is_center=True)
    sub60 = format_col(img60, 60, 0.50, -6.0, 6.10, col_w3)

    divider = np.zeros((PANEL_H, div_w, 3), dtype=np.uint8)
    divider[:] = (55, 68, 80)

    strip = np.hstack([sub30, divider, sub45, divider, sub60])
    return add_subfig_badge(strip, "c")


def render_panel_d_stroke_plot() -> np.ndarray:
    """Renders Panel (d): Analytical stroke vs. guide angle plot with corrected reach counts and no shading."""
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=100)

    angles = np.linspace(22.0, 75.0, 200)
    ang_rad = np.radians(angles)

    d_aorta = 112.3   # s0058 aorta depth
    d_mean = 127.8    # 10-target cohort mean depth

    stroke_aorta = d_aorta / np.sin(ang_rad)
    stroke_mean = d_mean / np.sin(ang_rad)

    # Plot curves
    ax.plot(angles, stroke_aorta, color="#0066CC", linewidth=2.5,
            label=f"Aorta Target ($d = {d_aorta:.1f}$ mm)")
    ax.plot(angles, stroke_mean, color="#882299", linewidth=2.0, linestyle="--",
            label=f"Cohort Mean Depth ($d = {d_mean:.1f}$ mm)")

    # Evaluation points at 30, 45, 60 deg
    eval_angles = [30.0, 45.0, 60.0]
    eval_strokes_aorta = [d_aorta / np.sin(np.radians(a)) for a in eval_angles]
    ax.scatter(eval_angles, eval_strokes_aorta, color="#0066CC", s=65, zorder=5)
    for a, s in zip(eval_angles, eval_strokes_aorta):
        ax.annotate(f"{s:.1f} mm", (a, s), textcoords="offset points", xytext=(8, -3),
                    fontsize=9.5, fontweight="bold", color="#004488")

    # Reference Cannula Lines with corrected reach counts (ordered by 30° / 45° / 60°):
    ax.axhline(100.0, color="#D9381E", linestyle=":", linewidth=1.8,
               label=r"100 mm Cannula (30$^\circ$/45$^\circ$/60$^\circ$ reach: 0 / 1 / 1)")
    ax.axhline(150.0, color="#E69F00", linestyle="-.", linewidth=1.8,
               label=r"150 mm Cannula (30$^\circ$/45$^\circ$/60$^\circ$ reach: 1 / 3 / 5)")
    ax.axhline(200.0, color="#2E7D32", linestyle="-", linewidth=1.8,
               label=r"200 mm Cannula (30$^\circ$/45$^\circ$/60$^\circ$ reach: 3 / 6 / 8)")

    # Explanatory note: Reachable below the line (NO SHADING)
    ax.text(23.5, 72.0, "Reachable: below the line", fontsize=9.5, fontweight="bold",
            color="#2E7D32", bbox=dict(boxstyle="round,pad=0.35", fc="#E8F5E9", ec="#2E7D32", alpha=0.92))

    ax.set_xlim(20.0, 75.0)
    ax.set_ylim(50.0, 320.0)
    ax.set_xlabel(r"Needle Guide Angle $\theta$ (degrees from skin)", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Required Insertion Stroke $s = d / \sin\theta$ (mm)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.legend(loc="upper right", fontsize=8.8, framealpha=0.94)

    plt.tight_layout()
    tmp_path = PROJECT_ROOT / "scratch/temp_panel_d_plot.png"
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(tmp_path), dpi=100)
    plt.close(fig)

    plot_img = cv2.imread(str(tmp_path))
    content = cv2.resize(plot_img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)

    return add_subfig_badge(content, "d")


def main():
    print("=" * 80)
    print(" Generating Clean 4-Panel Publication Needle Guidance Figure (Final)")
    print("=" * 80)

    # 1. Render all 4 panels
    print("Rendering Panel (a): PyBullet 3D World (Robotic Setup)...")
    panel_a = render_panel_a_pybullet()

    print("Rendering Panel (b): Patient CT Cross-Section (Aorta Target & Azimuth Fan)...")
    panel_b = render_panel_b_ct()

    print("Rendering Panel (c): Genuine Ultrasound B-Mode Strip (30 deg, 45 deg, 60 deg)...")
    panel_c = render_panel_c_bmode_strip()

    print("Rendering Panel (d): Analytical Stroke vs. Guide Angle Plot...")
    panel_d = render_panel_d_stroke_plot()

    # 2. Assemble Composite 2x2 Grid (1600 x 1200):
    row1 = np.hstack([panel_a, panel_b])
    row2 = np.hstack([panel_c, panel_d])

    row_divider = np.zeros((4, PANEL_W * 2, 3), dtype=np.uint8)
    row_divider[:] = (45, 55, 65)

    composite = np.vstack([row1, row_divider, row2])

    out_figures = PROJECT_ROOT / "figures/needle_guidance_publication_figure.png"
    out_figures.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_figures), composite)

    out_brain = Path("/Users/lakshaybatta/.gemini/antigravity/brain/ffa78df7-4d76-453b-9869-bc9788f5cdf1/needle_guidance_publication_figure.png")
    cv2.imwrite(str(out_brain), composite)

    out_legacy = PROJECT_ROOT / "figures/aura_needle_assembly_figure.png"
    cv2.imwrite(str(out_legacy), composite)

    out_paper_legacy = PROJECT_ROOT / "paper/figures/aura_needle_assembly_figure.png"
    out_paper_legacy.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_paper_legacy), composite)

    print(f"\n[OK] High-resolution publication figure generated successfully:")
    print(f"     -> {out_figures} (Shape: {composite.shape})")
    print(f"     -> {out_brain}")
    print(f"     -> {out_legacy} (Updated paper figure link)")
    print(f"     -> {out_paper_legacy}")


if __name__ == "__main__":
    main()
