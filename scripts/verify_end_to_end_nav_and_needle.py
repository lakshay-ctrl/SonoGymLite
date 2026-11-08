#!/usr/bin/env python3
"""
scripts/verify_end_to_end_nav_and_needle.py
============================================
Full End-to-End Verification of Autonomous Ultrasound Navigation + Needle Insertion.

Demonstrates:
  Phase 1: Goal-Conditioned PPO policy autonomously navigates the ultrasound probe
           from a distant random initial placement on the skin (pos_err ~50 mm)
           across the surface to the target acoustic window (pos_err < 5 mm, rot_err < 10 deg).
  Phase 2: Once navigation reaches first success, the probe locks pose, mounts the
           virtual in-plane needle guide, and dynamically advances the needle cannula
           into the patient toward the internal anatomical target.
  Phase 3: Verifies bone clearance via CT bone raycasting and checks target hit criteria.
"""

import sys
import os
import time
import math
import argparse
import json
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import numpy as np
import cv2
import nibabel as nib
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import live_unet_demo as demo
from registration import compute_registered_ct_center, load_registration_meta, voxel_to_world
from surface_motion_planner import load_or_create_surface_map
from robotic_us_env import RoboticUltrasoundGymEnv, snap_panda_to_pose
from scripts.step5b_needle_nav_hit import compute_tilt_rotation_matrix


def draw_end_to_end_hud(
    bmode_img: np.ndarray,
    target_name: str,
    patient: str,
    phase: str,
    step_num: int,
    pos_err_mm: float,
    rot_err_deg: float,
    guide_angle: int,
    needle_len: float,
    curr_stroke: float,
    req_stroke: float,
    bone_hit: bool,
    target_hit: bool,
    tip_dist_mm: float
) -> np.ndarray:
    """Draws a combined navigation + needle guidance HUD."""
    if bmode_img.ndim == 3:
        bmode_img = bmode_img[0]
    if bmode_img.dtype != np.uint8:
        if bmode_img.max() <= 1.0:
            bmode_img = (bmode_img * 255.0).astype(np.uint8)
        else:
            bmode_img = np.clip(bmode_img, 0, 255).astype(np.uint8)

    us_display = cv2.resize(bmode_img, (400, 400), interpolation=cv2.INTER_CUBIC)
    us_bgr = cv2.cvtColor(us_display, cv2.COLOR_GRAY2BGR)

    # Needle overlay if in needle phase
    if phase == "NEEDLE_INSERTION" and curr_stroke > 0:
        ang_rad = math.radians(guide_angle)
        vis_factor = max(0.1, math.cos(ang_rad))
        frac = min(curr_stroke / max(req_stroke, 1e-3), 1.0)
        start_pt = (50, 40)
        end_pt = (int(50 + frac * 280 * math.cos(ang_rad)), int(40 + frac * 320 * math.sin(ang_rad)))
        intensity = int(180 + 75 * vis_factor)
        cv2.line(us_bgr, start_pt, end_pt, (intensity, intensity, intensity), 3, cv2.LINE_AA)
        cv2.circle(us_bgr, end_pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # Canvas (840 x 460)
    canvas = np.zeros((460, 840, 3), dtype=np.uint8)
    canvas[:] = (20, 24, 30)

    # Place B-mode image
    canvas[30:430, 20:420] = us_bgr
    cv2.rectangle(canvas, (18, 28), (422, 432), (80, 90, 105), 2)
    cv2.putText(canvas, f"B-MODE ULTRASOUND SLICE", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    tx = 440
    cv2.putText(canvas, "END-TO-END SONOSIMLITE PIPELINE", (tx, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (tx, 55), (820, 55), (60, 70, 85), 2)

    # Phase Banner
    if phase == "NAVIGATION":
        cv2.rectangle(canvas, (tx, 65), (820, 100), (90, 60, 20), -1)
        cv2.rectangle(canvas, (tx, 65), (820, 100), (255, 180, 50), 1)
        cv2.putText(canvas, f"PHASE 1: PROBE NAVIGATION (STEP {step_num})", (tx + 15, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (tx, 65), (820, 100), (30, 80, 110), -1)
        cv2.rectangle(canvas, (tx, 65), (820, 100), (50, 200, 255), 1)
        cv2.putText(canvas, "PHASE 2: NEEDLE CANNULA INSERTION", (tx + 15, 89), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Navigation Telemetry
    cv2.putText(canvas, f"Target Anatomy:  {target_name.upper()}", (tx, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Patient Subject: {patient}", (tx, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 220), 1, cv2.LINE_AA)

    nav_color = (0, 255, 0) if pos_err_mm < 5.0 and rot_err_deg < 10.0 else (0, 200, 255)
    cv2.putText(canvas, "Navigation Pos Error:", (tx, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{pos_err_mm:.2f} mm  (Threshold < 5.0 mm)", (tx + 200, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, nav_color, 2, cv2.LINE_AA)

    cv2.putText(canvas, "Navigation Tilt Error:", (tx, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{rot_err_deg:.2f} deg  (Threshold < 10.0 deg)", (tx + 200, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, nav_color, 2, cv2.LINE_AA)

    cv2.line(canvas, (tx, 235), (820, 235), (60, 70, 85), 1)

    # Needle Telemetry
    cv2.putText(canvas, f"Needle Guide: {guide_angle} deg | Length: {int(needle_len)} mm", (tx, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Stroke:", (tx, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{curr_stroke:.1f} / {req_stroke:.1f} mm", (tx + 80, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Bone Status:", (tx + 220, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    if bone_hit:
        cv2.putText(canvas, "BONE HIT!", (tx + 325, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "BONE FREE", (tx + 325, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Tip-to-Target Distance:", (tx, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
    tip_col = (0, 255, 0) if tip_dist_mm <= 5.0 else ((0, 200, 255) if tip_dist_mm < 20.0 else (100, 100, 255))
    cv2.putText(canvas, f"{tip_dist_mm:.2f} mm", (tx + 220, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.60, tip_col, 2, cv2.LINE_AA)

    # Status Banner
    if target_hit:
        cv2.rectangle(canvas, (tx, 355), (820, 420), (20, 130, 40), -1)
        cv2.rectangle(canvas, (tx, 355), (820, 420), (0, 255, 100), 2)
        cv2.putText(canvas, "TARGET HIT SUCCESSFUL", (tx + 50, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    elif phase == "NAVIGATION":
        cv2.rectangle(canvas, (tx, 355), (820, 420), (35, 45, 60), -1)
        cv2.rectangle(canvas, (tx, 355), (820, 420), (80, 95, 120), 1)
        cv2.putText(canvas, "POLICY NAVIGATING TO SLICE...", (tx + 25, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 50), 2, cv2.LINE_AA)
    elif curr_stroke >= needle_len:
        cv2.rectangle(canvas, (tx, 355), (820, 420), (120, 60, 20), -1)
        cv2.rectangle(canvas, (tx, 355), (820, 420), (0, 180, 255), 2)
        cv2.putText(canvas, "MAX CANNULA REACH EXCEEDED", (tx + 20, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (tx, 355), (820, 420), (30, 40, 55), -1)
        prog = min(max(curr_stroke / max(req_stroke, 1e-3), 0.0), 1.0)
        cv2.rectangle(canvas, (tx + 10, 375), (tx + 10 + int(prog * 340), 400), (0, 200, 255), -1)
        cv2.putText(canvas, f"INSERTING NEEDLE ({int(prog*100)}%)...", (tx + 40, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def run_end_to_end_verification(
    anatomy: str = "aorta",
    patient: str = "s0011",
    seed: int = 1000,
    guide_angle: int = 60,
    needle_len: float = 200.0,
    video_output_path: str = "results/videos/end_to_end_nav_needle_demo.mp4",
    gui: bool = False
):
    print("=" * 86)
    print("      SONOSIMLITE END-TO-END NAVIGATION + NEEDLE INSERTION VERIFICATION")
    print("=" * 86)
    print(f"  Target Anatomy:    {anatomy.upper()}")
    print(f"  Patient ID:        {patient}")
    print(f"  Seed:              {seed}")
    print(f"  Needle Guide:      {guide_angle}° In-Plane")
    print(f"  Cannula Length:    {needle_len} mm")
    print(f"  PyBullet Mode:     {'GUI (Live 3D Interactive Window)' if gui else 'DIRECT (Headless)'}")
    print(f"  Video Output:      {video_output_path}")
    print("=" * 86)

    ckpt_path = PROJECT_DIR / f"ppo_checkpoints/{anatomy}_mb/ppo_mb_{anatomy}_final_model.zip"
    if not ckpt_path.exists():
        ckpt_path = PROJECT_DIR / f"aorta_checkpoints_ppo/ppo_final_model.zip"

    print(f"[+] Loading Policy Checkpoint: {ckpt_path.name}")
    model = PPO.load(str(ckpt_path), device="cpu")

    # Initialize Environment
    subj_dir = PROJECT_DIR / "totalseg_patients" / patient
    env = RoboticUltrasoundGymEnv(
        subject_dir=str(subj_dir),
        task="navigation",
        anatomy=anatomy,
        sim_mode="conv",
        size=256,
        pixel_spacing=1.0,
        max_episode_steps=100,
        early_termination=False,
        goal_conditioned=True,
        device="cpu",
        render_mode="human" if gui else "rgb_array"
    )

    # Preload bone label volume for needle collision
    bone_label_path = subj_dir / "bone_label.nii.gz"
    bone_img = nib.load(str(bone_label_path))
    bone_data = bone_img.get_fdata()
    bone_shape = bone_data.shape

    reg_meta = load_registration_meta(subj_dir / "registration_meta.json")
    inv_affine = reg_meta["inv_affine"]
    mesh_offset = reg_meta["mesh_centering_offset"]

    # Video setup
    video_writer = None
    if video_output_path:
        rec_path = Path(video_output_path)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(rec_path), fourcc, 20, (1320, 460))

    # Reset environment with specified seed
    obs, info = env.reset(seed=seed)
    body_pos = env.reg_body_position
    body_rot = env.reg_body_orientation_matrix
    target_world_pos = env.target_internal_world_pos

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=0.85,
            cameraYaw=50,
            cameraPitch=-25,
            cameraTargetPosition=[target_world_pos[0], target_world_pos[1], target_world_pos[2] + 0.10]
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

    initial_pos_err = float(info.get("pos_error_mm", 50.0))
    initial_rot_err = float(info.get("rot_error_deg", 25.0))
    print(f"\n--- PHASE 1: PROBE NAVIGATION ---")
    print(f"  Target Internal Center: {target_world_pos}")
    print(f"  Target Depth d:        {env.target_depth_mm:.1f} mm")
    print(f"  Initial State @ Step 0: Pos Err = {initial_pos_err:.2f} mm, Rot Err = {initial_rot_err:.2f} deg")

    nav_succeeded = False
    first_succ_pose = None
    step_num = 0


    # ----------------------------------------------------
    # PHASE 1: ROBOTIC NAVIGATION
    # ----------------------------------------------------
    for step in range(1, 40):
        step_num = step
        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)

        pos_err_mm = float(info.get("pos_error_mm", 0.0))
        rot_err_deg = float(info.get("rot_error_deg", 0.0))
        bmode_slice = obs.get("image", np.zeros((128, 128), dtype=np.uint8))

        # Check navigation success threshold
        if pos_err_mm < 5.0 and rot_err_deg < 10.0 and not nav_succeeded:
            nav_succeeded = True
            first_succ_pose = {
                "x": float(env.current_surface_x),
                "y": float(env.current_surface_y),
                "quat": env.current_probe_quat.tolist(),
                "step": step,
                "pos_err": pos_err_mm,
                "rot_err": rot_err_deg
            }
            print(f"  [SUCCESS] Navigation Target Reached at Step {step}!")
            print(f"            Pos Error: {pos_err_mm:.2f} mm (< 5.0 mm), Tilt Error: {rot_err_deg:.2f}° (< 10.0°)")

        # Render 3D PyBullet Visuals
        p.removeAllUserDebugItems()
        p.addUserDebugText(f"TARGET LANDMARK", [target_world_pos[0], target_world_pos[1], target_world_pos[2] + 0.03], textColorRGB=[0, 1, 0], textSize=1.2)
        d_c = 0.015
        p.addUserDebugLine([target_world_pos[0]-d_c, target_world_pos[1], target_world_pos[2]], [target_world_pos[0]+d_c, target_world_pos[1], target_world_pos[2]], [0, 1, 0], lineWidth=2)
        p.addUserDebugLine([target_world_pos[0], target_world_pos[1]-d_c, target_world_pos[2]], [target_world_pos[0], target_world_pos[1]+d_c, target_world_pos[2]], [0, 1, 0], lineWidth=2)

        # PyBullet camera capture
        cam_view = p.computeViewMatrixFromYawPitchRoll([target_world_pos[0], target_world_pos[1], target_world_pos[2]], 0.55, 55, -25, 0, 2)
        cam_proj = p.computeProjectionMatrixFOV(60, 480.0 / 460.0, 0.05, 5.0)
        pyb_img = p.getCameraImage(480, 460, cam_view, cam_proj)
        pyb_bgr = cv2.cvtColor(np.reshape(pyb_img[2], (460, 480, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        cv2.rectangle(pyb_bgr, (0, 0), (480, 36), (15, 20, 25), -1)
        cv2.putText(pyb_bgr, "3D PYBULLET: PROBE SCANNING", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

        # Generate HUD
        hud = draw_end_to_end_hud(
            bmode_img=bmode_slice,
            target_name=anatomy,
            patient=patient,
            phase="NAVIGATION",
            step_num=step,
            pos_err_mm=pos_err_mm,
            rot_err_deg=rot_err_deg,
            guide_angle=guide_angle,
            needle_len=needle_len,
            curr_stroke=0.0,
            req_stroke=0.0,
            bone_hit=False,
            target_hit=False,
            tip_dist_mm=pos_err_mm
        )

        frame = np.hstack([pyb_bgr, hud])
        if video_writer is not None:
            video_writer.write(frame)

        if gui:
            time.sleep(0.08)

        if nav_succeeded:
            # Linger frames on successful navigation pose
            for _ in range(8):
                if video_writer is not None:
                    video_writer.write(frame)
                if gui:
                    time.sleep(0.05)
            break

    # ----------------------------------------------------
    # PHASE 2: NEEDLE INSERTION GUIDANCE
    # ----------------------------------------------------
    print(f"\n--- PHASE 2: NEEDLE INSERTION GUIDANCE ---")
    final_x = first_succ_pose["x"]
    final_y = first_succ_pose["y"]
    final_quat = first_succ_pose["quat"]

    skin_z, normal, _ = env.surface_planner.query(final_x, final_y)
    depth_m = abs(skin_z - target_world_pos[2])
    depth_mm = depth_m * 1000.0

    ang_rad = math.radians(guide_angle)
    req_stroke_mm = depth_mm / math.sin(ang_rad)
    lat_offset_m = depth_m / math.tan(ang_rad)

    # In-plane needle entry and direction applying the policy's final beam tilt
    R_probe = np.array(p.getMatrixFromQuaternion(final_quat)).reshape(3, 3)
    probe_beam = -R_probe[:, 2]
    nominal_beam = -env.target_world_normal
    R_tilt = compute_tilt_rotation_matrix(nominal_beam, probe_beam)

    p_probe = np.array([final_x, final_y, skin_z], dtype=np.float64)
    p_probe_target = np.array([target_world_pos[0], target_world_pos[1], skin_z], dtype=np.float64)
    p_entry_nom = p_probe_target + np.array([lat_offset_m, 0.0, 0.0], dtype=np.float64)
    u_nom = (target_world_pos - p_entry_nom) / np.linalg.norm(target_world_pos - p_entry_nom)

    p_entry = np.array([final_x + lat_offset_m, final_y, skin_z], dtype=np.float64)
    u_needle = R_tilt @ u_nom
    u_needle /= np.linalg.norm(u_needle)

    print(f"  Target Center Depth: {depth_mm:.1f} mm")
    print(f"  Required Stroke:     {req_stroke_mm:.1f} mm (Cannula Max: {needle_len:.1f} mm)")

    curr_stroke = 0.0
    needle_steps = int(math.ceil(req_stroke_mm / 3.0))

    target_hit = False
    bone_collision = False
    final_tip_dist = 999.0

    for i in range(1, needle_steps + 15):
        curr_stroke = min(i * 3.0, req_stroke_mm if req_stroke_mm <= needle_len else needle_len)
        p_tip = p_entry + (curr_stroke / 1000.0) * u_needle
        final_tip_dist = float(np.linalg.norm(p_tip - target_world_pos) * 1000.0)

        # Check bone collision along trajectory
        dist_m = curr_stroke / 1000.0
        n_steps = max(1, int(math.ceil(dist_m / 0.0005)))
        bone_hit = False
        for k in range(n_steps + 1):
            pt = p_entry + (k * 0.0005) * u_needle
            vox = compute_registered_ct_center(pt, body_pos, body_rot, inv_affine, mesh_offset, bone_shape, 1.0)
            ix, iy, iz = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
            if 0 <= ix < bone_shape[0] and 0 <= iy < bone_shape[1] and 0 <= iz < bone_shape[2]:
                if bone_data[ix, iy, iz] > 0:
                    bone_hit = True
                    break

        if bone_hit:
            bone_collision = True

        if final_tip_dist <= 5.0 and not bone_collision:
            target_hit = True

        # Render PyBullet 3D scene
        p.removeAllUserDebugItems()
        p.addUserDebugText(f"TARGET LANDMARK", [target_world_pos[0], target_world_pos[1], target_world_pos[2] + 0.03], textColorRGB=[0, 1, 0], textSize=1.2)
        # Guide rail (cyan)
        rail_start = p_entry - 0.06 * u_needle
        p.addUserDebugLine(rail_start.tolist(), p_entry.tolist(), [0, 1, 1], lineWidth=4)
        # Cannula (yellow / green if hit)
        cann_col = [0, 1, 0] if target_hit else ([1, 0, 0] if bone_hit else [1, 0.9, 0.1])
        p.addUserDebugLine(p_entry.tolist(), p_tip.tolist(), cann_col, lineWidth=6)

        cam_view = p.computeViewMatrixFromYawPitchRoll([target_world_pos[0], target_world_pos[1], target_world_pos[2]], 0.50, 55, -25, 0, 2)
        cam_proj = p.computeProjectionMatrixFOV(60, 480.0 / 460.0, 0.05, 5.0)
        pyb_img = p.getCameraImage(480, 460, cam_view, cam_proj)
        pyb_bgr = cv2.cvtColor(np.reshape(pyb_img[2], (460, 480, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        cv2.rectangle(pyb_bgr, (0, 0), (480, 36), (15, 20, 25), -1)
        cv2.putText(pyb_bgr, "3D PYBULLET: NEEDLE CANNULA ADVANCE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)

        hud = draw_end_to_end_hud(
            bmode_img=bmode_slice,
            target_name=anatomy,
            patient=patient,
            phase="NEEDLE_INSERTION",
            step_num=step_num,
            pos_err_mm=first_succ_pose["pos_err"],
            rot_err_deg=first_succ_pose["rot_err"],
            guide_angle=guide_angle,
            needle_len=needle_len,
            curr_stroke=curr_stroke,
            req_stroke=req_stroke_mm,
            bone_hit=bone_collision,
            target_hit=target_hit,
            tip_dist_mm=final_tip_dist
        )

        frame = np.hstack([pyb_bgr, hud])
        if video_writer is not None:
            video_writer.write(frame)

        if gui:
            time.sleep(0.05)

        if target_hit:
            print(f"  🎯 [SUCCESS] Cannula reached target center within {final_tip_dist:.2f} mm! Stroke: {curr_stroke:.1f} mm.")
            # Record final 20 linger frames
            for _ in range(20):
                if video_writer is not None:
                    video_writer.write(frame)
                if gui:
                    time.sleep(0.04)
            break

    if video_writer is not None:
        video_writer.release()

    print("\n" + "=" * 86)
    print("                      VERIFICATION SUMMARY REPORT")
    print("=" * 86)
    print(f"  1. Probe Initial Position Error:  {initial_pos_err:.2f} mm")
    print(f"  2. Navigation Convergence Step:    Step {first_succ_pose['step']}")
    print(f"  3. Final Probe Position Error:    {first_succ_pose['pos_err']:.2f} mm  (< 5.0 mm PASS)")
    print(f"  4. Final Probe Tilt Error:        {first_succ_pose['rot_err']:.2f}°  (< 10.0° PASS)")
    print(f"  5. Required Needle Stroke:        {req_stroke_mm:.1f} mm (Cannula: {needle_len:.1f} mm)")
    print(f"  6. Bone Path Clearance:           {'PASS (0 bone voxels)' if not bone_collision else 'BLOCKED'}")
    print(f"  7. Final Needle Tip Distance:     {final_tip_dist:.2f} mm  (Threshold ≤ 5.0 mm)")
    print(f"  8. Final Needle Guidance Result:  {'🎯 TARGET HIT VERIFIED (PASS)' if target_hit else 'FAIL'}")
    if video_output_path:
        print(f"  9. Video Recording Saved To:      {video_output_path}")
    print("=" * 86)

    if gui:
        print("\n[LIVE 3D VIEW] Verification finished! You can interactively inspect the 3D PyBullet scene.")
        print("                (Press Ctrl+C in this terminal window when done to close)")
        try:
            while True:
                p.stepSimulation()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nClosing interactive session.")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Navigation + Needle Verification")
    parser.add_argument("--anatomy", type=str, default="aorta", choices=["aorta", "liver"])
    parser.add_argument("--patient", type=str, default="s0011")
    parser.add_argument("--seed", type=int, default=1014)
    parser.add_argument("--angle", type=int, default=60)
    parser.add_argument("--length", type=float, default=200.0)
    parser.add_argument("--output", type=str, default="results/videos/end_to_end_nav_needle_demo.mp4")
    parser.add_argument("--gui", action="store_true", help="Launch live interactive 3D PyBullet window")
    parser.add_argument("--live", action="store_true", help="Alias for --gui")
    args = parser.parse_args()

    run_end_to_end_verification(
        anatomy=args.anatomy,
        patient=args.patient,
        seed=args.seed,
        guide_angle=args.angle,
        needle_len=args.length,
        video_output_path=args.output,
        gui=(args.gui or args.live)
    )

