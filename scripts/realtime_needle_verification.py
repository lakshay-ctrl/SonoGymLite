#!/usr/bin/env python3
"""
scripts/realtime_needle_verification.py
=======================================
Real-Time Interactive PyBullet Needle Insertion Verification Suite (SonoSimLite Step 5).
Launches:
  1. Interactive 3D PyBullet GUI showing:
     - 3D Patient skin and internal anatomical target landmark.
     - Ultrasound transducer placed on the skin.
     - Needle guide assembly mounted at selectable in-plane angles (30°, 45°, 60°).
     - Physical needle cannula dynamically advancing through soft tissue towards the target.
     - Real-time bone collision detection raycasting through CT bone label volume.
  2. Synchronized Live Ultrasound Monitor (OpenCV GUI) displaying:
     - Simulated B-mode ultrasound slice with specular needle reflection in the beam plane.
     - Live clinical telemetry HUD (stroke, tip depth, bone clearance, target hit verification).

Controls:
  - Spacebar: Play / Pause insertion animation
  - [UP] / [DOWN] arrow keys: Manually advance / retract needle cannula (mm by mm)
  - [A]: Cycle guide angle (30° -> 45° -> 60°)
  - [L]: Cycle cannula length (100 mm -> 150 mm -> 200 mm)
  - [T]: Toggle Target Anatomy (Aorta Bifurcation <-> Liver Portal Vein)
  - [P]: Cycle Patient Subject (s0058 -> s0011 -> s0344)
  - [R]: Reset cannula to retracted skin entry
  - [Q] / [ESC]: Exit verification

Usage:
  # Verify Aorta needle insertion in real-time 3D PyBullet:
  python3 scripts/realtime_needle_verification.py --anatomy aorta --angle 45

  # Verify Liver needle insertion with video recording:
  python3 scripts/realtime_needle_verification.py --anatomy liver --angle 45 --record-video results/videos/needle_liver_demo.mp4
"""

import sys
import os
import time
import math
import argparse
import json
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib
import pybullet as p
import pybullet_data

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import live_unet_demo as demo
from registration import compute_registered_ct_center, load_registration_meta, voxel_to_world
from surface_motion_planner import load_or_create_surface_map
from robotic_us_env import snap_panda_to_pose





PATIENTS = ["s0058", "s0011", "s0250", "s0344", "s0358"]
ANGLES = [30, 45, 60]
LENGTHS = [100.0, 150.0, 200.0]

def draw_needle_hud(bmode_img, target_name, patient, angle, length, stroke, req_stroke, depth, bone_hit, hit, dist_tip):
    """Draws a clinical telemetry HUD for the needle insertion procedure."""
    if bmode_img.ndim == 3:
        bmode_img = bmode_img[0]
    if bmode_img.dtype != np.uint8:
        if bmode_img.max() <= 1.0:
            bmode_img = (bmode_img * 255.0).astype(np.uint8)
        else:
            bmode_img = np.clip(bmode_img, 0, 255).astype(np.uint8)

    # Scale up B-mode image (400x400)
    us_display = cv2.resize(bmode_img, (400, 400), interpolation=cv2.INTER_CUBIC)
    us_bgr = cv2.cvtColor(us_display, cv2.COLOR_GRAY2BGR)

    # Add simulated specular needle overlay to ultrasound view if needle is entering beam
    ang_rad = math.radians(angle)
    # Visible needle length inside ultrasound plane
    vis_factor = max(0.1, math.cos(ang_rad))
    if stroke > 0:
        frac = min(stroke / max(req_stroke, 1e-3), 1.0)
        # Entry point on top-left of ultrasound screen
        start_pt = (50, 40)
        end_pt = (int(50 + frac * 280 * math.cos(ang_rad)), int(40 + frac * 320 * math.sin(ang_rad)))
        # Hyperechoic needle shaft line
        intensity = int(180 + 75 * vis_factor)
        cv2.line(us_bgr, start_pt, end_pt, (intensity, intensity, intensity), 3, cv2.LINE_AA)
        # Needle tip echo
        cv2.circle(us_bgr, end_pt, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # Create HUD Canvas (Width: 840, Height: 460)
    canvas = np.zeros((460, 840, 3), dtype=np.uint8)
    canvas[:] = (22, 26, 32)  # Dark medical UI

    # Insert Ultrasound slice
    canvas[30:430, 20:420] = us_bgr
    cv2.rectangle(canvas, (18, 28), (422, 432), (80, 90, 105), 2)
    cv2.putText(canvas, f"IN-PLANE B-MODE (THETA = {angle} deg)", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Acoustic Visibility: {vis_factor:.2f} ({20*math.log10(vis_factor):.1f} dB)", (25, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    # Right Telemetry Panel
    tx = 440
    cv2.putText(canvas, "NEEDLE GUIDANCE VERIFICATION", (tx, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (tx, 55), (820, 55), (60, 70, 85), 2)

    # Metadata
    cv2.putText(canvas, f"Target Site:    {target_name.upper()}", (tx, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Patient ID:     {patient}", (tx, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Guide Angle:    {angle} deg  [Key: A]", (tx, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Cannula Length: {int(length)} mm  [Key: L]", (tx, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)

    cv2.line(canvas, (tx, 175), (820, 175), (60, 70, 85), 1)

    # Insertion Telemetry
    cv2.putText(canvas, "Target Center Depth:", (tx, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{depth:.1f} mm", (tx + 220, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(canvas, "Cannula Stroke s:", (tx, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{stroke:.1f} / {req_stroke:.1f} mm", (tx + 200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    # Bone Collision Alert
    cv2.putText(canvas, "Bone Collision Status:", (tx, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1, cv2.LINE_AA)
    if bone_hit:
        cv2.putText(canvas, "BONE HIT! (OBSTRUCTED)", (tx + 200, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "BONE-FREE PATH", (tx + 200, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2, cv2.LINE_AA)

    # Tip Distance to Target
    cv2.putText(canvas, "Tip-to-Target Dist:", (tx, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1, cv2.LINE_AA)
    dist_color = (0, 255, 0) if dist_tip < 5.0 else ((0, 200, 255) if dist_tip < 20.0 else (100, 100, 255))
    cv2.putText(canvas, f"{dist_tip:.2f} mm", (tx + 220, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.65, dist_color, 2, cv2.LINE_AA)

    cv2.line(canvas, (tx, 335), (820, 335), (60, 70, 85), 1)

    # Big Status Banner
    if hit:
        cv2.rectangle(canvas, (tx, 350), (820, 410), (20, 130, 40), -1)
        cv2.rectangle(canvas, (tx, 350), (820, 410), (0, 255, 100), 2)
        cv2.putText(canvas, " TARGET HIT VERIFIED (SUCCESS) ", (tx + 12, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    elif bone_hit:
        cv2.rectangle(canvas, (tx, 350), (820, 410), (20, 20, 130), -1)
        cv2.rectangle(canvas, (tx, 350), (820, 410), (0, 0, 255), 2)
        cv2.putText(canvas, " PATH BLOCKED BY BONE CORTEX ", (tx + 18, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    elif stroke >= length and not hit:
        cv2.rectangle(canvas, (tx, 350), (820, 410), (130, 80, 20), -1)
        cv2.rectangle(canvas, (tx, 350), (820, 410), (0, 180, 255), 2)
        cv2.putText(canvas, " NEEDLE LENGTH EXHAUSTED ", (tx + 28, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (tx, 350), (820, 410), (35, 40, 50), -1)
        cv2.rectangle(canvas, (tx, 350), (820, 410), (80, 90, 105), 1)
        prog = min(max(stroke / max(req_stroke, 1e-3), 0.0), 1.0)
        cv2.rectangle(canvas, (tx + 10, 370), (tx + 10 + int(prog * 340), 390), (255, 170, 30), -1)
        cv2.putText(canvas, f"ADVANCING CANNULA... ({int(prog*100)}%)", (tx + 30, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer
    cv2.putText(canvas, "Keys: [SPACE] Insert/Pause | [UP/DOWN] Manual | [A] Angle | [L] Length | [Q] Quit", (tx - 50, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (130, 140, 150), 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Real-Time PyBullet Needle Insertion Verification Suite")
    parser.add_argument("--anatomy", type=str, default="aorta", choices=["aorta", "liver"], help="Target anatomy site")
    parser.add_argument("--patient", type=str, default="s0058", choices=PATIENTS, help="Patient subject ID")
    parser.add_argument("--angle", type=int, default=45, choices=ANGLES, help="Guide angle in degrees (30, 45, 60)")
    parser.add_argument("--length", type=float, default=200.0, choices=LENGTHS, help="Needle cannula length in mm")
    parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI (DIRECT mode)")
    parser.add_argument("--no-opencv-gui", action="store_true", help="Disable OpenCV popup window")
    parser.add_argument("--record-video", type=str, default=None, help="Save MP4 verification video")
    parser.add_argument("--auto", action="store_true", help="Automatically run insertion to completion and exit without waiting for keypresses")
    parser.add_argument("--auto-exit", action="store_true", help="Alias for --auto")
    parser.add_argument("--split-screen", action="store_true", default=True, help="Render 3D PyBullet camera + Ultrasound HUD side-by-side in video")
    args = parser.parse_args()

    patient = args.patient
    target_anatomy = args.anatomy
    guide_angle = args.angle
    needle_len = args.length

    subj_dir = PROJECT_DIR / "totalseg_patients" / patient
    bone_label_path = subj_dir / "bone_label.nii.gz"
    skin_obj_path = subj_dir / "patient_skin.obj"
    reg_meta_path = subj_dir / "registration_meta.json"

    print("=" * 86)
    print("      SONOGYM REAL-TIME PYBULLET NEEDLE INSERTION & GUIDANCE VERIFICATION")
    print("=" * 86)
    print(f"  Target Anatomy:    {target_anatomy.upper()} ({'Supine Anterior' if target_anatomy == 'aorta' else 'Prone Posterior'})")
    print(f"  Patient Subject:   {patient}")
    print(f"  Guide Angle theta: {guide_angle}° (In-Plane)")
    print(f"  Cannula Length L:  {needle_len} mm")
    print(f"  PyBullet Mode:     {'DIRECT (Headless)' if args.headless else 'GUI (Interactive 3D Window)'}")
    print("=" * 86)

    # 1. Connect to PyBullet
    cid = p.connect(p.DIRECT if args.headless else p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # 2. Spawn Hospital Bed, Patient Skin, Robot, and Probe
    bed_top_z = demo.create_hospital_room()
    orientation = "supine" if target_anatomy == "aorta" else "prone"
    mesh_min, _ = demo.read_obj_bounds(skin_obj_path)

    if orientation == "supine":
        body_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
        body_rot = np.array(p.getMatrixFromQuaternion(body_quat), dtype=np.float64).reshape(3, 3)
        body_z = bed_top_z - mesh_min[1] + 0.01
    else:
        body_quat = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
        body_rot = np.array(p.getMatrixFromQuaternion(body_quat), dtype=np.float64).reshape(3, 3)
        body_z = bed_top_z - mesh_min[1] + 0.01

    body_pos = np.array([float(demo.BED_CENTER[0]), float(demo.BED_CENTER[1]), float(body_z)], dtype=np.float64)

    # Load Skin Visual & Collision
    col_shape = p.createCollisionShape(p.GEOM_MESH, fileName=str(skin_obj_path), meshScale=[1, 1, 1], flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    vis_shape = p.createVisualShape(p.GEOM_MESH, fileName=str(skin_obj_path), meshScale=[1, 1, 1], rgbaColor=[0.9, 0.8, 0.7, 0.85])
    patient_body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape, baseVisualShapeIndex=vis_shape, basePosition=body_pos, baseOrientation=body_quat)

    # Load Surface Motion Planner & Target Coordinate
    surface_planner = load_or_create_surface_map(subj_dir, bed_top_z=bed_top_z, mesh_scale=1.0, orientation=orientation)
    reg_meta = load_registration_meta(reg_meta_path)
    inv_affine = reg_meta["inv_affine"]
    mesh_offset = reg_meta["mesh_centering_offset"]

    # Target landmark location
    if target_anatomy == "aorta":
        target_name = "Aorta Bifurcation"
        target_json = PROJECT_DIR / "aorta_bifurcation_targets.json"
        with open(target_json, "r") as f:
            targets_dict = json.load(f)
        target_vox = targets_dict[patient]["voxel_bifurcation_zyx"]
    else:
        target_name = "Portal Vein / Liver Lesion"
        target_json = PROJECT_DIR / "portal_vein_targets.json"
        with open(target_json, "r") as f:
            targets_dict = json.load(f)
        target_vox = targets_dict[patient]["voxel_centroid_zyx"]

    bone_img = nib.load(str(bone_label_path))
    bone_data = bone_img.get_fdata()
    bone_shape = bone_data.shape

    world_target = voxel_to_world(target_vox, body_pos, body_rot, reg_meta["affine"], mesh_offset, 1.0)

    # Surface entry query
    skin_z, normal, _ = surface_planner.query(float(world_target[0]), float(world_target[1]))
    depth_m = abs(skin_z - world_target[2])
    depth_mm = depth_m * 1000.0

    ang_rad = math.radians(guide_angle)
    req_stroke_mm = depth_mm / math.sin(ang_rad)
    lat_offset_m = depth_m / math.tan(ang_rad)

    # Entry point & needle unit direction
    p_probe = np.array([world_target[0], world_target[1], skin_z], dtype=np.float64)
    p_entry = p_probe + np.array([lat_offset_m, 0.0, 0.0], dtype=np.float64)
    u_needle = (world_target - p_entry) / np.linalg.norm(world_target - p_entry)

    # Spawn Franka Panda Robot & Ultrasound Probe
    panda_id = demo.create_panda_robot()
    probe_body_ids = demo.create_probe_model()
    try:
        (probe_tip_pos, probe_quat, ee_pos, ee_quat, _, _, _) = surface_planner.compute_surface_pose(
            float(world_target[0]), float(world_target[1]), 0.0, indentation=0.002
        )
        demo.update_probe_model(probe_body_ids, probe_tip_pos, probe_quat)
        snap_panda_to_pose(panda_id, ee_pos, ee_quat)
    except Exception as e:
        print(f"[WARN] Robot IK alignment skipped: {e}")

    # 3. Setup Camera & 3D Markers
    if not args.headless:
        p.resetDebugVisualizerCamera(
            cameraDistance=0.75,
            cameraYaw=55,
            cameraPitch=-25,
            cameraTargetPosition=[world_target[0], world_target[1], world_target[2]]
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Draw Target Sphere & Text in PyBullet
        p.addUserDebugText(f"★ {target_name.upper()} TARGET", [world_target[0], world_target[1], world_target[2] + 0.04], textColorRGB=[0, 1, 0], textSize=1.3)
        # Entry point marker
        p.addUserDebugText("ENTRY POINT", [p_entry[0], p_entry[1], p_entry[2] + 0.03], textColorRGB=[1, 1, 0], textSize=1.1)

    print(f"\n[+] Target Center Depth:    {depth_mm:.1f} mm")
    print(f"[+] Required Stroke (s):    {req_stroke_mm:.1f} mm  (Needle Length: {needle_len:.1f} mm -> {'REACHABLE ✅' if req_stroke_mm <= needle_len else 'EXCEEDS CANNULA ⚠️'})")
    print(f"[+] Skin Entry Offset:      {lat_offset_m * 1000.0:.1f} mm lateral")
    print("-" * 86)

    # Prepare dummy ultrasound slice (or extract actual from volume)
    bmode_slice = np.zeros((128, 128), dtype=np.uint8)
    # Add tissue speckle
    np.random.seed(42)
    bmode_slice = (np.clip(np.random.normal(0.25, 0.05, (128, 128)), 0.0, 1.0) * 255.0).astype(np.uint8)
    # Add anechoic target lumen
    cv2.circle(bmode_slice, (80, 75), 18, (15, 15, 15), -1)

    is_auto = args.auto or args.auto_exit or args.headless
    video_writer = None
    if args.record_video:
        rec_path = Path(args.record_video)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_w = (480 + 840) if args.split_screen else 840
        video_writer = cv2.VideoWriter(str(rec_path), fourcc, 25, (vid_w, 460))

    curr_stroke_mm = 0.0
    auto_advance = True
    bone_collision_triggered = False
    final_linger_frames = 0
    done_reason = None

    while True:
        # Check OpenCV keyboard input
        key = -1
        if not args.no_opencv_gui:
            key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            auto_advance = not auto_advance
            print(f"\n[USER] Insertion {'RESUMED' if auto_advance else 'PAUSED'}.")
        elif key == 82 or key == ord('w'):  # UP arrow
            curr_stroke_mm = min(curr_stroke_mm + 2.0, needle_len)
        elif key == 84 or key == ord('s'):  # DOWN arrow
            curr_stroke_mm = max(curr_stroke_mm - 2.0, 0.0)
        elif key == ord('a'):  # Cycle angle
            idx = (ANGLES.index(guide_angle) + 1) % len(ANGLES)
            guide_angle = ANGLES[idx]
            ang_rad = math.radians(guide_angle)
            req_stroke_mm = depth_mm / math.sin(ang_rad)
            lat_offset_m = depth_m / math.tan(ang_rad)
            p_entry = p_probe + np.array([lat_offset_m, 0.0, 0.0], dtype=np.float64)
            u_needle = (world_target - p_entry) / np.linalg.norm(world_target - p_entry)
            curr_stroke_mm = 0.0
            print(f"\n[ANGLE CHANGED] Guide Angle: {guide_angle}° | Req Stroke: {req_stroke_mm:.1f} mm")
        elif key == ord('l'):  # Cycle length
            l_idx = (LENGTHS.index(needle_len) + 1) % len(LENGTHS)
            needle_len = LENGTHS[l_idx]
            print(f"\n[LENGTH CHANGED] Cannula Length: {int(needle_len)} mm")
        elif key == ord('r'):
            curr_stroke_mm = 0.0
            print("\n[RESET] Cannula retracted to skin entry.")

        # Advance cannula if auto
        if auto_advance and curr_stroke_mm < req_stroke_mm and curr_stroke_mm < needle_len:
            curr_stroke_mm += 2.0
            if curr_stroke_mm > req_stroke_mm:
                curr_stroke_mm = req_stroke_mm

        # Current needle tip position
        p_tip = p_entry + (curr_stroke_mm / 1000.0) * u_needle
        dist_tip_mm = float(np.linalg.norm(p_tip - world_target) * 1000.0)
        is_hit = (dist_tip_mm <= 5.0)

        # Check bone collision along trajectory up to current tip
        dist_m = curr_stroke_mm / 1000.0
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

        # Draw 3D elements in PyBullet
        p.removeAllUserDebugItems()
        p.addUserDebugText(f"{target_name.upper()}", [world_target[0], world_target[1], world_target[2] + 0.04], textColorRGB=[0, 1, 0], textSize=1.3)
        # Target crosshair
        d_c = 0.015
        p.addUserDebugLine([world_target[0]-d_c, world_target[1], world_target[2]], [world_target[0]+d_c, world_target[1], world_target[2]], [0, 1, 0], lineWidth=2)
        p.addUserDebugLine([world_target[0], world_target[1]-d_c, world_target[2]], [world_target[0], world_target[1]+d_c, world_target[2]], [0, 1, 0], lineWidth=2)
        p.addUserDebugLine([world_target[0], world_target[1], world_target[2]-d_c], [world_target[0], world_target[1], world_target[2]+d_c], [0, 1, 0], lineWidth=2)

        # Draw needle guide rail (cyan)
        rail_start = p_entry - 0.06 * u_needle
        p.addUserDebugLine(rail_start.tolist(), p_entry.tolist(), [0, 1, 1], lineWidth=4)

        # Draw needle cannula (bright yellow / green if hit / red if bone collision)
        cannula_color = [1, 0, 0] if bone_hit else ([0, 1, 0] if is_hit else [1, 0.9, 0.1])
        p.addUserDebugLine(p_entry.tolist(), p_tip.tolist(), cannula_color, lineWidth=6)

        # Render OpenCV HUD
        hud_frame = draw_needle_hud(bmode_slice, target_name, patient, guide_angle, needle_len, curr_stroke_mm, req_stroke_mm, depth_mm, bone_hit, is_hit, dist_tip_mm)
        
        frame_to_show = hud_frame
        if args.split_screen:
            # Capture 3D view from PyBullet camera
            cam_view = p.computeViewMatrixFromYawPitchRoll([world_target[0], world_target[1], world_target[2]], 0.50, 55, -25, 0, 2)
            cam_proj = p.computeProjectionMatrixFOV(60, 480.0 / 460.0, 0.05, 5.0)
            pyb_img = p.getCameraImage(480, 460, cam_view, cam_proj)
            pyb_rgba = np.reshape(pyb_img[2], (460, 480, 4)).astype(np.uint8)
            pyb_bgr = cv2.cvtColor(pyb_rgba, cv2.COLOR_RGBA2BGR)
            cv2.rectangle(pyb_bgr, (0, 0), (480, 36), (15, 20, 25), -1)
            cv2.putText(pyb_bgr, "3D PYBULLET SIMULATION", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
            frame_to_show = np.hstack([pyb_bgr, hud_frame])

        if not args.no_opencv_gui:
            cv2.imshow("SonoSimLite Needle Guidance Verification", frame_to_show)

        if video_writer is not None:
            video_writer.write(frame_to_show)

        # Check termination condition in automated mode
        if is_hit:
            if done_reason is None:
                done_reason = f"TARGET HIT VERIFIED (tip dist: {dist_tip_mm:.2f} mm <= 5.0 mm, stroke: {curr_stroke_mm:.1f} mm)"
                print(f"\n🎯 [SUCCESS] {done_reason}")
            auto_advance = False
            final_linger_frames += 1
            if is_auto and final_linger_frames >= 25:
                break

        elif bone_hit:
            if done_reason is None:
                done_reason = f"BONE COLLISION TRIGGERED (path blocked at stroke {curr_stroke_mm:.1f} mm)"
                print(f"\n🚫 [OBSTRUCTED] {done_reason}")
            auto_advance = False
            final_linger_frames += 1
            if is_auto and final_linger_frames >= 25:
                break

        elif curr_stroke_mm >= needle_len:
            if done_reason is None:
                done_reason = f"MAX CANNULA LENGTH REACHED ({needle_len} mm < required stroke {req_stroke_mm:.1f} mm)"
                print(f"\n⚠️ [EXHAUSTED] {done_reason}")
            auto_advance = False
            final_linger_frames += 1
            if is_auto and final_linger_frames >= 25:
                break


    if video_writer is not None:
        video_writer.release()
        print(f"[DONE] Saved needle insertion video to: {args.record_video}")

    cv2.destroyAllWindows()
    p.disconnect()
    print("\n[VERIFICATION COMPLETE] Needle insertion verification finished cleanly.\n")


if __name__ == "__main__":
    main()
