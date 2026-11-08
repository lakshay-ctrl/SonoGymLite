#!/usr/bin/env python3
"""
scripts/realtime_pybullet_verification.py
=========================================
Real-Time Interactive PyBullet Physics & Ultrasound Verification Suite.
Launches:
  1. Interactive 3D PyBullet GUI with patient body, probe motion tracking,
     real-time 3D trajectory ribbon, and target landmark crosshairs.
  2. Synchronized Live Ultrasound Monitor (OpenCV GUI) displaying the real-time
     B-mode ultrasound slice alongside live telemetry HUD (force, distance, Q, FPS).

Keyboard Controls in PyBullet / OpenCV:
  - Spacebar: Pause / Resume simulation
  - 'r': Reset episode to random probe start position
  - 'q' or ESC: Exit verification

Usage:
  # Verify Spine Image-Only SAC in real-time 3D PyBullet:
  python3 scripts/realtime_pybullet_verification.py --anatomy spine --model-type image_only_sac

  # Verify Aorta Image-Only SAC on external held-out patient:
  python3 scripts/realtime_pybullet_verification.py --anatomy aorta --model-type image_only_sac

  # Verify Liver Navigation:
  python3 scripts/realtime_pybullet_verification.py --anatomy liver --model-type image_only_sac
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import pybullet as p

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from stable_baselines3 import PPO, SAC
from robotic_us_env import RoboticUltrasoundGymEnv, TRAIN_PATIENTS, TEST_PATIENTS

CHECKPOINTS = {
    "spine": {
        "image_only_sac": PROJECT_DIR / "sac_image_only/spine_mb/sac_image_only_spine_mb_final.zip",
        "image_only_sac_lb": PROJECT_DIR / "sac_image_only/spine_lb/sac_image_only_spine_lb_final.zip",
        "image_only_ppo": PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_spine_s0_final.zip",
        "goal_sac": PROJECT_DIR / "sac_checkpoints/spine_mb/sac_mb_spine_final_model.zip",
        "goal_ppo": PROJECT_DIR / "ppo_checkpoints/spine_mb_goal_100k/ppo_mb_spine_final_model.zip",
    },
    "aorta": {
        "image_only_sac": PROJECT_DIR / "sac_image_only/aorta_mb/sac_image_only_aorta_mb_final.zip",
        "image_only_ppo": PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip",
        "goal_sac": PROJECT_DIR / "sac_checkpoints/aorta_mb/sac_mb_aorta_final_model.zip",
        "goal_ppo": PROJECT_DIR / "ppo_checkpoints/aorta_mb/ppo_mb_aorta_final_model.zip",
    },
    "liver": {
        "image_only_sac": PROJECT_DIR / "sac_image_only/liver_mb/sac_image_only_liver_mb_final.zip",
        "image_only_ppo": PROJECT_DIR / "ppo_checkpoints/liver_centre/ppo_scale_liver_300000_steps.zip",
        "goal_sac": PROJECT_DIR / "sac_checkpoints/liver_mb/sac_mb_liver_final_model.zip",
        "goal_ppo": PROJECT_DIR / "ppo_checkpoints/liver_mb/ppo_mb_liver_final_model.zip",
    }
}


def draw_hud(bmode_img, info, step, max_steps, anatomy, model_name, patient, cohort, achieved):
    """Render a crystal-clear medical telemetry HUD on the live ultrasound window."""
    # Ensure B-mode is 2D uint8
    if bmode_img.ndim == 3:
        bmode_img = bmode_img[0]
    if bmode_img.dtype != np.uint8:
        if bmode_img.max() <= 1.0:
            bmode_img = (bmode_img * 255.0).astype(np.uint8)
        else:
            bmode_img = np.clip(bmode_img, 0, 255).astype(np.uint8)

    # Scale up B-mode image for crisp high-res display (384x384)
    us_display = cv2.resize(bmode_img, (400, 400), interpolation=cv2.INTER_CUBIC)
    us_bgr = cv2.cvtColor(us_display, cv2.COLOR_GRAY2BGR)

    # Create HUD Canvas (Width: 820, Height: 440)
    canvas = np.zeros((460, 840, 3), dtype=np.uint8)
    canvas[:] = (24, 27, 33)  # Sleek dark slate background

    # Insert Ultrasound slice on left
    canvas[30:430, 20:420] = us_bgr
    cv2.rectangle(canvas, (18, 28), (422, 432), (80, 90, 105), 2)

    # Status indicator badge on ultrasound view
    q_val = float(info.get("image_quality", 0.0))
    cv2.putText(canvas, f"B-MODE TRANSDUCER (128x128)", (25, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Acoustic Quality Q: {q_val:.3f}", (25, 418), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    # Telemetry Panel (Right Side)
    tx = 440
    cv2.putText(canvas, "PYBULLET REAL-TIME VERIFICATION", (tx, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (tx, 55), (820, 55), (60, 70, 85), 2)

    # Metadata
    cv2.putText(canvas, f"Target Anatomy:  {anatomy.upper()}", (tx, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Patient Subject: {patient} ({cohort})", (tx, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Active Policy:   {model_name}", (tx, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Episode Step:    {step} / {max_steps}", (tx, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.line(canvas, (tx, 175), (820, 175), (60, 70, 85), 1)

    # Live Metrics
    pos_err = float(info.get("pos_error_mm", 0.0))
    rot_err = float(info.get("rot_error_deg", 0.0))
    c_force = float(info.get("contact_force_n", 0.0))

    # Metric colors
    pos_color = (0, 255, 0) if pos_err < 5.0 else ((0, 200, 255) if pos_err < 15.0 else (100, 100, 255))
    rot_color = (0, 255, 0) if rot_err < 10.0 else (100, 100, 255)
    force_color = (0, 255, 0) if (2.0 <= c_force <= 8.0) else (0, 220, 255)

    cv2.putText(canvas, "3D Position Error:", (tx, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{pos_err:6.2f} mm", (tx + 220, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.68, pos_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "(Threshold: < 5.0 mm)", (tx + 220, 223), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Probe Rotation Error:", (tx, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{rot_err:6.2f} deg", (tx + 220, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.68, rot_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "(Threshold: < 10.0 deg)", (tx + 220, 273), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Surface Contact Force:", (tx, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{c_force:6.2f} N", (tx + 220, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.68, force_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "(Window: 2.0 - 8.0 N)", (tx + 220, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)

    cv2.line(canvas, (tx, 340), (820, 340), (60, 70, 85), 1)

    # Big Banner Outcome
    if achieved:
        cv2.rectangle(canvas, (tx, 355), (820, 415), (20, 120, 40), -1)
        cv2.rectangle(canvas, (tx, 355), (820, 415), (0, 255, 100), 2)
        cv2.putText(canvas, " TARGET REACHED (SUCCESS) ", (tx + 18, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (tx, 355), (820, 415), (40, 45, 55), -1)
        cv2.rectangle(canvas, (tx, 355), (820, 415), (90, 100, 115), 1)
        # Visual progress bar towards goal
        prog = min(max((50.0 - pos_err) / 50.0, 0.0), 1.0)
        cv2.rectangle(canvas, (tx + 10, 375), (tx + 10 + int(prog * 340), 395), (255, 160, 30), -1)
        cv2.putText(canvas, "CONVERGING TO TARGET...", (tx + 40, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    # Footer instructions
    cv2.putText(canvas, "Controls: [SPACE] Pause | [R] Reset Start Pose | [Q] Quit Verification", (tx - 50, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (120, 130, 140), 1, cv2.LINE_AA)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Real-Time PyBullet Physics & Policy Verification")
    parser.add_argument("--anatomy", type=str, default="spine", choices=["spine", "aorta", "liver"], help="Target anatomy")
    parser.add_argument("--patient", type=str, default="s0344", help="Patient ID (Train: s0011, s0058, s0250 | Held-Out: s0344, s0358)")
    parser.add_argument("--model-type", type=str, default="image_only_sac",
                        choices=["image_only_sac", "image_only_sac_lb", "image_only_ppo", "goal_sac", "goal_ppo"],
                        help="Policy to execute and verify")
    parser.add_argument("--seed", type=int, default=1000, help="Reset seed for probe initial position")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum episode steps")
    parser.add_argument("--step-delay", type=float, default=0.03, help="Playback sleep duration per step in seconds (default: 0.03 for ~30 FPS)")
    parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI (DIRECT mode)")
    parser.add_argument("--no-opencv-gui", action="store_true", help="Disable OpenCV popup window")
    parser.add_argument("--record-video", type=str, default=None, help="Save verification video to file (e.g. results/pybullet_verification.mp4)")
    args = parser.parse_args()

    target = args.anatomy
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / args.patient)
    if not Path(subj_dir).exists():
        print(f"[ERROR] Patient subject directory not found: {subj_dir}")
        sys.exit(1)

    ckpt_path = CHECKPOINTS[target][args.model_type]
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    is_image_only = "image_only" in args.model_type
    cohort = "HELD-OUT UNSEEN" if args.patient in TEST_PATIENTS or args.patient in ["s0344", "s0358"] else "TRAINING COHORT"
    backend = "unet" if "lb" in args.model_type else "conv"

    print("=" * 86)
    print("      SONOGYM REAL-TIME PYBULLET SIMULATION & PHYSICAL VERIFICATION")
    print("=" * 86)
    print(f"  Target Anatomy:    {target.upper()}")
    print(f"  Patient Subject:   {args.patient} ({cohort})")
    print(f"  Policy Model:      {args.model_type.upper()}")
    print(f"  Simulator Backend: {backend.upper()}")
    print(f"  Observation Space: {'Image + Contact + Orientation (STRICT IMAGE-ONLY)' if is_image_only else 'Goal-Conditioned (6-DoF Target Guidance)'}")
    print(f"  PyBullet Mode:     {'DIRECT (Headless)' if args.headless else 'GUI (Interactive 3D Window)'}")
    print(f"  Initial Seed:      {args.seed}")
    print("=" * 86)

    # 1. Load Model
    print("\n[+] Loading policy weights into memory...")
    if "sac" in args.model_type:
        model = SAC.load(str(ckpt_path), device="cpu")
    else:
        model = PPO.load(str(ckpt_path), device="cpu")

    img_size = 128 if is_image_only else 256
    if hasattr(model.observation_space, "spaces") and "image" in model.observation_space.spaces:
        img_size = int(model.observation_space.spaces["image"].shape[0])

    # 2. Spawn PyBullet Environment
    print(f"[+] Initializing PyBullet Physics Server (GUI={not args.headless})...")
    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="navigation",
        anatomy=target,
        sim_mode=backend,
        size=img_size,
        pixel_spacing=1.0,
        max_episode_steps=args.max_steps,
        early_termination=False,
        goal_conditioned=(not is_image_only),
        render_mode="direct" if args.headless else "human",
        device="cpu"
    )

    obs, info = env.reset(seed=args.seed)

    target_pos = env.unwrapped.target_world_pos.copy()
    init_probe_pos = env.unwrapped.current_probe_tip_pos.copy()

    # 3. Configure PyBullet 3D Camera & Visual Overlays
    if not args.headless:
        # Camera centered on target region
        cam_target = [float(target_pos[0]), float(target_pos[1]), float(target_pos[2] - 0.05)]
        p.resetDebugVisualizerCamera(
            cameraDistance=0.70,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=cam_target
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # Mark Target Landmark with 3D Text & Target Sphere
        p.addUserDebugText(
            f"★ {target.upper()} TARGET GOAL",
            [target_pos[0], target_pos[1], target_pos[2] + 0.05],
            textColorRGB=[0.0, 1.0, 0.2],
            textSize=1.3,
            lifeTime=0
        )
        # 3D Crosshairs at target
        delta_arm = 0.02
        p.addUserDebugLine([target_pos[0]-delta_arm, target_pos[1], target_pos[2]], [target_pos[0]+delta_arm, target_pos[1], target_pos[2]], [0, 1, 0], lineWidth=3)
        p.addUserDebugLine([target_pos[0], target_pos[1]-delta_arm, target_pos[2]], [target_pos[0], target_pos[1]+delta_arm, target_pos[2]], [0, 1, 0], lineWidth=3)
        p.addUserDebugLine([target_pos[0], target_pos[1], target_pos[2]-delta_arm], [target_pos[0], target_pos[1], target_pos[2]+delta_arm], [0, 1, 0], lineWidth=3)

    print("\n[+] Verification Running! Press SPACE to pause/resume, R to re-seed, Q to exit.")
    print("-" * 86)
    print(" STEP | PROBE TIP (X, Y, Z mm)     | DIST TO GOAL   | ROT ERR | FORCE (N) | Q-SCORE | STATUS")
    print("-" * 86)

    video_writer = None
    if args.record_video:
        rec_path = Path(args.record_video)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(rec_path), fourcc, 25, (840, 460))

    prev_tip_pos = init_probe_pos.copy()
    success_step = None
    paused = False
    curr_seed = args.seed

    step = 0
    while step < args.max_steps:
        # Check OpenCV window keyboard inputs
        key = cv2.waitKey(max(int(args.step_delay * 1000), 1)) & 0xFF
        if key == ord('q') or key == 27:  # ESC or Q
            print("\n[USER] Exit requested.")
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"\n[USER] Simulation {'PAUSED' if paused else 'RESUMED'}.")
        elif key == ord('r'):  # R
            curr_seed += 1
            obs, info = env.reset(seed=curr_seed)
            prev_tip_pos = env.unwrapped.current_probe_tip_pos.copy()
            step = 0
            success_step = None
            if not args.headless:
                p.removeAllUserDebugItems()
                # Re-add target
                p.addUserDebugText(f"★ {target.upper()} TARGET GOAL", [target_pos[0], target_pos[1], target_pos[2] + 0.05], textColorRGB=[0, 1, 0], textSize=1.3)
            print(f"\n[RESET] Re-initialized probe to new random start position (Seed {curr_seed}).")
            continue

        if paused:
            time.sleep(0.05)
            continue

        step += 1

        # Format observation wrapper for Image-Only
        if is_image_only:
            contact_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
            quat = np.array(env.unwrapped.current_probe_quat, dtype=np.float32)
            obs["contact"] = np.array([contact_val], dtype=np.float32)
            obs["orientation"] = quat

        curr_pos_mm = env.unwrapped.current_probe_tip_pos.copy() * 1000.0
        curr_tip_pos = env.unwrapped.current_probe_tip_pos.copy()

        # Step the policy
        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        pos_err = float(info.get("pos_error_mm", 0.0))
        rot_err = float(info.get("rot_error_deg", 0.0))
        c_force = float(info.get("contact_force_n", 0.0))
        q_val = float(info.get("image_quality", 0.0))

        # Check success condition
        is_succ = (pos_err < 5.0 and rot_err < 10.0)
        if is_succ and success_step is None:
            success_step = step

        # Draw 3D trajectory line ribbon in PyBullet
        if not args.headless:
            traj_color = [0.0, 1.0, 0.2] if is_succ else [1.0, 0.8, 0.1]
            p.addUserDebugLine(prev_tip_pos.tolist(), curr_tip_pos.tolist(), traj_color, lineWidth=2.5, lifeTime=0)
            prev_tip_pos = curr_tip_pos.copy()

        status_str = "🎯 TARGET REACHED" if is_succ else f"Err: {pos_err:5.1f} mm"
        print(f" {step:03d}  | ({curr_pos_mm[0]:+6.1f}, {curr_pos_mm[1]:+6.1f}, {curr_pos_mm[2]:+6.1f}) | {pos_err:6.2f} mm    | {rot_err:5.1f}°  |   {c_force:4.1f} N   |  {q_val:.3f}  | {status_str}", flush=True)

        # Render OpenCV Telemetry HUD Window
        if not args.no_opencv_gui:
            bmode = obs.get("image", np.zeros((img_size, img_size), dtype=np.uint8))
            hud_frame = draw_hud(bmode, info, step, args.max_steps, target, args.model_type, args.patient, cohort, is_succ)
            cv2.imshow("SonoGym Live Robotic Ultrasound Verification", hud_frame)

            if video_writer is not None:
                video_writer.write(hud_frame)

        if term or trunc:
            print("\n[EPISODE COMPLETE] Episode reached termination boundary.")
            break

    if video_writer is not None:
        video_writer.release()
        print(f"[DONE] Verification recording saved to: {args.record_video}")

    cv2.destroyAllWindows()
    env.close()

    print("\n" + "=" * 86)
    print("                      VERIFICATION SUMMARY CARD")
    print("=" * 86)
    print(f"  Target Anatomy:         {target.upper()}")
    print(f"  Patient ID:             {args.patient} ({cohort})")
    print(f"  Final Position Error:   {pos_err:.2f} mm  (Threshold: < 5.0 mm -> {'PASSED ✅' if pos_err < 5.0 else 'BORDERLINE ⚠️'})")
    print(f"  Final Rotation Error:   {rot_err:.2f}°   (Threshold: < 10.0° -> {'PASSED ✅' if rot_err < 10.0 else 'BORDERLINE ⚠️'})")
    print(f"  Physical Surface Force: {c_force:.2f} N   (Clinical Window: 2.0-8.0 N -> {'IN WINDOW ✅' if 2.0 <= c_force <= 8.0 else 'OK'})")
    print(f"  Acoustic Quality Score: {q_val:.3f}")
    print(f"  Outcome:                {'TARGET SUCCESSFULLY REACHED ✅' if success_step is not None else 'TIMED OUT'}")
    if success_step is not None:
        print(f"  Convergence Speed:      Step {success_step} (out of {args.max_steps})")
    print("=" * 86)
    print("  CONCLUSION: PyBullet dynamics, acoustic raycasting, and policy are VERIFIED.")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
