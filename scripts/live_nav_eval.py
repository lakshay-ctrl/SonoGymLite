#!/usr/bin/env python3
"""
scripts/live_nav_eval.py
========================
Interactive Live Navigation Evaluation for Autonomous Robotic Ultrasound.
Allows the user to evaluate and watch image-only and goal-conditioned policies step-by-step.

Usage:
  # 1. Live terminal evaluation of Spine Image-Only SAC on external held-out patient s0344:
  python3 scripts/live_nav_eval.py --anatomy spine --patient s0344 --model-type image_only_sac --seed 1000

  # 2. Live evaluation with side-by-side video recording (B-mode slice + distance curve):
  python3 scripts/live_nav_eval.py --anatomy spine --patient s0344 --model-type image_only_sac --seed 1000 --render-video results/videos/spine_sac_s0344.mp4
"""

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

import torch as th
from stable_baselines3 import PPO, SAC
from robotic_us_env import RoboticUltrasoundGymEnv, TRAIN_PATIENTS, TEST_PATIENTS

CHECKPOINTS = {
    "spine": {
        "image_only_ppo": {
            0: PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_spine_s0_final.zip"
        },
        "image_only_sac": PROJECT_DIR / "sac_image_only/spine_mb/sac_image_only_spine_mb_final.zip",
        "image_only_sac_lb": PROJECT_DIR / "sac_image_only/spine_lb/sac_image_only_spine_lb_final.zip",
        "goal_ppo": PROJECT_DIR / "ppo_checkpoints/spine_mb_goal_100k/ppo_mb_spine_final_model.zip",
        "goal_sac": PROJECT_DIR / "sac_checkpoints/spine_mb/sac_mb_spine_final_model.zip"
    },
    "aorta": {
        "image_only_ppo": {
            0: PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip",
            1: PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip",
            2: PROJECT_DIR / "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip"
        },
        "image_only_sac": PROJECT_DIR / "sac_image_only/aorta_mb/sac_image_only_aorta_mb_final.zip",
        "goal_ppo": PROJECT_DIR / "ppo_checkpoints/aorta_mb/ppo_mb_aorta_final_model.zip",
        "goal_sac": PROJECT_DIR / "sac_checkpoints/aorta_mb/sac_mb_aorta_final_model.zip"
    },
    "liver": {
        "image_only_ppo": {
            0: PROJECT_DIR / "ppo_checkpoints/liver_centre/ppo_scale_liver_300000_steps.zip"
        },
        "image_only_sac": PROJECT_DIR / "sac_image_only/liver_mb/sac_image_only_liver_mb_final.zip",
        "goal_ppo": PROJECT_DIR / "ppo_checkpoints/liver_mb/ppo_mb_liver_final_model.zip",
        "goal_sac": PROJECT_DIR / "sac_checkpoints/liver_mb/sac_mb_liver_final_model.zip"
    }
}


def make_ascii_bar(val: float, max_val: float = 100.0, width: int = 20) -> str:
    """Render an ASCII bar for real-time terminal display."""
    frac = min(max(val / max(max_val, 1e-4), 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "█" * filled + "░" * (width - filled)
    return bar


def main():
    parser = argparse.ArgumentParser(description="Live Interactive Autonomous Ultrasound Evaluation")
    parser.add_argument("--anatomy", type=str, default="spine", choices=["spine", "aorta", "liver"], help="Target anatomy")
    parser.add_argument("--patient", type=str, default="s0344", help="Patient ID (Train: s0011, s0058, s0250 | Held-Out: s0344, s0358)")
    parser.add_argument("--model-type", type=str, default="image_only_sac", 
                        choices=["image_only", "image_only_ppo", "image_only_sac", "image_only_sac_lb", "goal_ppo", "goal_sac"], 
                        help="Model type to evaluate")
    parser.add_argument("--model-seed", type=int, default=0, help="Model seed for image_only_ppo (0 for spine; 0, 1, 2 for aorta)")
    parser.add_argument("--backend", type=str, default="conv", choices=["conv", "unet"], help="Acoustic simulation backend")
    parser.add_argument("--seed", type=int, default=1000, help="Environment reset seed (controls random probe starting position)")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum episode horizon")
    parser.add_argument("--fixed-horizon", action="store_true", help="Disable early termination to evaluate full 300 steps")
    parser.add_argument("--gui", action="store_true", help="Launch interactive PyBullet 3D GUI window")
    parser.add_argument("--step-delay", type=float, default=0.03, help="Terminal display pause per step (seconds)")
    parser.add_argument("--render-video", type=str, default=None, help="Optional output path for side-by-side MP4 video (e.g. live_demo.mp4)")
    args = parser.parse_args()

    # Determine checkpoint
    target = args.anatomy
    mtype = "image_only_ppo" if args.model_type == "image_only" else args.model_type
    is_image_only = "image_only" in mtype
    early_termination = not args.fixed_horizon

    if mtype == "image_only_ppo":
        ckpt_dict = CHECKPOINTS[target]["image_only_ppo"]
        m_seed = args.model_seed if args.model_seed in ckpt_dict else list(ckpt_dict.keys())[0]
        ckpt_path = ckpt_dict[m_seed]
        model_name = f"Image-Only PPO (Seed {m_seed})"
    else:
        ckpt_path = CHECKPOINTS[target][mtype]
        if "sac" in mtype and "image_only" in mtype:
            model_name = f"Image-Only SAC ({'LB' if 'lb' in mtype else 'MB'})"
        elif "sac" in mtype:
            model_name = "Goal-Conditioned SAC"
        else:
            model_name = "Goal-Conditioned PPO"

    if not Path(ckpt_path).exists():
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    subj_dir = str(PROJECT_DIR / "totalseg_patients" / args.patient)
    if not Path(subj_dir).exists():
        print(f"[ERROR] Patient directory not found: {subj_dir}")
        sys.exit(1)

    cohort_label = "HELD-OUT (External / Unseen)" if args.patient in TEST_PATIENTS or args.patient in ["s0344", "s0358"] else "IN-DOMAIN (Training)"

    print("=" * 82)
    print("      SONOGYM AUTONOMOUS ULTRASOUND LIVE NAVIGATION EVALUATION")
    print("=" * 82)
    print(f"  Anatomy:           {target.upper()}")
    print(f"  Patient:           {args.patient} [{cohort_label}]")
    print(f"  Model:             {model_name}")
    print(f"  Strict Image-Only: {'YES (NO goal coordinate leak)' if is_image_only else 'NO (Goal coordinate guidance)'}")
    print(f"  Checkpoint:        {ckpt_path}")
    print(f"  Horizon:           {args.max_steps} steps | Early Termination: {early_termination}")
    print(f"  PyBullet Mode:     {'GUI (Interactive 3D Window)' if args.gui else 'DIRECT (Headless)'}")
    print(f"  Reset Seed:        {args.seed}")
    print("=" * 82)

    # Load model
    print("\n[+] Loading policy weights into CPU memory...")
    if "sac" in mtype:
        model = SAC.load(str(ckpt_path), device="cpu")
    else:
        model = PPO.load(str(ckpt_path), device="cpu")

    img_size = 128 if is_image_only else 256
    if hasattr(model.observation_space, "spaces") and "image" in model.observation_space.spaces:
        img_size = int(model.observation_space.spaces["image"].shape[0])

    backend = "unet" if "lb" in mtype or args.backend == "unet" else "conv"
    print(f"[+] Spawning PyBullet robotic environment (backend: {backend}, image resolution: {img_size}x{img_size})...")
    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        task="navigation",
        anatomy=target,
        sim_mode=backend,
        size=img_size,
        pixel_spacing=1.0,
        max_episode_steps=args.max_steps,
        early_termination=early_termination,
        goal_conditioned=(not is_image_only),
        render_mode="human" if args.gui else "direct",
        device="cpu"
    )

    obs, info = env.reset(seed=args.seed)

    target_pos_world = env.unwrapped.target_world_pos.copy() * 1000.0  # mm
    initial_probe_pos = env.unwrapped.current_probe_tip_pos.copy() * 1000.0  # mm
    initial_dist = float(np.linalg.norm(initial_probe_pos - target_pos_world))

    if args.gui:
        import pybullet as p
        t_pos_m = env.unwrapped.target_world_pos
        p.resetDebugVisualizerCamera(
            cameraDistance=0.75, cameraYaw=50, cameraPitch=-30,
            cameraTargetPosition=[float(t_pos_m[0]), float(t_pos_m[1]), float(t_pos_m[2] - 0.05)]
        )
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.addUserDebugText(f"★ {target.upper()} GOAL", [t_pos_m[0], t_pos_m[1], t_pos_m[2] + 0.05], textColorRGB=[0, 1, 0], textSize=1.3)

    print(f"[+] Initial Probe Position:  ({initial_probe_pos[0]:+.1f}, {initial_probe_pos[1]:+.1f}, {initial_probe_pos[2]:+.1f}) mm")
    print(f"[+] Target Anatomical Goal:  ({target_pos_world[0]:+.1f}, {target_pos_world[1]:+.1f}, {target_pos_world[2]:+.1f}) mm")
    print(f"[+] Initial Start Distance:  {initial_dist:.2f} mm")
    print(f"[+] Success Criteria:        Position Error < 5.0 mm AND Rotation Error < 10.0°\n")

    print("-" * 82)
    print(" STEP | PROBE POS (X, Y, Z mm)     | DIST TO GOAL   | ROT ERR | CONTACT | QUALITY | STATUS")
    print("-" * 82)

    frames = []
    dist_history = []
    success_step = None
    prev_pos_m = env.unwrapped.current_probe_tip_pos.copy()
    t_start = time.time()

    for step in range(1, args.max_steps + 1):
        # Format image-only observation wrapper if needed
        if is_image_only:
            contact_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
            quat = np.array(env.unwrapped.current_probe_quat, dtype=np.float32)
            obs["contact"] = np.array([contact_val], dtype=np.float32)
            obs["orientation"] = quat

        curr_pos_mm = env.unwrapped.current_probe_tip_pos.copy() * 1000.0

        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        pos_err = float(info.get("pos_error_mm", 0.0))
        rot_err = float(info.get("rot_error_deg", 0.0))
        q_val = float(info.get("image_quality", 0.0))
        c_state = "YES" if (obs.get("force", [0.0])[0] * 10.0 >= 1.0) else "NO "
        dist_history.append(pos_err)

        # Success check
        is_success = (pos_err < 5.0 and rot_err < 10.0)
        if is_success and success_step is None:
            success_step = step

        if args.gui:
            curr_pos_m = env.unwrapped.current_probe_tip_pos.copy()
            if step > 1:
                p.addUserDebugLine(prev_pos_m.tolist(), curr_pos_m.tolist(), [0, 1, 0] if is_success else [1, 0.8, 0], lineWidth=2.5)
            prev_pos_m = curr_pos_m.copy()

        status_tag = "🎯 SUCCESS" if is_success else f"[{make_ascii_bar(pos_err, initial_dist, width=12)}]"

        print(f" {step:03d}  | ({curr_pos_mm[0]:+6.1f}, {curr_pos_mm[1]:+6.1f}, {curr_pos_mm[2]:+6.1f}) | {pos_err:6.2f} mm    | {rot_err:5.1f}°  |   {c_state}   |  {q_val:.3f}  | {status_tag}", flush=True)

        # Video frame collection
        if args.render_video:
            bmode_img = obs.get("image", None)
            if bmode_img is not None:
                if isinstance(bmode_img, np.ndarray) and bmode_img.ndim == 3 and bmode_img.shape[0] == 1:
                    bmode_img = bmode_img[0]
                frames.append((bmode_img.copy(), curr_pos_mm.copy(), pos_err, rot_err, q_val, step))

        if args.step_delay > 0:
            time.sleep(args.step_delay)

        if term or trunc:
            break

    total_time = time.time() - t_start
    final_pos = float(info.get("pos_error_mm", 0.0))
    final_rot = float(info.get("rot_error_deg", 0.0))
    achieved = (success_step is not None)

    print("-" * 82)
    print("\n============================== EVALUATION SUMMARY ==============================")
    print(f"  Outcome:             {'✅ TARGET REACHED (SUCCESS)' if achieved else '❌ TIMED OUT'}")
    if achieved:
        print(f"  First Success Step:  Step {success_step} (out of {args.max_steps})")
    print(f"  Final Position Error: {final_pos:.2f} mm  (Threshold: < 5.0 mm)")
    print(f"  Final Rotation Error: {final_rot:.2f}°   (Threshold: < 10.0°)")
    print(f"  Steps Executed:      {step}")
    print(f"  Execution Time:      {total_time:.2f} seconds ({step / max(total_time, 0.001):.1f} steps/s)")
    print("================================================================================\n")

    # Render video if requested
    if args.render_video and len(frames) > 0:
        out_video_path = Path(args.render_video)
        out_video_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[+] Compiling side-by-side diagnostic video to: {out_video_path} ...")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import cv2

        fig, (ax_us, ax_curve) = plt.subplots(1, 2, figsize=(11, 5), dpi=120)
        plt.tight_layout()

        canvas_frames = []
        for f_idx, (bmode, pos_pt, p_err, r_err, q, s_num) in enumerate(frames):
            ax_us.clear()
            ax_curve.clear()

            # Left: B-mode ultrasound
            ax_us.imshow(bmode, cmap="gray", vmin=0, vmax=255 if bmode.max() > 1.0 else 1.0)
            ax_us.set_title(f"Live Simulated B-Mode (Step {s_num})\nQuality Q = {q:.3f}", fontsize=11, fontweight="bold")
            ax_us.axis("off")

            # Right: Distance to Goal Curve
            steps_x = list(range(1, s_num + 1))
            errs_y = dist_history[:s_num]
            ax_curve.plot(steps_x, errs_y, color="#1f77b4", lw=2.5, label="Distance to Target (mm)")
            ax_curve.axhline(5.0, color="green", linestyle="--", lw=1.8, label="Success Zone (< 5.0 mm)")
            ax_curve.scatter([s_num], [p_err], color="red", s=60, zorder=5)

            ax_curve.set_xlim(1, max(args.max_steps, s_num + 5))
            ax_curve.set_ylim(0, max(initial_dist * 1.1, 30.0))
            ax_curve.set_xlabel("Episode Step", fontsize=10)
            ax_curve.set_ylabel("Error to Goal (mm)", fontsize=10)
            status_text = "[SUCCESS] TARGET REACHED" if p_err < 5.0 and r_err < 10.0 else f"Distance: {p_err:.1f} mm | Rot: {r_err:.1f}°"
            ax_curve.set_title(f"Real-Time Convergence\n{status_text}", fontsize=11, fontweight="bold", color="green" if p_err < 5.0 else "black")
            ax_curve.legend(loc="upper right", fontsize=9)
            ax_curve.grid(True, linestyle=":", alpha=0.6)

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            canvas_frames.append(bgr)

        plt.close(fig)

        # Write video
        h, w, _ = canvas_frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, 15, (w, h))
        for cf in canvas_frames:
            writer.write(cf)
        writer.release()
        print(f"[DONE] Video saved successfully to {out_video_path} ({len(canvas_frames)} frames @ 15 fps)!\n")

    env.close()


if __name__ == "__main__":
    main()
