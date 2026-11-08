"""
enjoy_policy.py — Run a trained RL or IL agent visually in the PyBullet GUI simulator.
=============================================================================

Loads a trained Stable-Baselines3 model checkpoint (.zip) — A2C, PPO, or SAC —
or an imitation learning policy (BC, GAIL) and executes it inside the
RoboticUltrasoundGymEnv environment in "human" rendering mode.

Usage
-----
    python enjoy_policy.py --checkpoint a2c_checkpoints/a2c_final_model.zip
    python enjoy_policy.py --checkpoint sac_checkpoints/sac_final_model.zip
    python enjoy_policy.py --checkpoint my_model.zip --algo sac
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from robotic_us_env import RoboticUltrasoundGymEnv

# Monkey-patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility
try:
    import torch
    original_load = torch.load
    def patched_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
except Exception:
    pass

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained A2C agent visually in PyBullet")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the trained SB3 model .zip file")
    parser.add_argument("--expert", action="store_true", help="Run deterministic geometric admittance trajectory toward target (no checkpoint required)")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--delay", type=float, default=0.02, help="Delay between steps in seconds for smoother viewing")
    parser.add_argument("--algo", type=str, default=None, choices=["a2c", "ppo", "sac", "bc", "gail"], help="Algorithm (auto-detected from filename if not set)")
    parser.add_argument("--headless", action="store_true", help="Run in headless (rgb_array) mode without opening PyBullet GUI")
    parser.add_argument("--skip-unet", action="store_true", help="Skip U-Net inference and return raw bone segmentation masks")
    parser.add_argument("--scale-y", type=float, default=1.0, help="Multiplier for Y-axis action to speed up visual sweep (default: 1.0)")
    parser.add_argument("--smooth-alpha", type=float, default=0.35, help="EMA smoothing factor for evaluation actions (default: 0.35)")
    parser.add_argument("--lock-x", action="store_true", help="Lock lateral X movement to keep probe centered on the spine ridge")
    parser.add_argument("--anatomy", type=str, default="spine", choices=["spine", "liver", "aorta"], help="Target anatomy")
    parser.add_argument("--sim-mode", type=str, default="unet", choices=["unet", "conv", "ray"], help="Ultrasound simulator mode")
    parser.add_argument("--show-us", action="store_true", help="Display real-time OpenCV ultrasound viewer window")
    args = parser.parse_args()

    if not args.expert and args.checkpoint is None:
        parser.error("Either --checkpoint <path> or --expert must be provided.")

    print("=" * 60)
    print("  Evaluating Ultrasound Agent" if not args.expert else "  Evaluating Expert Geometric Admittance Controller")
    print("=" * 60)
    print(f"  Mode:       {'Expert Controller' if args.expert else 'Trained Policy'}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Subject:    {args.subject}")
    print(f"  Anatomy:    {args.anatomy}")
    print(f"  Sim Mode:   {args.sim_mode}")
    print(f"  Episodes:   {args.episodes}")
    print(f"  Headless:   {args.headless}")
    print(f"  Skip U-Net: {args.skip_unet}")
    print("=" * 60)

    # 1. Verify checkpoint exists if provided
    if args.checkpoint and not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

    # 2. Instantiate the Gymnasium environment
    render_mode = "rgb_array" if args.headless else "human"
    print(f"\n[env] Launching PyBullet simulator (mode: {render_mode})...")
    
    import numpy as np
    eval_substeps = int(np.clip(5 * (args.scale_y ** 0.5), 5, 20))
    eval_max_vel = float(np.clip(1.5 * args.scale_y, 1.5, 8.0))
    
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env = RoboticUltrasoundGymEnv(
        subject_dir=args.subject,
        checkpoint_path="runs/cavalcanti_unet/best_model.pth",
        device=device_str,
        render_mode=render_mode,
        max_episode_steps=args.max_steps if hasattr(args, "max_steps") else 200,
        size=256,
        pixel_spacing=1.0,
        task="navigation",
        anatomy=args.anatomy,
        sim_mode=args.sim_mode,
        skip_unet=args.skip_unet,
        substeps=eval_substeps,
        max_velocity=eval_max_vel,
    )

    # 3. Auto-detect algorithm or load policy if not expert
    model = None
    policy = None
    if not args.expert:
        algo_name = args.algo
        if algo_name is None:
            cp_lower = os.path.basename(args.checkpoint).lower()
            if "sac" in cp_lower:
                algo_name = "sac"
            elif "ppo" in cp_lower:
                algo_name = "ppo"
            elif "gail" in cp_lower:
                algo_name = "gail"
            elif "bc" in cp_lower or "dagger" in cp_lower:
                algo_name = "bc"
            else:
                algo_name = "ppo"
        
        from us_feature_extractor import UltrasoundCNNExtractor
        if algo_name == "gail":
            from gail_wrapper import FlattenMultiInputWrapper
            env = FlattenMultiInputWrapper(env)
            print(f"[model] Loading trained GAIL generator policy...")
            from stable_baselines3.common.policies import ActorCriticPolicy
            policy = ActorCriticPolicy.load(args.checkpoint, device=device_str)
        elif algo_name == "bc":
            print(f"[model] Loading trained BC/IL policy...")
            from stable_baselines3.common.policies import MultiInputActorCriticPolicy
            policy = MultiInputActorCriticPolicy.load(args.checkpoint, device=device_str)
        else:
            if algo_name == "sac":
                AlgoClass = SAC
            elif algo_name == "ppo":
                AlgoClass = PPO
            else:
                AlgoClass = A2C
            print(f"[model] Loading trained {algo_name.upper()} policy...")
            model = AlgoClass.load(args.checkpoint, env=env, device=device_str)

    try:
        import cv2
        for ep in range(1, args.episodes + 1):
            print(f"\n--- Starting Episode {ep}/{args.episodes} ---")
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            done = False
            
            while not done:
                if args.expert:
                    delta = env.target_world_pos[:2] - np.array([env.current_surface_x, env.current_surface_y])
                    dist = float(np.linalg.norm(delta))
                    if dist < 0.002:
                        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        u = delta / dist
                        action = np.array([u[0], u[1], 0.0], dtype=np.float32)
                elif model is not None:
                    action, _states = model.predict(obs, deterministic=True)
                else:
                    action, _states = policy.predict(obs, deterministic=True)
                
                # Format action for environment
                exec_action = np.asarray(action, dtype=np.float32).copy()
                if len(exec_action.shape) > 1:
                    exec_action = exec_action.reshape(-1)

                if getattr(args, 'lock_x', False) and len(exec_action) >= 1:
                    exec_action[0] = 0.0
                
                obs, reward, terminated, truncated, info = env.step(exec_action)
                episode_reward += float(reward)
                step_count += 1
                done = terminated or truncated
                
                pos_err = info.get("pos_error_mm", 0.0)
                rot_err = info.get("rot_error_deg", 0.0)
                force = info.get("contact_force_n", 0.0)
                quality = info.get("quality_score", 0.0)
                is_success = info.get("is_success", False)

                print(f"  Step {step_count:3d} | Pos Err: {pos_err:5.1f} mm | Angle Err: {rot_err:4.1f}° | Force: {force:4.1f} N | Q: {quality:.2f} | Target Reached: {is_success}")
                
                if args.show_us and "image" in obs:
                    disp_img = cv2.resize(obs["image"], (256, 256), interpolation=cv2.INTER_NEAREST)
                    if len(disp_img.shape) == 2:
                        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_GRAY2BGR)
                    color_dist = (0, 255, 0) if pos_err < 10.0 else (0, 200, 255)
                    color_force = (0, 255, 0) if 3.0 <= force <= 5.0 else (0, 0, 255)
                    color_q = (0, 255, 0) if quality >= 0.50 else (0, 0, 255)
                    cv2.putText(disp_img, f"Dist: {pos_err:.1f} mm", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_dist, 2)
                    cv2.putText(disp_img, f"Force: {force:.1f} N (Admittance)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_force, 2)
                    cv2.putText(disp_img, f"Q: {quality:.2f} (MICCAI)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_q, 2)
                    cv2.imshow("Live Robotic Ultrasound (B-Mode)", disp_img)
                    cv2.waitKey(int(max(args.delay * 1000, 1)))

                time.sleep(args.delay)
                
            print(f"\n[Episode {ep} Finished] Steps: {step_count} | Total Reward: {episode_reward:.2f} | Final Pos Error: {info.get('pos_error_mm', 0.0):.2f} mm | Success: {info.get('is_success', False)}")
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n[info] Demo interrupted by user.")
    finally:
        if args.show_us:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        env.close()
        print("[info] Simulator environment closed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
