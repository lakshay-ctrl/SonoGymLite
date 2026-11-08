"""
train_sac.py — Soft Actor-Critic (SAC) training for autonomous robotic ultrasound.
===================================================================================

Trains a Soft Actor-Critic (SAC) policy (via Stable-Baselines3) inside the
``RoboticUltrasoundGymEnv`` Gymnasium environment.

SAC is an off-policy maximum entropy actor-critic algorithm specifically designed
for continuous robotic control. Key advantages for robotic ultrasound scanning:
  1. Replay Buffer (50k steps): Learns from all past physical transitions,
     offering 10x-50x higher sample efficiency than on-policy A2C/PPO.
  2. Maximum Entropy Objective: Optimizes for (Reward + Entropy), which naturally
     produces smooth, human-like probe trajectories without aggressive jitter.
  3. Automatic Temperature Tuning (ent_coef="auto"): Auto-tunes the entropy
     weight alpha to balance exploration vs. exploitation during training.

Usage
-----
Basic training run (50 k steps, default settings)::

    python train_sac.py

Extended training with custom options::

    python train_sac.py \\
        --timesteps 100000 \\
        --subject totalseg_patients/s0058 \\
        --buffer-size 50000 \\
        --batch-size 256 \\
        --lr 3e-4 \\
        --save-freq 10000 \\
        --tb-log ./sac_tensorboard/ \\
        --save-dir ./sac_checkpoints/

Monitor training progress with TensorBoard::

    tensorboard --logdir ./sac_tensorboard/

Saved checkpoints are stored as ``sac_model_<N>_steps.zip`` inside ``--save-dir``.
The final model is saved as ``sac_final_model.zip``.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import gymnasium as gym

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from robotic_us_env import RoboticUltrasoundGymEnv

def main():
    parser = argparse.ArgumentParser(description="Train a SAC Agent for Autonomous Robotic Ultrasound Scanning")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total timesteps to train (default: 50k)")
    parser.add_argument("--tb-log", type=str, default="./sac_tensorboard/", help="TensorBoard log directory")
    parser.add_argument("--save-dir", type=str, default="./sac_checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Subject patient directory")
    parser.add_argument("--save-freq", type=int, default=10000, help="Save a checkpoint every N steps")
    parser.add_argument("--buffer-size", type=int, default=50000, help="Replay buffer capacity (default: 50000)")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for gradient updates (default: 256)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for actor/critic (default: 3e-4)")
    parser.add_argument("--learning-starts", type=int, default=1000, help="Steps before gradient updates begin (default: 1000)")
    parser.add_argument("--task", type=str, default="navigation", choices=["navigation", "reconstruction", "legacy"], help="Task mode")
    parser.add_argument("--anatomy", type=str, default="spine", choices=["spine", "liver", "aorta"], help="Target anatomy")
    parser.add_argument("--sim-mode", type=str, default="conv", choices=["conv", "ray", "unet", "pix2pix"], help="Ultrasound simulator mode")
    parser.add_argument("--multi-patient", action="store_true", help="Enable domain randomization across training patients")
    parser.add_argument("--size", type=int, default=128, help="Observation image size (default: 128)")
    parser.add_argument("--device", type=str, default="cuda", help="Device for SAC policy (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--obs", type=str, default="full", choices=["full", "image_only"], help="Observation space")
    parser.add_argument("--horizon", type=int, default=150, help="Evaluation episode horizon")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.tb_log, exist_ok=True)

    print("=" * 60)
    print("  Training SAC Agent for Robotic Ultrasound Scanning")
    print("=" * 60)
    print(f"  Task Mode:         {args.task}")
    print(f"  Target Anatomy:    {args.anatomy}")
    print(f"  Simulation Mode:   {args.sim_mode} ({'Model-Based' if args.sim_mode in ['conv', 'ray'] else 'Learning-Based'})")
    print(f"  Multi-Patient:     {args.multi_patient}")
    print(f"  Patient Subject:   {args.subject}")
    print(f"  Total Timesteps:   {args.timesteps}")
    print(f"  Buffer Size:       {args.buffer_size}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Learning Rate:     {args.lr}")
    print(f"  Learning Starts:   {args.learning_starts}")
    # Force CUDA acceleration — non-negotiable for fast training
    assert torch.cuda.is_available(), (
        "[FATAL ERROR] CUDA GPU acceleration is MANDATORY for training!\n"
        "No CUDA-capable GPU was detected by PyTorch."
    )
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n[GPU REQUIRED] Training locked to CUDA: {gpu_name} ({gpu_mem:.1f} GB VRAM)\n")

    from us_feature_extractor import GoalConditionedExtractor, UltrasoundCNNExtractor
    from results_logger import save_run_result, ALL_PATIENTS, EVAL_SEEDS, TRAIN_PATIENTS

    class ImageOnlyObservationWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            orig_img = env.observation_space["image"]
            self.observation_space = gym.spaces.Dict({
                "image": orig_img,
                "contact": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            })

        def observation(self, obs):
            contact_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
            return {
                "image": obs["image"],
                "contact": np.array([contact_val], dtype=np.float32)
            }

    # 1. Instantiate the Gymnasium environment in headless (rgb_array) mode
    def make_env():
        env = RoboticUltrasoundGymEnv(
            subject_dir=args.subject,
            checkpoint_path="runs/cavalcanti_unet/best_model.pth",
            device="cuda",
            render_mode="rgb_array",
            max_episode_steps=150,
            size=args.size,
            task=args.task,
            anatomy=args.anatomy,
            sim_mode=args.sim_mode,
            multi_patient=args.multi_patient,
            goal_conditioned=(args.obs != "image_only"),
        )
        if args.obs == "image_only":
            env = ImageOnlyObservationWrapper(env)
        return Monitor(env)

    train_env = DummyVecEnv([make_env])
    if args.multi_patient:
        print("  [multi-patient] Dynamic patient domain randomization across s0011, s0058, s0250 ENABLED")
    else:
        print(f"  [single-patient] Static subject: {args.subject}")

    # 3. Setup periodic checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix="sac_model",
        verbose=1
    )

    callbacks = [checkpoint_callback]

    if args.obs == "image_only":
        policy_kwargs = dict(
            features_extractor_class=UltrasoundCNNExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=256),
        )
    else:
        policy_kwargs = dict(
            features_extractor_class=GoalConditionedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=128, vec_output_dim=128),
        )

    # 4. Initialize SAC Model
    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),     # Update Q-networks and actor every step
        gradient_steps=1,
        ent_coef="auto",            # Automatically tunes entropy coefficient alpha
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.tb_log,
        seed=args.seed,
        verbose=1,
        device=args.device
    )

    print("\n  Starting training loop ...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True
        )
        print("\n  [OK] Training completed successfully!")
        
        mode_tag = "mb" if args.sim_mode in ["conv", "ray"] else "lb"
        # Save final model under both specific and general filenames
        named_path = os.path.join(args.save_dir, f"sac_{mode_tag}_{args.anatomy}_final_model.zip")
        final_path = os.path.join(args.save_dir, "sac_final_model.zip")
        model.save(named_path)
        model.save(final_path)
        print(f"  Final model saved to: {named_path} and {final_path}")

        # Standardized 25-episode benchmark evaluation
        episodes = []
        for pid in ALL_PATIENTS:
            is_train_p = pid in TRAIN_PATIENTS
            e_env = RoboticUltrasoundGymEnv(
                subject_dir=f"totalseg_patients/{pid}",
                task=args.task,
                anatomy=args.anatomy,
                sim_mode=args.sim_mode,
                size=args.size,
                max_episode_steps=args.horizon,
                goal_conditioned=(args.obs != "image_only")
            )
            if args.obs == "image_only":
                e_env = ImageOnlyObservationWrapper(e_env)
            for ep_seed in EVAL_SEEDS:
                obs, _ = e_env.reset(seed=ep_seed)
                traj_len = 0.0
                prev_pos = e_env.current_probe_tip_pos.copy()
                for _ in range(args.horizon):
                    act, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = e_env.step(act)
                    curr_pos = e_env.current_probe_tip_pos.copy()
                    traj_len += float(np.linalg.norm(curr_pos - prev_pos))
                    prev_pos = curr_pos
                    if term or trunc:
                        break
                episodes.append({
                    "patient": pid,
                    "split": "train" if is_train_p else "test",
                    "eval_seed": ep_seed,
                    "final_pos_err_mm": float(info["pos_error_mm"]),
                    "final_rot_err_deg": float(info["rot_error_deg"]),
                    "success": bool(info["pos_error_mm"] < 5.0 and info["rot_error_deg"] < 10.0),
                    "quality_Q": float(info["quality_score"]),
                    "path_length_m": float(traj_len)
                })
            e_env.close()

        method_str = f"sac_{mode_tag}_{args.obs}"
        settings = {
            "anatomy": args.anatomy,
            "sim_mode": args.sim_mode,
            "obs": args.obs,
            "timesteps": args.timesteps,
            "seed": args.seed,
            "horizon": args.horizon
        }
        out_json = save_run_result(
            block="block_a",
            task="nav",
            target=args.anatomy,
            method=method_str,
            seed=args.seed,
            settings=settings,
            episodes=episodes
        )
        print(f"[RESULTS] Saved compliant benchmark to {out_json}")
        
    except KeyboardInterrupt:
        print("\n  [Warning] Training interrupted by user. Saving current checkpoint...")
        mode_tag = "mb" if args.sim_mode in ["conv", "ray"] else "lb"
        interrupted_path = os.path.join(args.save_dir, f"sac_{mode_tag}_{args.anatomy}_interrupted_model.zip")
        model.save(interrupted_path)
        print(f"  Interrupted model saved to: {interrupted_path}")
        
    finally:
        train_env.close()
        print("  Environment closed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
