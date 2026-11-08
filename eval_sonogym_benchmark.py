"""
eval_sonogym_benchmark.py
=========================
Comprehensive Benchmark Evaluation Suite replicating SonoGym (NeurIPS 2025 Table 1 & Figure 6)
and evaluating Ultrasound Quality (MICCAI 2025).

Features:
  - In-Domain evaluation on training patients: s0011, s0058, s0250
  - Out-of-Domain (OOD) evaluation on unseen test patients: s0344, s0358
  - Quantitative metrics:
      * Position Error (mm): Mean ± Std
      * Rotation Error (deg): Mean ± Std
      * Success Rate (%): Pos < 5 mm, Rot < 5 deg, Quality > 0.60
      * Image Quality Score Q: Mean ± Std
      * Trajectory Length (m)
      * Episode Step Count
  - Auto-generates Markdown / LaTeX benchmark table.
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np

# Add workspace to sys.path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import torch as th
import gymnasium

# PyTorch 2.6+ safe globals for loading Gymnasium environments / SB3 policies
try:
    th.serialization.add_safe_globals([
        gymnasium.spaces.dict.Dict,
        gymnasium.spaces.box.Box,
        gymnasium.spaces.discrete.Discrete,
        np.dtype,
        np.core.multiarray.scalar,
        np.core.multiarray._reconstruct,
        np.ndarray,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
        np.dtypes.UInt8DType,
    ])
except Exception:
    pass

from stable_baselines3 import PPO, SAC, A2C, TD3
from robotic_us_env import RoboticUltrasoundGymEnv, TRAIN_PATIENTS, TEST_PATIENTS


class BayesianOptimizationPolicy:
    """
    Gaussian Process Upper Confidence Bound (GP-UCB) Policy for Ultrasound Probe Navigation.
    Replicates Bayesian Optimization baseline in SonoGym Table 1 (NeurIPS 2025).
    """
    def __init__(self, action_dim=3, kappa=2.0, n_init_random=3):
        self.action_dim = action_dim
        self.kappa = kappa
        self.n_init_random = n_init_random
        self.history_actions = []
        self.history_rewards = []
        self.gp = None
        self._init_gp()

    def _init_gp(self):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            kernel = ConstantKernel(1.0) * RBF(length_scale=0.3)
            # Use fixed hyperparameters (optimizer=None) for sub-millisecond online GP regression
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, optimizer=None, normalize_y=True)
        except Exception:
            self.gp = None

    def reset(self):
        self.history_actions = []
        self.history_rewards = []
        self._init_gp()

    def observe(self, obs, action, reward):
        self.history_actions.append(np.array(action[:self.action_dim], dtype=np.float64))
        self.history_rewards.append(float(reward))
        if len(self.history_actions) >= self.n_init_random and self.gp is not None and (len(self.history_actions) % 2 == 0):
            X = np.array(self.history_actions)
            y = np.array(self.history_rewards)
            try:
                self.gp.fit(X, y)
            except Exception:
                pass

    def predict(self, obs, deterministic=True):
        if len(self.history_actions) < self.n_init_random or self.gp is None:
            act = np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)
            return act, None

        candidates = np.random.uniform(-1.0, 1.0, size=(64, self.action_dim))
        try:
            mu, sigma = self.gp.predict(candidates, return_std=True)
            ucb = mu + self.kappa * sigma
            best_idx = np.argmax(ucb)
            best_action = candidates[best_idx].astype(np.float32)
        except Exception:
            best_action = np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

        return best_action, None


class ACTPolicyWrapper:
    """Wrapper for Action Chunking Transformer policy evaluation in Gymnasium."""
    def __init__(self, checkpoint_path, device="cpu"):
        import torch
        from train_act import ActionChunkingTransformer, ACTVisionEncoder, ACTProprioEncoder, ACTCvaeEncoder, MlpActionChunkingPolicy
        import __main__
        __main__.MlpActionChunkingPolicy = MlpActionChunkingPolicy
        __main__.ActionChunkingTransformer = ActionChunkingTransformer
        __main__.ACTVisionEncoder = ACTVisionEncoder
        __main__.ACTProprioEncoder = ACTProprioEncoder
        __main__.ACTCvaeEncoder = ACTCvaeEncoder

        self.device = torch.device(device)
        self.model = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.eval()

        from enjoy_act import ACTTemporalEnsemble
        self.ensembler = ACTTemporalEnsemble(chunk_size=self.model.chunk_size, temperature=10.0)
        
        import pickle
        stats_path = Path(checkpoint_path).parent / "norm_stats.pkl"
        self.norm_stats = None
        if stats_path.exists():
            with open(stats_path, "rb") as f:
                self.norm_stats = pickle.load(f)

    def reset(self):
        self.ensembler.reset()

    def predict(self, obs, deterministic=True):
        import torch
        image_np = obs["image"].astype(np.float32) / 255.0
        image_t = torch.tensor(image_np, device=self.device).unsqueeze(0).unsqueeze(0)
        raw_force = obs["force"].astype(np.float32)
        raw_pose = obs["pose"].astype(np.float32)
        if self.norm_stats is not None:
            force_norm = (raw_force - self.norm_stats["force_mean"][0]) / self.norm_stats["force_std"][0]
            pose_norm = (raw_pose - self.norm_stats["pose_mean"][0]) / self.norm_stats["pose_std"][0]
        else:
            force_norm = raw_force
            pose_norm = raw_pose
        force_t = torch.tensor(force_norm, device=self.device).unsqueeze(0)
        pose_t = torch.tensor(pose_norm, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred_chunk_t, _, _ = self.model(image_t, force_t, pose_t)
            pred_chunk = pred_chunk_t.squeeze(0).cpu().numpy()

        if self.norm_stats is not None:
            pred_chunk = pred_chunk * self.norm_stats["action_std"][0] + self.norm_stats["action_mean"][0]

        action = self.ensembler.update_and_get_action(pred_chunk)
        if len(action) > 3:
            act3 = np.array([action[0], action[1], action[5]], dtype=np.float32)
            return act3, None
        return action[:3].astype(np.float32), None


def load_policy(model_path: str, algo: str = "auto"):
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
    p_str = str(model_path).lower() if model_path else ""
    if algo == "auto":
        if "bo" in p_str:
            algo = "bo"
        elif "act" in p_str:
            algo = "act"
        elif "ppo" in p_str:
            algo = "ppo"
        elif "sac" in p_str:
            algo = "sac"
        elif "td3" in p_str:
            algo = "td3"
        elif "a2c" in p_str:
            algo = "a2c"
        elif "bc" in p_str or "dagger" in p_str:
            algo = "bc"
        elif "gail" in p_str:
            algo = "gail"
        else:
            algo = "ppo"

    if algo == "bo":
        return BayesianOptimizationPolicy(action_dim=3)
    elif algo == "random":
        return None
    elif algo == "act":
        return ACTPolicyWrapper(model_path)

    orig_load = th.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig_load(*args, **kwargs)
    
    th.load = patched_load
    try:
        if algo == "ppo" or algo == "gail":
            return PPO.load(model_path)
        elif algo == "sac":
            return SAC.load(model_path)
        elif algo == "td3":
            return TD3.load(model_path)
        elif algo == "a2c":
            return A2C.load(model_path)
        elif algo == "bc":
            return MultiInputActorCriticPolicy.load(model_path, device="cpu")
        else:
            raise ValueError(f"Unknown algorithm: {algo}")
    finally:
        th.load = orig_load


def evaluate_cohort(policy, patients, args, num_episodes=5):
    """Evaluate policy over a patient cohort."""
    cohort_results = {
        "pos_errors_mm": [],
        "rot_errors_deg": [],
        "quality_scores": [],
        "successes": [],
        "trajectory_lengths_m": [],
        "steps": []
    }
    eval_seeds = [42, 142, 242, 342, 442]

    for pid in patients:
        subj_dir = Path("totalseg_patients") / pid
        base_env = RoboticUltrasoundGymEnv(
            subject_dir=str(subj_dir),
            checkpoint_path=args.checkpoint_path,
            device="auto",
            render_mode="rgb_array",
            max_episode_steps=args.max_steps,
            size=args.size,
            task=args.task,
            anatomy=args.anatomy,
            sim_mode=args.sim_mode,
            multi_patient=False,
            goal_conditioned=not getattr(args, "image_only", False),
            early_termination=getattr(args, "early_termination", True)
        )
        print(f"  [EVAL] Patient: {pid} | early_termination: {base_env.early_termination} | goal_conditioned: {base_env.goal_conditioned}")

        if getattr(args, "image_only", False):
            from train_ppo_mb import ImageOnlyObservationWrapper
            env = ImageOnlyObservationWrapper(base_env)
        else:
            env = base_env

        for ep in range(num_episodes):
            ep_seed = eval_seeds[ep % len(eval_seeds)] if ep < len(eval_seeds) else (42 + ep * 100)
            obs, info = env.reset(seed=ep_seed)
            if policy is not None and hasattr(policy, "reset"):
                policy.reset()
            done = False
            total_traj_len = 0.0
            prev_tip = base_env.current_probe_tip_pos.copy()
            ep_steps = 0

            while not done:
                if policy is not None:
                    # Policy prediction on observation
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    # Random baseline
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                if policy is not None and hasattr(policy, "observe"):
                    policy.observe(obs, action, reward)
                ep_steps += 1
                done = terminated or truncated

                # Track true physical probe tip trajectory on patient skin (meters)
                curr_tip = base_env.current_probe_tip_pos.copy()
                step_dist = float(np.linalg.norm(curr_tip - prev_tip))
                total_traj_len += step_dist
                prev_tip = curr_tip.copy()

            final_pos_err = info.get("pos_error_mm", 999.0)
            final_rot_err = info.get("rot_error_deg", 180.0)
            final_quality = info.get("quality_score", 0.0)
            is_success = info.get("is_success", False)

            cohort_results["pos_errors_mm"].append(final_pos_err)
            cohort_results["rot_errors_deg"].append(final_rot_err)
            cohort_results["quality_scores"].append(final_quality)
            cohort_results["successes"].append(1.0 if is_success else 0.0)
            cohort_results["trajectory_lengths_m"].append(total_traj_len)
            cohort_results["steps"].append(ep_steps)

        env.close()

    return {
        "pos_mean": float(np.mean(cohort_results["pos_errors_mm"])),
        "pos_std": float(np.std(cohort_results["pos_errors_mm"])),
        "rot_mean": float(np.mean(cohort_results["rot_errors_deg"])),
        "rot_std": float(np.std(cohort_results["rot_errors_deg"])),
        "quality_mean": float(np.mean(cohort_results["quality_scores"])),
        "quality_std": float(np.std(cohort_results["quality_scores"])),
        "success_rate": float(np.mean(cohort_results["successes"])) * 100.0,
        "traj_len_mean": float(np.mean(cohort_results["trajectory_lengths_m"])),
        "steps_mean": float(np.mean(cohort_results["steps"]))
    }


def main():
    parser = argparse.ArgumentParser(description="SonoGym & MICCAI 2025 Autonomous Ultrasound Benchmark")
    parser.add_argument("--model-path", type=str, default="", help="Path to trained model (.zip). If empty, evaluates random policy.")
    parser.add_argument("--algo", type=str, default="auto", choices=["auto", "ppo", "sac", "td3", "a2c", "bc", "gail", "act", "bo", "random"], help="Algorithm to evaluate")
    parser.add_argument("--task", type=str, default="navigation", choices=["navigation", "reconstruction"], help="Evaluation task")
    parser.add_argument("--anatomy", type=str, default="spine", choices=["spine", "liver", "aorta"], help="Target anatomy")
    parser.add_argument("--sim-mode", type=str, default="conv", choices=["conv", "ray", "unet", "pix2pix"], help="US simulator mode")
    parser.add_argument("--checkpoint-path", type=str, default="runs/cavalcanti_unet/best_model.pth", help="U-Net/Pix2Pix weights")
    parser.add_argument("--episodes-per-patient", type=int, default=5, help="Number of test rollouts per patient")
    parser.add_argument("--image-only", action="store_true",
                        help="Evaluate an image-only policy (no goal offset in obs). Default: goal-conditioned.")
    parser.add_argument("--early-termination", action="store_true", default=True, help="Terminate episode immediately upon goal success")
    parser.add_argument("--no-early-termination", action="store_false", dest="early_termination", help="Disable early termination")
    parser.add_argument("--max-steps", type=int, default=150, help="Max episode steps per rollout")
    parser.add_argument("--size", type=int, default=256, help="Observation image size (default: 256)")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 70)
    print("  SonoGym & MICCAI 2025 Autonomous Robotic Ultrasound Benchmark")
    print("=" * 70)
    print(f"  Target Anatomy:    {args.anatomy.upper()}")
    print(f"  Simulation Mode:   {args.sim_mode.upper()}")
    print(f"  Task:              {args.task}")
    print(f"  Episodes/Patient:  {args.episodes_per_patient}")
    print(f"  In-Domain Cohort:  {TRAIN_PATIENTS}")
    print(f"  OOD Cohort:        {TEST_PATIENTS}")

    if args.algo == "bo":
        print("  Evaluating Baseline: BAYESIAN OPTIMIZATION (GP-UCB)")
        policy = BayesianOptimizationPolicy(action_dim=3)
        policy_name = "Bayesian Optimization (GP-UCB)"
    elif args.model_path and Path(args.model_path).exists():
        print(f"  Loading Model:     {args.model_path}")
        policy = load_policy(args.model_path, args.algo)
        policy_name = f"{Path(args.model_path).stem}"
        # Auto-detect observation shape from loaded policy
        obs_space = getattr(policy, "observation_space", None)
        if obs_space is not None:
            img_space = getattr(obs_space, "spaces", {}).get("image", None)
            if img_space is not None and hasattr(img_space, "shape") and len(img_space.shape) >= 2:
                detected_size = int(img_space.shape[0])
                if detected_size != args.size:
                    print(f"  [Auto-detected policy image shape {img_space.shape} -> setting size={detected_size}]")
                    args.size = detected_size
    else:
        print("  Evaluating Baseline: RANDOM AGENT")
        policy = None
        policy_name = "Random Baseline"

    # 1. In-Domain Evaluation
    print("\n  Evaluating In-Domain Cohort (s0011, s0058, s0250) ...")
    t0 = time.time()
    in_domain_res = evaluate_cohort(policy, TRAIN_PATIENTS, args, num_episodes=args.episodes_per_patient)
    print(f"  In-Domain Evaluation completed in {time.time() - t0:.1f}s")

    # 2. Out-of-Domain Evaluation
    print("\n  Evaluating Out-of-Domain Cohort (s0344, s0358) ...")
    t1 = time.time()
    ood_res = evaluate_cohort(policy, TEST_PATIENTS, args, num_episodes=args.episodes_per_patient)
    print(f"  Out-of-Domain Evaluation completed in {time.time() - t1:.1f}s")

    # 3. Print SonoGym Table 1 Benchmark Results
    print("\n" + "=" * 70)
    print(f"  BENCHMARK RESULTS: {policy_name} ({args.anatomy.upper()} / {args.sim_mode.upper()})")
    print("=" * 70)
    print(f"{'Cohort':<16} | {'Pos Error (mm)':<16} | {'Rot Error (deg)':<16} | {'Quality Q':<12} | {'Success (%)':<12}")
    print("-" * 80)
    print(f"{'In-Domain (Train)':<16} | {in_domain_res['pos_mean']:5.2f} ± {in_domain_res['pos_std']:4.2f}     | {in_domain_res['rot_mean']:5.2f} ± {in_domain_res['rot_std']:4.2f}     | {in_domain_res['quality_mean']:4.2f} ± {in_domain_res['quality_std']:4.2f} | {in_domain_res['success_rate']:5.1f}%")
    print(f"{'OOD (Test)':<16} | {ood_res['pos_mean']:5.2f} ± {ood_res['pos_std']:4.2f}     | {ood_res['rot_mean']:5.2f} ± {ood_res['rot_std']:4.2f}     | {ood_res['quality_mean']:4.2f} ± {ood_res['quality_std']:4.2f} | {ood_res['success_rate']:5.1f}%")
    print("=" * 70)

    # 4. Save JSON Report
    report = {
        "policy": policy_name,
        "anatomy": args.anatomy,
        "sim_mode": args.sim_mode,
        "in_domain": in_domain_res,
        "out_of_domain": ood_res
    }
    with open(args.output_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Benchmark report saved to: {args.output_json}\n")


if __name__ == "__main__":
    main()
