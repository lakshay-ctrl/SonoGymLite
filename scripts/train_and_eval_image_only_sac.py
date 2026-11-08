#!/usr/bin/env python3
"""
scripts/train_and_eval_image_only_sac.py
========================================
Trains and evaluates Image-Only SAC for SonoSimLite navigation across 4 settings:
1. Spine, MB backend (anatomy='spine', sim_mode='conv')
2. Spine, LB backend (anatomy='spine', sim_mode='unet')
3. Aorta, MB backend (anatomy='aorta', sim_mode='conv')
4. Liver, MB backend (anatomy='liver', sim_mode='conv')

Budget: 300,000 environment steps per setting, seed=0.
Observations: Image (128x128 uint8) + Contact force (1D float32) + Probe orientation (4D float32 quaternion).
Policy: MultiInputPolicy with GoalConditionedExtractor (CNN: 128, Vec MLP: 128 -> 256 embedding).
Hyperparameters: Exact same as existing goal-conditioned SAC (lr=3e-4, buffer_size=50k, batch_size=256,
                 learning_starts=1k, tau=0.005, gamma=0.99, train_freq=(1, 'step'), gradient_steps=1).
Checkpoints saved to NEW directories: sac_image_only/<target>_<backend>/

Evaluation Protocol (identical to image-only PPO table):
- 100 trials per cohort:
  * Training cohort (seeds 1000-1099): s0011 (34), s0058 (33), s0250 (33)
  * Held-Out cohort (seeds 2000-2099): s0344 (50), s0358 (50)
- Horizon: 300 steps
- Success criterion: pos_error < 5.0 mm and rot_error < 10.0 deg
- Two modes reported:
  (a) Early termination at first success
  (b) Fixed horizon (300 steps)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import multiprocessing as mp
import numpy as np
import torch as th
import gymnasium as gym

# Project path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from robotic_us_env import RoboticUltrasoundGymEnv
from us_feature_extractor import GoalConditionedExtractor

TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
HELD_OUT_PATIENTS = ["s0344", "s0358"]

SETTINGS = {
    "spine_mb": {
        "target": "spine",
        "sim_mode": "conv",
        "save_dir": PROJECT_DIR / "sac_image_only" / "spine_mb",
        "name": "Spine MB Image-Only SAC"
    },
    "spine_lb": {
        "target": "spine",
        "sim_mode": "unet",
        "save_dir": PROJECT_DIR / "sac_image_only" / "spine_lb",
        "name": "Spine LB Image-Only SAC"
    },
    "aorta_mb": {
        "target": "aorta",
        "sim_mode": "conv",
        "save_dir": PROJECT_DIR / "sac_image_only" / "aorta_mb",
        "name": "Aorta MB Image-Only SAC"
    },
    "liver_mb": {
        "target": "liver",
        "sim_mode": "conv",
        "save_dir": PROJECT_DIR / "sac_image_only" / "liver_mb",
        "name": "Liver MB Image-Only SAC"
    }
}


class ImageContactOrientationWrapper(gym.Wrapper):
    """
    Observation Wrapper enforcing Image + Contact + Orientation (NO goal displacement).
    """
    def __init__(self, env):
        super().__init__(env)
        orig_img = env.observation_space["image"]
        self.observation_space = gym.spaces.Dict({
            "image": orig_img,
            "contact": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "orientation": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        })
        for key in self.observation_space.spaces.keys():
            assert "goal" not in key.lower() and "delta" not in key.lower(), f"Goal leak in observation: {key}"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._transform_obs(obs), reward, term, trunc, info

    def _transform_obs(self, obs):
        contact_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
        quat = np.array(self.env.unwrapped.current_probe_quat, dtype=np.float32)
        return {
            "image": obs["image"],
            "contact": np.array([contact_val], dtype=np.float32),
            "orientation": quat
        }


def make_train_env(anatomy: str, sim_mode: str, seed: int = 0, log_dir: Path = None):
    def _init():
        subj_dir = str(PROJECT_DIR / "totalseg_patients" / "s0011")
        env = RoboticUltrasoundGymEnv(
            subject_dir=subj_dir,
            checkpoint_path="runs/cavalcanti_unet/best_model.pth",
            task="navigation",
            anatomy=anatomy,
            sim_mode=sim_mode,
            size=128,
            pixel_spacing=1.0,
            max_episode_steps=150,
            early_termination=True,
            multi_patient=True,
            goal_conditioned=False,
            device="cuda" if th.cuda.is_available() else "cpu"
        )
        env = ImageContactOrientationWrapper(env)
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            return Monitor(env, str(log_dir / "monitor.csv"))
        return Monitor(env)
    return _init


def train_single_run(setting_key: str, timesteps: int = 300000, seed: int = 0):
    cfg = SETTINGS[setting_key]
    save_dir = cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = save_dir / "tb_logs"
    tb_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = save_dir / f"sac_image_only_{setting_key}_final.zip"
    if final_model_path.exists():
        print(f"\n[INFO] Checkpoint {final_model_path} already exists. Skipping training.")
        return str(final_model_path), 0.0, None

    print("\n" + "=" * 80)
    print(f"  STARTING TRAINING: {cfg['name']}")
    print(f"  Target: {cfg['target']} | Backend: {cfg['sim_mode']} | Timesteps: {timesteps:,d} | Seed: {seed}")
    print(f"  Output Dir: {save_dir}")
    print("=" * 80)

    train_env = make_train_env(cfg["target"], cfg["sim_mode"], seed=seed, log_dir=save_dir)()

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(save_dir),
        name_prefix="sac_model",
        verbose=1
    )

    policy_kwargs = dict(
        features_extractor_class=GoalConditionedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=128, vec_output_dim=128),
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_dir),
        seed=seed,
        verbose=1,
        device="cuda" if th.cuda.is_available() else "cpu"
    )

    t0 = time.time()
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )
        wall_clock_sec = time.time() - t0
        print(f"\n[SUCCESS] Finished {timesteps:,d} steps in {wall_clock_sec:.1f}s ({wall_clock_sec/3600:.2f} hours).")

        # Get final ep_rew_mean
        ep_rew_mean = None
        if hasattr(model, "ep_info_buffer") and len(model.ep_info_buffer) > 0:
            ep_rew_mean = float(np.mean([ep["r"] for ep in model.ep_info_buffer]))
            print(f"  Final ep_rew_mean: {ep_rew_mean:.2f}")

        model.save(str(final_model_path))
        print(f"  Saved final model to: {final_model_path}")

        train_env.close()
        return str(final_model_path), wall_clock_sec, ep_rew_mean

    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        train_env.close()
        return None, time.time() - t0, None


def run_eval_trial_worker(args):
    """
    Evaluates 1 episode with SAC model on CPU.
    args: (ckpt_path, target, sim_mode, patient, seed, max_steps, mode)
    mode: "early_term" or "fixed"
    """
    th.set_num_threads(1)
    ckpt_path, target, sim_mode, patient, seed, max_steps, mode = args
    subj_dir = str(PROJECT_DIR / "totalseg_patients" / patient)

    model = SAC.load(ckpt_path, device="cpu")

    env = RoboticUltrasoundGymEnv(
        subject_dir=subj_dir,
        checkpoint_path="runs/cavalcanti_unet/best_model.pth",
        task="navigation",
        anatomy=target,
        sim_mode=sim_mode,
        size=128,
        pixel_spacing=1.0,
        max_episode_steps=max_steps,
        early_termination=False,
        multi_patient=False,
        goal_conditioned=False,
        device="cpu"
    )
    wrapped_env = ImageContactOrientationWrapper(env)

    obs, info = wrapped_env.reset(seed=seed)
    first_success = False
    first_success_step = None

    term_pos_err = None
    term_rot_err = None
    term_step = max_steps
    is_success = False

    for step in range(1, max_steps + 1):
        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = wrapped_env.step(action)

        pos_err_mm = float(info.get("pos_error_mm", 0.0))
        rot_err_deg = float(info.get("rot_error_deg", 0.0))

        if pos_err_mm < 5.0 and rot_err_deg < 10.0:
            if not first_success:
                first_success = True
                first_success_step = step

            if mode == "early_term":
                is_success = True
                term_pos_err = pos_err_mm
                term_rot_err = rot_err_deg
                term_step = step
                break

        if term or trunc:
            term_pos_err = pos_err_mm
            term_rot_err = rot_err_deg
            term_step = step
            break

    wrapped_env.close()

    if mode == "early_term":
        if term_pos_err is None:
            term_pos_err = float(info.get("pos_error_mm", 0.0))
            term_rot_err = float(info.get("rot_error_deg", 0.0))
        return {
            "mode": "early_term",
            "patient": patient,
            "seed": seed,
            "success": is_success,
            "pos_err_mm": term_pos_err,
            "rot_err_deg": term_rot_err,
            "steps": term_step
        }
    else:
        final_pos_err = float(info.get("pos_error_mm", 0.0))
        final_rot_err = float(info.get("rot_error_deg", 0.0))
        final_succ = (final_pos_err < 5.0 and final_rot_err < 10.0)
        return {
            "mode": "fixed",
            "patient": patient,
            "seed": seed,
            "success": final_succ,
            "first_success": first_success,
            "pos_err_mm": final_pos_err,
            "rot_err_deg": final_rot_err,
            "steps": term_step
        }


def evaluate_single_run(setting_key: str, ckpt_path: str, n_workers: int = 4):
    cfg = SETTINGS[setting_key]
    target = cfg["target"]
    sim_mode = cfg["sim_mode"]

    print("\n" + "-" * 70)
    print(f"  EVALUATING: {cfg['name']}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Protocol: 100 trials Train (seeds 1000-1099), 100 trials Held-Out (seeds 2000-2099)")
    print("-" * 70)

    # 1. Prepare tasks for Train (100) & Held-Out (100)
    train_seeds = [1000 + i for i in range(100)]
    held_seeds = [2000 + i for i in range(100)]

    # Distribution across patients
    # Train: 34 s0011, 33 s0058, 33 s0250
    train_patient_map = ["s0011"] * 34 + ["s0058"] * 33 + ["s0250"] * 33
    # Held-Out: 50 s0344, 50 s0358
    held_patient_map = ["s0344"] * 50 + ["s0358"] * 50

    eval_results = {}

    for mode in ["fixed", "early_term"]:
        eval_results[mode] = {}
        for cohort, p_map, seeds in [("Train", train_patient_map, train_seeds), ("Held-Out", held_patient_map, held_seeds)]:
            tasks = [
                (ckpt_path, target, sim_mode, p_map[i], seeds[i], 300, mode)
                for i in range(100)
            ]
            t0 = time.time()
            with mp.Pool(processes=n_workers) as pool:
                res_list = list(pool.imap_unordered(run_eval_trial_worker, tasks))
            el = time.time() - t0

            succ_list = [1.0 if r["success"] else 0.0 for r in res_list]
            pos_list = [r["pos_err_mm"] for r in res_list]
            rot_list = [r["rot_error_deg"] if "rot_error_deg" in r else r["rot_err_deg"] for r in res_list]
            steps_list = [r["steps"] for r in res_list]

            eval_results[mode][cohort] = {
                "success_rate": float(np.mean(succ_list) * 100.0),
                "pos_err_mean": float(np.mean(pos_list)),
                "pos_err_std": float(np.std(pos_list)),
                "rot_err_mean": float(np.mean(rot_list)),
                "rot_err_std": float(np.std(rot_list)),
                "steps_mean": float(np.mean(steps_list)),
                "steps_std": float(np.std(steps_list)),
                "eval_time_sec": el
            }
            if mode == "fixed":
                first_succ = [1.0 if r.get("first_success", False) else 0.0 for r in res_list]
                eval_results[mode][cohort]["first_success_rate"] = float(np.mean(first_succ) * 100.0)

            print(f"  [{mode:10s}] {cohort:8s}: Succ={eval_results[mode][cohort]['success_rate']:.1f}% | "
                  f"Pos={eval_results[mode][cohort]['pos_err_mean']:.2f}±{eval_results[mode][cohort]['pos_err_std']:.2f} mm | "
                  f"Rot={eval_results[mode][cohort]['rot_err_mean']:.2f}±{eval_results[mode][cohort]['rot_err_std']:.2f}° | "
                  f"Steps={eval_results[mode][cohort]['steps_mean']:.1f} ({el:.1f}s)")

    return eval_results


def write_full_report(master_records: dict):
    out_dir = PROJECT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sac_image_only_report.json"
    md_path = out_dir / "sac_image_only_report.md"

    with open(json_path, "w") as f:
        json.dump(master_records, f, indent=2)

    lines = [
        "# Image-Only SAC Navigation Evaluation Report",
        "",
        "Evaluation Protocol: 100 trials per cohort (Train: seeds 1000-1099, Held-Out: seeds 2000-2099, Horizon: 300 steps).",
        "Success criterion: `pos_error < 5.0 mm` and `rot_error < 10.0 deg`.",
        "",
        "| Model / Setting | Mode | Cohort | Pos Error (mm) | Rot Error (deg) | Succ (%) | First-Succ (%) | Steps | ep_rew_mean | Wall Time | Checkpoint |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |"
    ]

    for key, data in master_records.items():
        name = SETTINGS[key]["name"]
        ckpt = data.get("ckpt_path", "N/A")
        wall_time_h = data.get("wall_clock_sec", 0.0) / 3600.0
        wall_str = f"{wall_time_h:.2f}h" if wall_time_h > 0 else "Cached"
        rew_mean = f"{data.get('ep_rew_mean', 0.0):.2f}" if data.get('ep_rew_mean') is not None else "N/A"
        ev = data.get("eval", {})

        for mode, mode_label in [("fixed", "Fixed"), ("early_term", "Early Term")]:
            if mode not in ev:
                continue
            for cohort in ["Train", "Held-Out"]:
                c_data = ev[mode].get(cohort, {})
                pos_str = f"{c_data.get('pos_err_mean', 0.0):.2f} ± {c_data.get('pos_err_std', 0.0):.2f}"
                rot_str = f"{c_data.get('rot_err_mean', 0.0):.2f} ± {c_data.get('rot_err_std', 0.0):.2f}°"
                succ_str = f"{c_data.get('success_rate', 0.0):.1f}%"
                first_succ_str = f"{c_data.get('first_success_rate', 0.0):.1f}%" if mode == "fixed" else "--"
                steps_str = f"{c_data.get('steps_mean', 300.0):.1f}"
                if mode == "early_term" and "steps_std" in c_data:
                    steps_str = f"{c_data['steps_mean']:.1f} ± {c_data['steps_std']:.1f}"

                lines.append(
                    f"| **{name}** | {mode_label} | {cohort} | {pos_str} | {rot_str} | {succ_str} | {first_succ_str} | {steps_str} | {rew_mean} | {wall_str} | `{ckpt}` |"
                )

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[DONE] Saved report to {md_path} and {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Image-Only SAC for SonoSimLite navigation")
    parser.add_argument("--run", type=str, default="all", choices=["all", "spine_mb", "spine_lb", "aorta_mb", "liver_mb"],
                        help="Which setting to run (default: all)")
    parser.add_argument("--timesteps", type=int, default=300000, help="Training timesteps per run (default: 300k)")
    parser.add_argument("--seed", type=int, default=0, help="Training seed (default: 0)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate existing checkpoints")
    parser.add_argument("--n-eval-workers", type=int, default=4, help="Parallel evaluation workers (default: 4)")
    args = parser.parse_args()

    targets_to_run = list(SETTINGS.keys()) if args.run == "all" else [args.run]

    # Load existing master records if any
    json_path = PROJECT_DIR / "results" / "sac_image_only_report.json"
    master_records = {}
    if json_path.exists():
        try:
            with open(json_path) as f:
                master_records = json.load(f)
        except Exception:
            pass

    for key in targets_to_run:
        print(f"\n=======================================================")
        print(f"  PROCESSING SETTING: {key} ({SETTINGS[key]['name']})")
        print(f"=======================================================")

        if args.eval_only:
            ckpt_path = str(SETTINGS[key]["save_dir"] / f"sac_image_only_{key}_final.zip")
            if not os.path.exists(ckpt_path):
                print(f"[SKIP] Checkpoint {ckpt_path} not found for eval-only mode.")
                continue
            wall_clock = 0.0
            ep_rew_mean = None
        else:
            ckpt_path, wall_clock, ep_rew_mean = train_single_run(
                key, timesteps=args.timesteps, seed=args.seed
            )

        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[ERROR] No valid checkpoint for {key}. Stopping pipeline for this setting.")
            continue

        eval_data = evaluate_single_run(key, ckpt_path, n_workers=args.n_eval_workers)

        master_records[key] = {
            "name": SETTINGS[key]["name"],
            "ckpt_path": ckpt_path,
            "wall_clock_sec": wall_clock,
            "ep_rew_mean": ep_rew_mean,
            "eval": eval_data
        }

        # Save incremental report after each setting completes
        write_full_report(master_records)

    print("\n[ALL COMPLETE] Image-Only SAC suite finished successfully.")


if __name__ == "__main__":
    main()
