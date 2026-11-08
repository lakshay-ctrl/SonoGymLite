#!/usr/bin/env python3
"""
scripts/step5b_needle_nav_hit.py
================================
Step 5b: Navigation-to-Needle Hit Test Benchmark in SonoSimLite.

Evaluates the geometric coupling between robotic ultrasound probe navigation and
percutaneous needle targeting across Aorta (supine, anterior) and Liver (prone, posterior).

Protocol & Constraints:
1. Policy Poses:
   - 100 trials per cohort (Train: s0011 [34], s0058 [33], s0250 [33]; Held-Out: s0344 [50], s0358 [50]).
   - Fixed reproducible seeds (Train: 1000-1099, Held-Out: 2000-2099).
   - Fixed horizon of 300 steps, early termination OFF.
   - device="cpu", sim_mode="conv", pixel_spacing=1.0 mm/px, size auto-detected from policy.
2. Needle Trajectory:
   - In-plane needle guide angles: theta in {30°, 45°, 60°}.
   - Needle lengths: L in {100, 150, 200} mm.
   - Planned guide direction taken from goal pose.
   - Applied errors: Final skin-plane position error (dx, dy) and beam-to-skin-normal tilt. Yaw is NOT used.
3. HIT Criteria (single unified definition):
   (a) Needle tip within 5 mm of target center.
   (b) Straight path from skin entry to tip does not cross bone label (bone_label.nii.gz > 0 at 0.5 mm steps).
   (c) Required stroke s <= needle length L.
4. Feasible Cases Policy Hit Rate:
   - Policy hit rate is evaluated ONLY among feasible cases where the reference perfect pose achieves HIT.
   - Reports feasible count (N_feas), policy feasible hits (K_feas), hit rate (%), and 95% Wilson confidence interval.
   - Reports median tip-to-target distance, median position error, and median tilt error.
5. Image-Only Policy Benchmark:
   - Evaluates strict image-only 300k PPO policies for Aorta (seeds 0, 1, 2).
   - Reports each seed and the Mean ± Std across seeds.
   - Liver image-only is marked "Not Evaluated" (since 4c trains liver-centre).
6. Outputs:
   - results/needle_nav_hit.json
   - results/needle_nav_hit.md
   - results/needle_nav_hit.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
os.makedirs("/tmp/matplotlib_cache", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pybullet as p
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import live_unet_demo as demo
from registration import compute_registered_ct_center, load_registration_meta, voxel_to_world
from surface_motion_planner import load_or_create_surface_map
from robotic_us_env import RoboticUltrasoundGymEnv

TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
TEST_PATIENTS = ["s0344", "s0358"]
ALL_PATIENTS = TRAIN_PATIENTS + TEST_PATIENTS

ANGLES_DEG = [30, 45, 60]
NEEDLE_LENGTHS_MM = [100.0, 150.0, 200.0]

TRAIN_SEED_BASE = 1000
TEST_SEED_BASE = 2000

CACHE_FILE = PROJECT_ROOT / "results" / "step5b_nav_poses_cache.json"


# ==============================================================================
# Statistical Utilities: Wilson Score Interval
# ==============================================================================
def compute_wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Computes Wilson score interval for binomial proportion k / n."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat = k / float(n)
    z = 1.959963984540054  # 95% two-sided normal quantile
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat)) / n + (z ** 2) / (4.0 * (n ** 2)))) / denom
    lower = max(0.0, (center - margin)) * 100.0
    upper = min(1.0, (center + margin)) * 100.0
    return p_hat * 100.0, lower, upper


# ==============================================================================
# Vector Math Utilities: Tilt Rotation (No Yaw)
# ==============================================================================
def compute_tilt_rotation_matrix(v_source: np.ndarray, v_target: np.ndarray) -> np.ndarray:
    """
    Computes minimum rotation matrix R rotating v_source into v_target.
    Applies pure beam-to-normal tilt with zero yaw around the beam axis.
    """
    v1 = v_source / (np.linalg.norm(v_source) + 1e-8)
    v2 = v_target / (np.linalg.norm(v_target) + 1e-8)

    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if dot > 0.999999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.999999:
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(v1[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(v1, ortho)
        axis /= np.linalg.norm(axis)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    axis = np.cross(v1, v2)
    axis_norm = np.linalg.norm(axis)
    axis /= axis_norm

    angle = math.acos(dot)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ], dtype=np.float64)

    R = np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    return R


# ==============================================================================
# Patient Data Preloading & Anatomy Caching
# ==============================================================================
class PatientAnatomyData:
    """Caches CT volumes, bone label masks, surface maps, and target locations for a patient."""
    def __init__(self, patient_id: str, anatomy: str, bed_top_z: float = 0.53):
        self.patient_id = patient_id
        self.anatomy = anatomy
        self.subject_dir = PROJECT_ROOT / "totalseg_patients" / patient_id
        self.meta = load_registration_meta(self.subject_dir / "registration_meta.json")
        self.mesh_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

        self.orientation = "supine" if anatomy == "aorta" else "prone"
        mesh_min, _ = demo.read_obj_bounds(self.subject_dir / "patient_skin.obj")

        if self.orientation == "supine":
            body_quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
            body_rot = np.array(p.getMatrixFromQuaternion(body_quat)).reshape(3, 3)
            body_z = bed_top_z - mesh_min[1] + 0.01
        else:
            body_quat = p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
            body_rot = np.array(p.getMatrixFromQuaternion(body_quat)).reshape(3, 3)
            body_z = bed_top_z - mesh_min[1] + 0.01

        self.body_pos = np.array([float(demo.BED_CENTER[0]), float(demo.BED_CENTER[1]), float(body_z)], dtype=np.float64)
        self.body_rot = body_rot

        self.surface_planner = load_or_create_surface_map(
            self.subject_dir, bed_top_z=bed_top_z, mesh_scale=self.mesh_scale, orientation=self.orientation
        )

        if anatomy == "aorta":
            target_json_path = PROJECT_ROOT / "aorta_bifurcation_targets.json"
            target_key = "voxel_bifurcation_zyx"
        else:
            target_json_path = PROJECT_ROOT / "portal_vein_targets.json"
            target_key = "voxel_centroid_zyx"

        with open(target_json_path, "r") as f:
            targets_dict = json.load(f)

        self.voxel_target = targets_dict[patient_id][target_key]
        self.world_target_pos = voxel_to_world(
            self.voxel_target, self.body_pos, self.body_rot,
            self.meta["affine"], self.meta["mesh_centering_offset"], mesh_scale=self.mesh_scale
        )

        skin_z, normal, is_valid = self.surface_planner.query(
            float(self.world_target_pos[0]), float(self.world_target_pos[1])
        )
        if not is_valid:
            skin_z = float(self.world_target_pos[2] + 0.10)
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self.skin_z = skin_z
        self.target_normal = normal
        self.target_depth_mm = float(abs(skin_z - self.world_target_pos[2]) * 1000.0)

        self.perfect_x = float(self.world_target_pos[0])
        self.perfect_y = float(self.world_target_pos[1])
        self.perfect_normal = normal
        self.nominal_beam = -normal

        self.bone_label_data = nib.load(str(self.subject_dir / "bone_label.nii.gz")).get_fdata()
        self.ct_shape = self.bone_label_data.shape


# ==============================================================================
# Needle Trajectory & Hit-Test Computation
# ==============================================================================
def evaluate_needle_from_pose(
    probe_x: float,
    probe_y: float,
    probe_beam_normal: np.ndarray,
    patient_data: PatientAnatomyData,
    step_mm: float = 0.50
) -> dict:
    """
    Evaluates needle insertion at 30°, 45°, 60° applying ONLY:
      1. Position error on skin plane: (dx, dy)
      2. Beam-to-skin-normal tilt error (Rodrigues rotation, NO YAW).
    """
    skin_z, normal, is_valid = patient_data.surface_planner.query(probe_x, probe_y)
    if not is_valid:
        skin_z = patient_data.skin_z
        normal = patient_data.target_normal

    target_pos_world = patient_data.world_target_pos
    target_depth_mm = patient_data.target_depth_mm
    d_m = target_depth_mm / 1000.0

    p_probe_target = np.array([target_pos_world[0], target_pos_world[1], patient_data.skin_z], dtype=np.float64)

    v_nom_beam = patient_data.nominal_beam
    v_actual_beam = probe_beam_normal / (np.linalg.norm(probe_beam_normal) + 1e-8)
    R_tilt = compute_tilt_rotation_matrix(v_nom_beam, v_actual_beam)

    angle_evals = {}

    for ang in ANGLES_DEG:
        ang_rad = math.radians(ang)
        req_stroke_mm = target_depth_mm / math.sin(ang_rad)
        lat_offset_m = d_m / math.tan(ang_rad)

        p_entry_nom = p_probe_target + np.array([lat_offset_m, 0.0, 0.0], dtype=np.float64)
        u_needle_nom = (target_pos_world - p_entry_nom) / np.linalg.norm(target_pos_world - p_entry_nom)

        p_entry = np.array([probe_x + lat_offset_m, probe_y, skin_z], dtype=np.float64)
        u_needle = R_tilt @ u_needle_nom
        u_needle /= np.linalg.norm(u_needle)

        cannula_evals = {}
        for L in NEEDLE_LENGTHS_MM:
            stroke_feasible = (req_stroke_mm <= L)
            phys_stroke_mm = min(req_stroke_mm, L)
            p_phys_tip = p_entry + (phys_stroke_mm / 1000.0) * u_needle

            dist_tip_to_target_mm = float(np.linalg.norm(p_phys_tip - target_pos_world) * 1000.0)
            tip_within_5mm = (dist_tip_to_target_mm <= 5.0)

            dist_m = float(phys_stroke_mm / 1000.0)
            n_steps = max(1, int(math.ceil(dist_m / (step_mm / 1000.0))))
            bone_hit = False
            for k in range(n_steps + 1):
                pt_k = p_entry + (k * (step_mm / 1000.0)) * u_needle
                vox_k = compute_registered_ct_center(
                    pt_k, patient_data.body_pos, patient_data.body_rot,
                    patient_data.meta["inv_affine"], patient_data.meta["mesh_centering_offset"],
                    patient_data.ct_shape, 1.0
                )
                kx, ky, kz = int(round(vox_k[0])), int(round(vox_k[1])), int(round(vox_k[2]))
                if 0 <= kx < patient_data.ct_shape[0] and 0 <= ky < patient_data.ct_shape[1] and 0 <= kz < patient_data.ct_shape[2]:
                    if patient_data.bone_label_data[kx, ky, kz] > 0:
                        bone_hit = True
                        break

            path_bone_free = not bone_hit
            is_hit = bool(tip_within_5mm and path_bone_free and stroke_feasible)

            cannula_evals[str(int(L))] = {
                "needle_length_mm": L,
                "stroke_feasible": stroke_feasible,
                "tip_within_5mm": tip_within_5mm,
                "path_bone_free": path_bone_free,
                "is_hit": is_hit,
                "dist_tip_to_target_mm": dist_tip_to_target_mm
            }

        angle_evals[str(ang)] = {
            "angle_deg": ang,
            "req_stroke_mm": req_stroke_mm,
            "cannula_evals": cannula_evals
        }

    return angle_evals


# ==============================================================================
# Deterministic Perfect-Pose Reference Evaluator (Per Patient)
# ==============================================================================
def evaluate_perfect_reference_per_patient(patient_data_dict: dict) -> dict:
    perf_per_patient = {}
    for pid, pdata in patient_data_dict.items():
        perf_eval = evaluate_needle_from_pose(
            pdata.perfect_x, pdata.perfect_y, pdata.nominal_beam, pdata
        )
        perf_per_patient[pid] = {
            "patient": pid,
            "target_depth_mm": pdata.target_depth_mm,
            "angles": {}
        }
        for ang in ANGLES_DEG:
            perf_per_patient[pid]["angles"][str(ang)] = {
                "req_stroke_mm": perf_eval[str(ang)]["req_stroke_mm"],
                "results_by_length": {}
            }
            for L in NEEDLE_LENGTHS_MM:
                c = perf_eval[str(ang)]["cannula_evals"][str(int(L))]
                perf_per_patient[pid]["angles"][str(ang)]["results_by_length"][str(int(L))] = {
                    "is_hit": c["is_hit"],
                    "verdict": "HIT" if c["is_hit"] else "MISS",
                    "dist_mm": c["dist_tip_to_target_mm"],
                    "bone_free": c["path_bone_free"],
                    "stroke_feasible": c["stroke_feasible"]
                }
    return perf_per_patient


# ==============================================================================
# Parallel Navigation Episode Worker
# ==============================================================================
def run_single_nav_episode_worker(task: tuple) -> dict:
    """
    Executes a single 300-step navigation episode on CPU without early termination.
    task = (policy_id, ckpt_path, target, cohort, pid, seed, fixed_horizon, is_image_only, img_size)
    """
    policy_id, ckpt_path, target, cohort, pid, seed, fixed_horizon, is_image_only, img_size = task
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    subj_dir = PROJECT_ROOT / "totalseg_patients" / pid

    env = RoboticUltrasoundGymEnv(
        subject_dir=str(subj_dir),
        device="cpu",
        render_mode="rgb_array",
        max_episode_steps=fixed_horizon,
        task="navigation",
        anatomy=target,
        sim_mode="conv",
        size=img_size,
        pixel_spacing=1.0,
        multi_patient=False,
        goal_conditioned=not is_image_only
    )

    model = PPO.load(ckpt_path, device="cpu")

    obs, info = env.reset(seed=seed)

    for _ in range(fixed_horizon):
        if is_image_only:
            c_val = 1.0 if obs.get("force", [0.0])[0] * 10.0 >= 1.0 else 0.0
            quat = np.array(env.current_probe_quat, dtype=np.float32)
            obs["contact"] = np.array([c_val], dtype=np.float32)
            obs["orientation"] = quat

        m_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(m_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)

    final_probe_x = float(env.current_surface_x)
    final_probe_y = float(env.current_surface_y)
    final_probe_quat = env.current_probe_quat.tolist()
    pos_err_mm = float(info.get("pos_error_mm", 0.0))
    rot_err_deg = float(info.get("rot_error_deg", 0.0))

    R_probe = np.array(p.getMatrixFromQuaternion(final_probe_quat)).reshape(3, 3)
    final_probe_beam = (-R_probe[:, 2]).tolist()

    env.close()

    return {
        "policy_id": policy_id,
        "target": target,
        "cohort": cohort,
        "patient": pid,
        "seed": seed,
        "final_probe_x": final_probe_x,
        "final_probe_y": final_probe_y,
        "final_probe_quat": final_probe_quat,
        "final_probe_beam": final_probe_beam,
        "pos_error_mm": pos_err_mm,
        "rot_error_deg": rot_err_deg
    }


# ==============================================================================
# Cohort Rollout Manager with Multi-Process Pool & Cache
# ==============================================================================
def collect_cohort_poses(
    policy_id: str,
    ckpt_path: str,
    target: str,
    cohort: str,
    patients: list[str],
    seed_base: int,
    total_trials: int = 100,
    fixed_horizon: int = 300,
    workers: int = 8
) -> list[dict]:
    """Runs 100 navigation trials in parallel across cohort with disk caching."""
    from stable_baselines3 import PPO

    cache = {}
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    # Inspect policy observation space to configure image size and image-only flag
    dummy_model = PPO.load(ckpt_path, device="cpu")
    is_image_only = not ("pose" in dummy_model.observation_space.spaces)
    img_size = int(dummy_model.observation_space.spaces["image"].shape[0])
    del dummy_model

    n_patients = len(patients)
    trials_per_patient = [total_trials // n_patients] * n_patients
    for rem in range(total_trials % n_patients):
        trials_per_patient[rem] += 1

    tasks = []
    trial_idx = 0
    for p_i, pid in enumerate(patients):
        num_trials_for_pat = trials_per_patient[p_i]
        for _ in range(num_trials_for_pat):
            current_seed = seed_base + trial_idx
            cache_key = f"{policy_id}_{target}_{cohort}_{pid}_{current_seed}_{fixed_horizon}"
            if cache_key not in cache:
                tasks.append((
                    policy_id, ckpt_path, target, cohort, pid, current_seed,
                    fixed_horizon, is_image_only, img_size
                ))
            trial_idx += 1

    if tasks:
        print(f"  Launching {len(tasks)} parallel trials for [{policy_id}] {target.upper()} {cohort.upper()} using {workers} workers...")
        t0 = time.time()
        with mp.Pool(processes=workers) as pool:
            for i, res in enumerate(pool.imap_unordered(run_single_nav_episode_worker, tasks), 1):
                cache_key = f"{res['policy_id']}_{res['target']}_{res['cohort']}_{res['patient']}_{res['seed']}_{fixed_horizon}"
                cache[cache_key] = res
                if i % 25 == 0 or i == len(tasks):
                    print(f"    Completed [{i}/{len(tasks)}] trials ({time.time() - t0:.1f}s)")

        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)

    # Gather in deterministic seed order
    ordered_results = []
    trial_idx = 0
    for p_i, pid in enumerate(patients):
        num_trials_for_pat = trials_per_patient[p_i]
        for _ in range(num_trials_for_pat):
            current_seed = seed_base + trial_idx
            cache_key = f"{policy_id}_{target}_{cohort}_{pid}_{current_seed}_{fixed_horizon}"
            ordered_results.append(cache[cache_key])
            trial_idx += 1

    return ordered_results


# ==============================================================================
# Metric Aggregation (Feasible-Only Hit Rate)
# ==============================================================================
def aggregate_cell_metrics(
    trial_poses: list[dict],
    perf_by_patient: dict,
    theta: int,
    length_mm: float,
    patient_data_dict: dict
) -> dict:
    """
    Evaluates needle targeting and aggregates metrics for a single cell:
      - Perfect pose: HIT/MISS per patient.
      - Policy hit rate ONLY among feasible cases (perfect pose = HIT).
      - Wilson 95% confidence interval on feasible cases.
      - Median tip-to-target distance (mm).
      - Median position and tilt errors of policy poses.
    """
    th_key = str(theta)
    l_key = str(int(length_mm))

    # Evaluate needle for each trial pose
    evaluated_trials = []
    for t in trial_poses:
        pid = t["patient"]
        pdata = patient_data_dict[pid]
        v_beam = np.array(t["final_probe_beam"], dtype=np.float64)
        n_eval = evaluate_needle_from_pose(t["final_probe_x"], t["final_probe_y"], v_beam, pdata)
        evaluated_trials.append({
            "trial": t,
            "eval": n_eval[th_key]["cannula_evals"][l_key]
        })

    # Cohort patients
    pids = sorted(list({t["patient"] for t in trial_poses}))
    perf_patient_statuses = {
        pid: perf_by_patient[pid]["angles"][th_key]["results_by_length"][l_key]["verdict"]
        for pid in pids
    }
    perf_hits_count = sum(1 for v in perf_patient_statuses.values() if v == "HIT")

    # Filter trials where perfect pose is a HIT (FEASIBLE CASES)
    feasible_trials = [
        et for et in evaluated_trials
        if perf_by_patient[et["trial"]["patient"]]["angles"][th_key]["results_by_length"][l_key]["is_hit"]
    ]

    n_feasible = len(feasible_trials)
    feasible_hits = sum(1 for et in feasible_trials if et["eval"]["is_hit"])

    if n_feasible > 0:
        feas_rate, ci_low, ci_high = compute_wilson_interval(feasible_hits, n_feasible)
    else:
        feas_rate, ci_low, ci_high = None, None, None

    # Overall metrics across all trials
    all_hits = sum(1 for et in evaluated_trials if et["eval"]["is_hit"])
    all_tip_dists = [et["eval"]["dist_tip_to_target_mm"] for et in evaluated_trials]
    pos_errors = [et["trial"]["pos_error_mm"] for et in evaluated_trials]
    rot_errors = [et["trial"]["rot_error_deg"] for et in evaluated_trials]

    return {
        "n_total_trials": len(trial_poses),
        "perfect": {
            "per_patient": perf_patient_statuses,
            "hits_count": perf_hits_count,
            "total_patients": len(pids)
        },
        "feasible": {
            "n_feasible_cases": n_feasible,
            "feasible_hits": feasible_hits,
            "hit_rate_pct": feas_rate,
            "ci_95": [ci_low, ci_high] if feas_rate is not None else None
        },
        "overall": {
            "hits": all_hits,
            "hit_rate_pct": (all_hits / float(len(trial_poses))) * 100.0 if trial_poses else 0.0,
            "median_tip_dist_mm": float(np.median(all_tip_dists)) if all_tip_dists else 0.0,
            "median_pos_error_mm": float(np.median(pos_errors)) if pos_errors else 0.0,
            "median_tilt_error_deg": float(np.median(rot_errors)) if rot_errors else 0.0
        }
    }


# ==============================================================================
# Markdown Report Generator
# ==============================================================================
def generate_step5b_markdown_report(results: dict) -> str:
    lines = [
        "# Step 5b: Navigation-to-Needle Hit Test Benchmark",
        "",
        "## Setup & Evaluation Protocol",
        "- **Goal-Conditioned Checkpoints:**",
        "  - Aorta: `aorta_checkpoints_ppo/ppo_final_model.zip` (sim_mode: `conv`, size: 256x256, pixel_spacing: 1.0 mm/px)",
        "  - Liver: `ppo_checkpoints/ppo_final_model.zip` (sim_mode: `conv`, size: 128x128, pixel_spacing: 1.0 mm/px)",
        "  - Liver (Step 3 MB Benchmark): `ppo_checkpoints/liver_mb/ppo_mb_liver_final_model.zip` (size: 256x256, pixel_spacing: 1.0 mm/px)",
        "- **Image-Only 300k Checkpoints (Aorta):**",
        "  - Seeds 0, 1, 2 from `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s{0,1,2}_final.zip` (size: 128x128, pixel_spacing: 1.0 mm/px)",
        "- **Liver Image-Only:** Marked *Not Evaluated* (Step 4c trains liver-centre landmark, not portal vein hilum).",
        "- **Protocol:** 100 trials per cohort (Train seeds 1000-1099, Held-Out seeds 2000-2099), fixed horizon = 300 steps, early termination OFF.",
        "- **Needle Hit Definition:**",
        "  1. Tip within 5 mm of target center ($\\|p_{\\text{tip}} - p_{\\text{target}}\\| \\le 5.0\\,\\text{mm}$).",
        "  2. Straight path from skin entry to tip does not cross bone (`bone_label.nii.gz` > 0 at 0.5 mm steps).",
        "  3. Required stroke $s \\le L$ (needle length $L \\in \\{100, 150, 200\\}\\,\\text{mm}$).",
        "- **Policy Poses & Errors:** Final pose after 300 steps. In-plane needle direction kept at planned guide direction; applied only skin-plane position error $(dx, dy)$ and beam-to-skin-normal tilt (Rodrigues rotation, no yaw).",
        "- **Feasible Hit Rate:** Evaluated strictly among trials where the reference perfect pose achieves HIT.",
        "",
        "---",
        "",
        "## Reference Ceiling: Perfect Pose Hit/Miss per Patient",
        "",
        "| Target | Patient | $\\theta$ | Required Stroke (mm) | 100 mm Needle Length | 150 mm Needle Length | 200 mm Needle Length |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for tgt in ["aorta", "liver"]:
        perf_data = results["perfect_reference_per_patient"][tgt]
        target_name = "Aorta Bifurcation" if tgt == "aorta" else "Liver Portal Vein"
        for pid in ALL_PATIENTS:
            for ang in ANGLES_DEG:
                p_ang = perf_data[pid]["angles"][str(ang)]
                s_req = p_ang["req_stroke_mm"]
                r100 = p_ang["results_by_length"]["100"]["verdict"]
                r150 = p_ang["results_by_length"]["150"]["verdict"]
                r200 = p_ang["results_by_length"]["200"]["verdict"]
                lines.append(
                    f"| {target_name} | `{pid}` | {ang}° | {s_req:.1f} mm | {r100} | {r150} | {r200} |"
                )

    lines.extend([
        "",
        "---",
        "",
        "## Policy Navigation Hit Rates Benchmark Table",
        "",
        "| Target | Cohort | Policy Type | $\\theta$ | Needle Length | Perfect Pose (Hits/Total) | Feasible Cases ($N_{\\text{feas}}$) | Feasible Hit Rate (%) [95% CI] | Median Tip Dist (mm) | Median Pos Error (mm) | Median Tilt Error (deg) |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ])

    for row in results["table_rows"]:
        tgt_name = row["target"]
        coh_name = row["cohort"]
        pol_type = row["policy_type"]
        ang = row["theta"]
        L = row["needle_length_mm"]
        perf_str = row["perfect_str"]
        n_feas = row["n_feasible"]
        feas_rate_str = row["feas_rate_str"]
        tip_dist_str = row["median_tip_dist_str"]
        pos_err_str = row["median_pos_err_str"]
        tilt_err_str = row["median_tilt_err_str"]

        lines.append(
            f"| {tgt_name} | {coh_name} | {pol_type} | {ang}° | {int(L)} mm | {perf_str} | {n_feas} | {feas_rate_str} | {tip_dist_str} | {pos_err_str} | {tilt_err_str} |"
        )

    return "\n".join(lines)


# ==============================================================================
# Vector PDF Figure Generation
# ==============================================================================
def render_step5b_pdf(results: dict, output_pdf_path: Path):
    """Renders results/needle_nav_hit.pdf multi-panel vector plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.5), dpi=300)
    colors = {30: "#1E88E5", 45: "#FB8C00", 60: "#43A047"}

    panels = [
        (axes[0], "aorta", "Aorta Bifurcation (Supine Anterior)"),
        (axes[1], "liver", "Liver Portal Vein (Prone Posterior)")
    ]

    for ax, tgt, title in panels:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

        for ang in ANGLES_DEG:
            for coh, marker_p, marker_l, coh_label in [
                ("train", "o", "s", "Train"),
                ("held_out", "^", "D", "Held-Out")
            ]:
                # Extract curves from table rows for goal-conditioned
                lengths = NEEDLE_LENGTHS_MM
                pol_rates = []
                perf_rates = []

                for L in lengths:
                    matching = [
                        r for r in results["table_rows"]
                        if r["target_key"] == tgt and r["cohort_key"] == coh
                        and r["policy_type_key"] == "gc_ppo"
                        and r["theta"] == ang and r["needle_length_mm"] == L
                    ]
                    if matching:
                        m = matching[0]
                        pol_rates.append(m["feasible_hit_rate_pct"] if m["feasible_hit_rate_pct"] is not None else 0.0)
                        perf_rates.append((m["perfect_hits"] / float(m["perfect_total"])) * 100.0)
                    else:
                        pol_rates.append(0.0)
                        perf_rates.append(0.0)

                # Reference Perfect Pose: Dashed line
                ax.plot(
                    lengths, perf_rates,
                    linestyle="--", marker=marker_p, markersize=5,
                    color=colors[ang], alpha=0.65, linewidth=1.5,
                    label=f"Perf {ang}° ({coh_label})"
                )

                # Goal-Conditioned Policy: Solid line
                ax.plot(
                    lengths, pol_rates,
                    linestyle="-", marker=marker_l, markersize=6,
                    color=colors[ang], alpha=0.95, linewidth=2.0,
                    label=f"GC-PPO {ang}° ({coh_label})"
                )

        ax.set_ylim(-5, 105)
        ax.set_xlim(85, 215)
        ax.set_xticks(NEEDLE_LENGTHS_MM)
        ax.set_xticklabels(["100 mm", "150 mm", "200 mm"], fontsize=10)
        ax.set_ylabel("Feasible Hit Rate (%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Needle Length (mm)", fontsize=11, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=6.8, ncol=2, framealpha=0.92)

    plt.tight_layout()
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_pdf_path), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f" [OK] Vector PDF plot saved to: {output_pdf_path}")


# ==============================================================================
# Main CLI Dispatcher
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="SonoSimLite Step 5b Navigation-to-Needle Hit Test Benchmark")
    parser.add_argument("--model-aorta", type=str, default="aorta_checkpoints_ppo/ppo_final_model.zip", help="Path to trained PPO aorta checkpoint")
    parser.add_argument("--model-liver", type=str, default="ppo_checkpoints/ppo_final_model.zip", help="Path to trained PPO liver checkpoint")
    parser.add_argument("--trials-per-cohort", type=int, default=100, help="Number of navigation trials per cohort")
    parser.add_argument("--fixed-horizon", type=int, default=300, help="Fixed horizon steps per episode (default: 300)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes")
    parser.add_argument("--output-json", type=str, default="results/needle_nav_hit.json", help="Path for JSON benchmark")
    parser.add_argument("--output-md", type=str, default="results/needle_nav_hit.md", help="Path for Markdown report")
    parser.add_argument("--output-pdf", type=str, default="results/needle_nav_hit.pdf", help="Path for PDF plot")
    args = parser.parse_args()

    print("=" * 85)
    print(" SONOSIMLITE: STEP 5b NAVIGATION-TO-NEEDLE HIT TEST BENCHMARK")
    print(f" Aorta Model: {args.model_aorta}")
    print(f" Liver Model: {args.model_liver}")
    print(f" Trials per Cohort: {args.trials_per_cohort} (Horizon = {args.fixed_horizon} steps, Early Termination OFF)")
    print(f" Parallel Workers: {args.workers}")
    print("=" * 85)

    start_time = time.time()

    # Preload Patient Anatomies
    print("\nPreloading Patient Anatomies & CT Bone Volumes...")
    patient_data = {
        "aorta": {p: PatientAnatomyData(p, "aorta") for p in ALL_PATIENTS},
        "liver": {p: PatientAnatomyData(p, "liver") for p in ALL_PATIENTS}
    }
    print("Preloading completed.\n")

    # 1. Evaluate Perfect Reference Pose per patient
    print("Evaluating Perfect-Pose Reference Ceiling across all patients...")
    perfect_reference = {
        "aorta": evaluate_perfect_reference_per_patient(patient_data["aorta"]),
        "liver": evaluate_perfect_reference_per_patient(patient_data["liver"])
    }

    # 2. Collect Navigation Trial Poses
    # Policies dictionary
    policies_to_run = {
        ("aorta", "gc_ppo", "Goal-Conditioned PPO"): args.model_aorta,
        ("aorta", "io_ppo_s0", "Image-Only PPO (Seed 0)"): "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip",
        ("aorta", "io_ppo_s1", "Image-Only PPO (Seed 1)"): "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip",
        ("aorta", "io_ppo_s2", "Image-Only PPO (Seed 2)"): "ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip",
        ("liver", "gc_ppo", "Goal-Conditioned PPO"): args.model_liver,
    }

    # Check if Step 3 liver checkpoint is also available
    step3_liver_ckpt = "ppo_checkpoints/liver_mb/ppo_mb_liver_final_model.zip"
    if os.path.exists(step3_liver_ckpt) and step3_liver_ckpt != args.model_liver:
        policies_to_run[("liver", "gc_ppo_step3", "Goal-Conditioned PPO (Step 3 MB)")] = step3_liver_ckpt

    trial_poses_db = {}

    for (target, pol_key, pol_label), ckpt_path in policies_to_run.items():
        if not os.path.exists(ckpt_path):
            print(f"[WARN] Checkpoint {ckpt_path} not found. Skipping {pol_label}.")
            continue

        trial_poses_db[(target, pol_key)] = {"train": [], "held_out": []}
        for coh_name, pids, s_base in [("train", TRAIN_PATIENTS, TRAIN_SEED_BASE), ("held_out", TEST_PATIENTS, TEST_SEED_BASE)]:
            poses = collect_cohort_poses(
                policy_id=pol_key,
                ckpt_path=ckpt_path,
                target=target,
                cohort=coh_name,
                patients=pids,
                seed_base=s_base,
                total_trials=args.trials_per_cohort,
                fixed_horizon=args.fixed_horizon,
                workers=args.workers
            )
            trial_poses_db[(target, pol_key)][coh_name] = poses

    # 3. Evaluate Needle Trajectories and Compute Metrics
    table_rows = []
    summary_data = {}

    for (target, pol_key, pol_label), ckpt_path in policies_to_run.items():
        if (target, pol_key) not in trial_poses_db:
            continue
        tgt_name = "Aorta Bifurcation" if target == "aorta" else "Liver Portal Vein"
        summary_data[pol_key] = {"train": {}, "held_out": {}}

        for coh_name in ["train", "held_out"]:
            coh_label = "In-Domain Train" if coh_name == "train" else "Held-Out Test"
            trials = trial_poses_db[(target, pol_key)][coh_name]
            pdata_dict = patient_data[target]

            summary_data[pol_key][coh_name] = {}

            for ang in ANGLES_DEG:
                summary_data[pol_key][coh_name][str(ang)] = {}
                for L in NEEDLE_LENGTHS_MM:
                    cell_metric = aggregate_cell_metrics(
                        trial_poses=trials,
                        perf_by_patient=perfect_reference[target],
                        theta=ang,
                        length_mm=L,
                        patient_data_dict=pdata_dict
                    )
                    summary_data[pol_key][coh_name][str(ang)][str(int(L))] = cell_metric

                    perf = cell_metric["perfect"]
                    perf_str = f"{perf['hits_count']}/{perf['total_patients']}"
                    feas = cell_metric["feasible"]
                    over = cell_metric["overall"]

                    if feas["hit_rate_pct"] is not None:
                        feas_str = f"{feas['hit_rate_pct']:.1f}% [{feas['ci_95'][0]:.1f}, {feas['ci_95'][1]:.1f}] ({feas['feasible_hits']}/{feas['n_feasible_cases']})"
                    else:
                        feas_str = "N/A (0 feasible)"

                    table_rows.append({
                        "target": tgt_name,
                        "target_key": target,
                        "cohort": coh_label,
                        "cohort_key": coh_name,
                        "policy_type": pol_label,
                        "policy_type_key": pol_key,
                        "theta": ang,
                        "needle_length_mm": L,
                        "perfect_str": perf_str,
                        "perfect_hits": perf["hits_count"],
                        "perfect_total": perf["total_patients"],
                        "n_feasible": feas["n_feasible_cases"],
                        "feasible_hits": feas["feasible_hits"],
                        "feasible_hit_rate_pct": feas["hit_rate_pct"],
                        "feasible_ci_95": feas["ci_95"],
                        "feas_rate_str": feas_str,
                        "median_tip_dist_str": f"{over['median_tip_dist_mm']:.1f} mm",
                        "median_tip_dist_mm": over["median_tip_dist_mm"],
                        "median_pos_err_str": f"{over['median_pos_error_mm']:.1f} mm",
                        "median_pos_err_mm": over["median_pos_error_mm"],
                        "median_tilt_err_str": f"{over['median_tilt_error_deg']:.1f}°",
                        "median_tilt_err_deg": over["median_tilt_error_deg"]
                    })

    # Add Image-Only Mean ± Std rows for Aorta
    for coh_name in ["train", "held_out"]:
        coh_label = "In-Domain Train" if coh_name == "train" else "Held-Out Test"
        for ang in ANGLES_DEG:
            for L in NEEDLE_LENGTHS_MM:
                s_matching = [
                    r for r in table_rows
                    if r["target_key"] == "aorta" and r["cohort_key"] == coh_name
                    and r["theta"] == ang and r["needle_length_mm"] == L
                    and r["policy_type_key"] in ["io_ppo_s0", "io_ppo_s1", "io_ppo_s2"]
                ]
                if len(s_matching) == 3:
                    f_rates = [m["feasible_hit_rate_pct"] for m in s_matching if m["feasible_hit_rate_pct"] is not None]
                    dists = [m["median_tip_dist_mm"] for m in s_matching]
                    pos_errs = [m["median_pos_err_mm"] for m in s_matching]
                    tilt_errs = [m["median_tilt_err_deg"] for m in s_matching]

                    if f_rates:
                        mean_f = np.mean(f_rates)
                        std_f = np.std(f_rates)
                        feas_str = f"**{mean_f:.1f} ± {std_f:.1f}%**"
                    else:
                        feas_str = "N/A"

                    table_rows.append({
                        "target": "Aorta Bifurcation",
                        "target_key": "aorta",
                        "cohort": coh_label,
                        "cohort_key": coh_name,
                        "policy_type": "**Image-Only PPO (Mean ± Std)**",
                        "policy_type_key": "io_ppo_mean",
                        "theta": ang,
                        "needle_length_mm": L,
                        "perfect_str": s_matching[0]["perfect_str"],
                        "perfect_hits": s_matching[0]["perfect_hits"],
                        "perfect_total": s_matching[0]["perfect_total"],
                        "n_feasible": s_matching[0]["n_feasible"],
                        "feasible_hits": int(np.mean([m["feasible_hits"] for m in s_matching])),
                        "feasible_hit_rate_pct": float(np.mean(f_rates)) if f_rates else None,
                        "feasible_ci_95": None,
                        "feas_rate_str": feas_str,
                        "median_tip_dist_str": f"{np.mean(dists):.1f} ± {np.std(dists):.1f} mm",
                        "median_tip_dist_mm": float(np.mean(dists)),
                        "median_pos_err_str": f"{np.mean(pos_errs):.1f} ± {np.std(pos_errs):.1f} mm",
                        "median_pos_err_mm": float(np.mean(pos_errs)),
                        "median_tilt_err_str": f"{np.mean(tilt_errs):.1f} ± {np.std(tilt_errs):.1f}°",
                        "median_tilt_err_deg": float(np.mean(tilt_errs))
                    })

    # Add Liver Image-Only "Not Evaluated" rows
    for coh_name in ["train", "held_out"]:
        coh_label = "In-Domain Train" if coh_name == "train" else "Held-Out Test"
        for ang in ANGLES_DEG:
            for L in NEEDLE_LENGTHS_MM:
                table_rows.append({
                    "target": "Liver Portal Vein",
                    "target_key": "liver",
                    "cohort": coh_label,
                    "cohort_key": coh_name,
                    "policy_type": "*Image-Only PPO*",
                    "policy_type_key": "io_ppo_liver",
                    "theta": ang,
                    "needle_length_mm": L,
                    "perfect_str": "-",
                    "perfect_hits": 0,
                    "perfect_total": 0,
                    "n_feasible": "-",
                    "feasible_hits": 0,
                    "feasible_hit_rate_pct": None,
                    "feasible_ci_95": None,
                    "feas_rate_str": "*Not Evaluated*",
                    "median_tip_dist_str": "*Not Evaluated*",
                    "median_tip_dist_mm": None,
                    "median_pos_err_str": "*Not Evaluated*",
                    "median_pos_err_mm": None,
                    "median_tilt_err_str": "*Not Evaluated*",
                    "median_tilt_err_deg": None
                })

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "settings": {
            "trials_per_cohort": args.trials_per_cohort,
            "train_seeds": f"{TRAIN_SEED_BASE}-{TRAIN_SEED_BASE + args.trials_per_cohort - 1}",
            "held_out_seeds": f"{TEST_SEED_BASE}-{TEST_SEED_BASE + args.trials_per_cohort - 1}",
            "fixed_horizon": args.fixed_horizon,
            "early_termination": "OFF",
            "guide_angles_deg": ANGLES_DEG,
            "needle_lengths_mm": NEEDLE_LENGTHS_MM,
            "applied_errors": ["skin_position_dx_dy", "beam_to_skin_normal_tilt"],
            "yaw_used": False
        },
        "perfect_reference_per_patient": perfect_reference,
        "table_rows": table_rows,
        "summary": summary_data
    }

    # Save Outputs
    out_json = PROJECT_ROOT / args.output_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n[OK] Results JSON saved to: {out_json}")

    out_md = PROJECT_ROOT / args.output_md
    md_content = generate_step5b_markdown_report(output_data)
    with open(out_md, "w") as f:
        f.write(md_content)
    print(f"[OK] Markdown report saved to: {out_md}")

    out_pdf = PROJECT_ROOT / args.output_pdf
    render_step5b_pdf(output_data, out_pdf)

    total_time = time.time() - start_time
    print(f"\nBenchmark completed successfully in {total_time:.2f} seconds.")


if __name__ == "__main__":
    main()
