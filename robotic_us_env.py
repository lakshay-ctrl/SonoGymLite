"""
robotic_us_env.py — High-Performance Gymnasium Environment for Robotic Ultrasound.
==================================================================================

Benchmarked directly against:
  1. SonoGym (NeurIPS 2025): Autonomous Navigation (Task 1) & Surface Reconstruction (Task 2).
  2. Ultrasound Foundation Model Quality Metrics (Huang et al., MICCAI 2025).

Supported Anatomical Targets:
  - Spine (L4 Vertebra / Lumbar Spine)
  - Liver & Portal Vein (Hepatic Soft-Tissue & Vascular Branching)
  - Abdominal Aorta & Bifurcation (Interventional Vascular Access)

Simulation Modes:
  - "conv": Physics/convolution-based Model-Based (MB) B-mode US with Rayleigh speckle.
  - "ray":  Physics ray-tracing Model-Based (MB) US with Snell's law refraction.
  - "unet": Learning-Based (LB) generative US synthesis via 2-channel U-Net.
  - "pix2pix": Learning-Based (LB) generative US synthesis via Pix2Pix GAN.
"""

import os
import sys
import types

# PyTorch 2.6 on Python 3.12 compatibility shim for torch.utils.serialization.
# Only installed when the real module is genuinely missing. Newer PyTorch builds
# (e.g. 2.7.0+cu128 in this venv) ship torch.utils.serialization, and replacing it
# with the fake proxy below silently changes torch.save/torch.load config values.
try:
    import torch.utils.serialization  # noqa: F401  (real module, if available)
    _NEED_SERIALIZATION_SHIM = False
except ImportError:
    _NEED_SERIALIZATION_SHIM = True

if _NEED_SERIALIZATION_SHIM and "torch.utils.serialization" not in sys.modules:
    class _ConfigProxy:
        def __getattr__(self, name):
            if "storage_alignment" in name: return 64
            return False
    m = types.ModuleType("torch.utils.serialization")
    cfg = types.ModuleType("torch.utils.serialization.config")
    cfg.save = _ConfigProxy()
    cfg.load = _ConfigProxy()
    m.config = cfg
    sys.modules["torch.utils.serialization"] = m
    sys.modules["torch.utils.serialization.config"] = cfg
    try:
        import torch.utils
        torch.utils.serialization = m
    except Exception:
        pass

import time
import json
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import torch
import gymnasium as gym
from gymnasium import spaces
import cv2

# Add current folder to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from live_unet_demo import (
    create_panda_robot,
    create_probe_model,
    create_registered_body_mesh,
    load_registration_meta,
    compute_registered_ct_center,
    load_unet,
    predict_ultrasound,
    drive_panda_to_pose,
    ModelBasedUSSimulator,
    make_label_map,
    PANDA_EE_LINK,
    PANDA_ARM_JOINTS,
    BED_CENTER,
    BED_MATTRESS_TOP_Z,
    PROBE_TIP_FROM_EE,
    PROBE_QUAT_FROM_EE,
    select_device,
    resolve_subject_dir,
    load_ct_subject,
    update_probe_model,
    get_probe_pose_from_ee,
    create_hospital_room
)
from extract_slice import extract_slice
from registration import voxel_to_world
from image_quality_metric import compute_ultrasound_quality
from surface_motion_planner import load_or_create_surface_map, SurfaceMap


def snap_panda_to_pose(panda_id: int, target_position: np.ndarray, target_orientation: np.ndarray):
    """Accurate nullspace IK joint solver with Franka Panda limits."""
    num_joints = p.getNumJoints(panda_id)
    lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
    for i in range(num_joints):
        j_info = p.getJointInfo(panda_id, i)
        low, high = j_info[8], j_info[9]
        if low >= high:
            low, high = -2 * np.pi, 2 * np.pi
        lower_limits.append(low)
        upper_limits.append(high)
        joint_ranges.append(high - low)
        rest_poses.append(0.0)

    joint_targets = p.calculateInverseKinematics(
        panda_id, PANDA_EE_LINK,
        targetPosition=target_position.tolist(),
        targetOrientation=list(target_orientation),
        lowerLimits=lower_limits,
        upperLimits=upper_limits,
        jointRanges=joint_ranges,
        restPoses=rest_poses,
        jointDamping=[0.01] * num_joints,
        maxNumIterations=200,
        residualThreshold=1e-5
    )
    for jid, val in zip(PANDA_ARM_JOINTS, joint_targets):
        p.resetJointState(panda_id, jid, val)


TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
TEST_PATIENTS = ["s0344", "s0358"]

# Absolute project root so multi-patient switching works regardless of the launch cwd
_PROJECT_ROOT_DIR = Path(__file__).parent.resolve()

# In-memory / GPU cache for multi-patient domain randomization
_GLOBAL_SUBJECT_CACHE = {}

def get_cached_subject(subject_dir: Path, device: torch.device, bed_top_z: float, mesh_scale: np.ndarray, anatomy: str = "spine"):
    subject_dir = Path(subject_dir).resolve()
    cache_key = (str(subject_dir), str(device), str(anatomy))
    if cache_key not in _GLOBAL_SUBJECT_CACHE:
        ct_vol, lbl_vol, spacing, volume_center = load_ct_subject(subject_dir, anatomy=anatomy)
        ct_vol = np.ascontiguousarray(ct_vol, dtype=np.float32)
        lbl_vol = np.ascontiguousarray(lbl_vol, dtype=np.float32)
        ct_vol_t = torch.from_numpy(ct_vol).to(device)
        lbl_vol_t = torch.from_numpy(lbl_vol).to(device)
        reg_meta = load_registration_meta(subject_dir / "registration_meta.json")
        orientation = "supine" if str(anatomy).lower() == "aorta" else "prone"
        surface_planner = load_or_create_surface_map(
            subject_dir, bed_top_z=bed_top_z, mesh_scale=mesh_scale, orientation=orientation
        )
        vessel_vol_t = None
        if str(anatomy).lower() == "liver":
            v_path = subject_dir / "liver_vessels.nii.gz"
            if v_path.exists():
                try:
                    import nibabel as nib
                    v_vol = nib.load(str(v_path)).get_fdata(dtype=np.float32)
                    v_vol = np.ascontiguousarray(v_vol, dtype=np.float32)
                    vessel_vol_t = torch.from_numpy(v_vol).to(device)
                except Exception:
                    pass
        bone_vol_t = None
        b_candidates = ["bone_label.nii.gz", "bone_full.nii.gz", "spine_full.nii.gz"]
        for b_name in b_candidates:
            b_path = subject_dir / b_name
            if b_path.exists():
                try:
                    import nibabel as nib
                    b_vol = nib.load(str(b_path)).get_fdata(dtype=np.float32)
                    b_vol = np.ascontiguousarray(b_vol, dtype=np.float32)
                    bone_vol_t = torch.from_numpy(b_vol).to(device)
                    break
                except Exception:
                    pass
        _GLOBAL_SUBJECT_CACHE[cache_key] = {
            "ct_volume": ct_vol_t,
            "label_volume": lbl_vol_t,
            "vessel_volume": vessel_vol_t,
            "bone_volume": bone_vol_t,
            "spacing": spacing,
            "volume_center": volume_center,
            "reg_meta": reg_meta,
            "surface_planner": surface_planner,
            "body_center": None,
        }
    return _GLOBAL_SUBJECT_CACHE[cache_key]


class RoboticUltrasoundGymEnv(gym.Env):
    """Gymnasium environment for autonomous robotic ultrasound scanning."""

    def __init__(self,
                 subject_dir="totalseg_patients/s0058",
                 checkpoint_path="runs/cavalcanti_unet/best_model.pth",
                 device="auto",
                 render_mode="rgb_array",
                 max_episode_steps=300,
                 mesh_scale=1.0,
                 size=256,
                 pixel_spacing=1.0,
                 base_features=64,
                 skip_unet=False,
                 substeps=5,
                 max_velocity=1.5,
                 task="navigation",
                 anatomy="spine",
                 sim_mode="conv",
                 multi_patient=False,
                 quality_weight=1.0,
                 action_dim=3,
                 goal_conditioned=False,
                 roi_radius=float("inf"),
                 roi_radius_mm=None,
                 early_termination=True):
        super().__init__()

        self.subject_dir = Path(resolve_subject_dir(subject_dir))
        self.checkpoint_path = Path(checkpoint_path)
        self.render_mode = render_mode
        self.max_episode_steps = int(max_episode_steps)
        self.mesh_scale_val = mesh_scale
        self.mesh_scale = np.array([mesh_scale]*3, dtype=np.float64)
        self.slice_size = size
        self.task = str(task).lower()
        self.anatomy = str(anatomy).lower()
        # Per-target ultrasound imaging planes:
        # aorta: 192 mm depth at 1.5 mm/px (128x128)
        # liver: 224 mm depth at 1.75 mm/px (128x128)
        # spine: 128 mm depth at 1.0 mm/px (128x128)
        if pixel_spacing in ["auto", None]:
            if self.anatomy == "aorta":
                self.pixel_spacing = 1.5
            elif self.anatomy == "liver":
                self.pixel_spacing = 1.75
            else:
                self.pixel_spacing = 1.0
        else:
            self.pixel_spacing = float(pixel_spacing)
        self.base_features = base_features
        self.skip_unet = skip_unet
        self.substeps = int(substeps)
        self.max_velocity = float(max_velocity)
        self.sim_mode = str(sim_mode).lower()
        self.multi_patient = bool(multi_patient)
        self.quality_weight = float(quality_weight)
        self.action_dim = int(action_dim)
        self.goal_conditioned = bool(goal_conditioned)
        # ROI radius for localized surface reconstruction (stored in meters; inf = whole organ)
        if roi_radius_mm is not None:
            r_mm = float(roi_radius_mm)
            self.roi_radius = float("inf") if np.isinf(r_mm) else (r_mm / 1000.0)
        elif roi_radius is not None:
            r_val = float(roi_radius)
            if r_val > 1.0 and not np.isinf(r_val):
                r_val = r_val / 1000.0
            self.roi_radius = r_val
        else:
            self.roi_radius = float("inf")
        self.early_termination = bool(early_termination)
        self.w1 = 0.5  # Position progress weight (0.5 per mm progress)

        # Device selection
        self.device = select_device(device)

        # Load Ground Truth 6-DoF Target Landmarks
        self._load_target_databases()

        # Initialize Physics Simulator for Model-Based mode
        self.mb_sim = ModelBasedUSSimulator()

        # Connect to PyBullet
        if self.render_mode == "human":
            self.client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Hospital Room & Environment Layout
        self.bed_top_z = create_hospital_room()

        # Spawn Patient Mesh
        self._spawn_patient_mesh()

        # Load CT Volume & Label Volume for initial subject
        self._load_subject_data(self.subject_dir)

        # Load SonoGym Surface Motion Planner & Surface Map
        orientation = "supine" if str(self.anatomy).lower() == "aorta" else "prone"
        self.surface_planner = load_or_create_surface_map(
            self.subject_dir, bed_top_z=self.bed_top_z, mesh_scale=self.mesh_scale, orientation=orientation
        )

        # Load U-Net / Pix2Pix Model if in Learning-Based mode
        if not self.skip_unet and self.sim_mode in ["unet", "pix2pix"]:
            self.model = load_unet(self.checkpoint_path, self.device, base_features=self.base_features, dropout=0.0)
        else:
            self.model = None

        # Action space: SonoGym 3-DoF [dx, dy, dyaw] by default, or 6-DoF for backwards compatibility
        if self.action_dim == 3:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(6,), dtype=np.float32
            )

        # Observation space: Dict containing B-mode image, contact force, and pose
        if self.task == "reconstruction":
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(self.slice_size, self.slice_size), dtype=np.uint8),
                "force": spaces.Box(low=0.0, high=15.0, shape=(1,), dtype=np.float32),
                "pose": spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float32),
                # 2-channel 40x40 spatial memory grid in local probe frame (top-down and frontal projections)
                "occupancy": spaces.Box(low=0.0, high=1.0, shape=(2, 40, 40), dtype=np.float32)
            })
        else:
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(self.slice_size, self.slice_size), dtype=np.uint8),
                "force": spaces.Box(low=0.0, high=15.0, shape=(1,), dtype=np.float32),
                "pose": spaces.Box(low=-5.0, high=5.0, shape=(7,), dtype=np.float32)
            })

        # Spawn Franka Panda Robot & Ultrasound Probe
        self.panda_id = create_panda_robot()
        self.probe_body_id = create_probe_model()
        for link in [9, 10]:
            p.changeVisualShape(self.panda_id, link, rgbaColor=[0.0, 0.0, 0.0, 0.0])

        # Default home poses
        self.home_orn = p.getQuaternionFromEuler([0, np.pi, 0])
        self.home_pos = np.array([self.body_center[0], self.body_center[1], self.bed_top_z + 0.35], dtype=np.float32)

        # State tracking variables
        self.step_counter = 0
        self.current_force = 4.0
        self.current_quality = 0.5
        self.reconstructed_voxels_world = set()
        self.current_surface_x = float(self.body_center[0])
        self.current_surface_y = float(self.body_center[1])
        self.current_surface_yaw = 0.0
        self.current_indentation = 0.002  # nominal 2 mm clinical acoustic coupling indentation
        self.current_is_valid = True
        self.current_probe_tip_pos = np.zeros(3, dtype=np.float32)
        self.current_probe_quat = np.array([0, 0, 0, 1], dtype=np.float32)
        self.current_ee_pos = np.zeros(3, dtype=np.float32)
        self.current_ee_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.current_surface_normal = np.array([0, 0, 1], dtype=np.float64)
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.target_world_pos = np.zeros(3, dtype=np.float64)
        self.target_internal_world_pos = np.zeros(3, dtype=np.float64)
        self.target_surface_world_pos = np.zeros(3, dtype=np.float64)
        self.target_depth_mm = 0.0
        self.target_world_normal = np.array([0, 0, 1], dtype=np.float64)
        self.prev_pos_err = 100.0
        self.prev_rot_err = 1.0
        self.visited_voxels = set()

    def _load_target_databases(self):
        """Load pre-computed 6-DoF anatomical ground truth target dictionaries."""
        self.l4_targets = {}
        self.pv_targets = {}
        self.aorta_targets = {}
        
        root = Path(__file__).parent.resolve()
        for fpath, target_dict in [
            (root / "l4_svp_targets.json", self.l4_targets),
            (root / "portal_vein_targets.json", self.pv_targets),
            (root / "aorta_bifurcation_targets.json", self.aorta_targets)
        ]:
            if fpath.exists():
                with open(fpath, "r") as f:
                    target_dict.update(json.load(f))

    def _load_subject_data(self, subject_dir: Path):
        """Load CT volume, label volume, and registration metadata for a subject.

        Uses the per-process ``_GLOBAL_SUBJECT_CACHE`` so that, in multi-patient mode,
        each patient's NIfTI volumes are read from disk and uploaded to the device only
        once instead of on every patient switch. The cached tensors are only read
        (sampled by ``extract_slice``), never modified.
        """
        self.subject_dir = Path(subject_dir)
        cached = get_cached_subject(self.subject_dir, self.device, self.bed_top_z, self.mesh_scale, anatomy=self.anatomy)
        self.ct_volume = cached["ct_volume"]
        self.label_volume = cached["label_volume"]
        self.spacing = cached["spacing"]
        self.volume_center = cached["volume_center"]
        self.reg_meta = cached["reg_meta"]
        self.surface_planner = cached["surface_planner"]
        self.vessel_volume = cached.get("vessel_volume", None)
        self.bone_volume = cached.get("bone_volume", None)

    def _spawn_patient_mesh(self):
        """Spawn and cache registered meshes for patients with zero memory leaks."""
        from live_unet_demo import get_body_bounds
        if not hasattr(self, "_spawned_patients"):
            self._spawned_patients = {}

        subj_name = self.subject_dir.name
        orientation = "supine" if str(self.anatomy).lower() == "aorta" else "prone"
        patient_key = (subj_name, orientation)
        if patient_key not in self._spawned_patients:
            (body_id,
             reg_body_position,
             reg_body_orientation_matrix,
             reg_meta) = create_registered_body_mesh(
                self.subject_dir, self.bed_top_z, self.mesh_scale, orientation=orientation
            )
            _, _, body_center, _ = get_body_bounds(body_id)
            quat = p.getQuaternionFromEuler([np.pi / 2, 0, 0]) if orientation == "supine" else p.getQuaternionFromEuler([-np.pi / 2, 0, 0])
            self._spawned_patients[patient_key] = {
                "body_id": body_id,
                "reg_body_position": reg_body_position,
                "reg_body_orientation_matrix": reg_body_orientation_matrix,
                "reg_meta": reg_meta,
                "body_center": body_center,
                "body_orientation_quat": quat,
            }

        # Hide inactive patients under the floor; bring active patient to bed
        for key, sdata in self._spawned_patients.items():
            if key == patient_key:
                p.resetBasePositionAndOrientation(
                    sdata["body_id"],
                    sdata["reg_body_position"].tolist(),
                    sdata["body_orientation_quat"]
                )
                self.body_id = sdata["body_id"]
                self.reg_body_position = sdata["reg_body_position"]
                self.reg_body_orientation_matrix = sdata["reg_body_orientation_matrix"]
                self.reg_meta = sdata["reg_meta"]
                self.body_center = sdata["body_center"]
            else:
                p.resetBasePositionAndOrientation(
                    sdata["body_id"],
                    [0.0, 0.0, -50.0],
                    sdata["body_orientation_quat"]
                )

    def _compute_current_target(self):
        """Calculate the 3D world position and orientation of the active anatomical landmark."""
        subj_name = self.subject_dir.name
        
        if self.anatomy == "spine":
            entry = self.l4_targets.get(subj_name, None)
            key = "voxel_centroid_zyx"
            axis_key = "principal_axis_mm"
        elif self.anatomy == "liver":
            entry = self.pv_targets.get(subj_name, None)
            key = "voxel_centroid_zyx"
            axis_key = None
        elif self.anatomy == "aorta":
            entry = self.aorta_targets.get(subj_name, None)
            key = "voxel_bifurcation_zyx"
            axis_key = None
        else:
            entry = None

        if entry is not None and key in entry:
            vox = entry[key]
            world_pos = voxel_to_world(
                vox, self.reg_body_position, self.reg_body_orientation_matrix,
                self.reg_meta["affine"], self.reg_meta["mesh_centering_offset"],
                mesh_scale=self.mesh_scale
            )
            self.target_internal_world_pos = np.asarray(world_pos, dtype=np.float64)
            
            # Find acoustic window on patient skin directly above internal organ (SonoGym SVP)
            skin_z, normal, is_valid = self.surface_planner.query(
                float(self.target_internal_world_pos[0]),
                float(self.target_internal_world_pos[1])
            )
            if not is_valid:
                skin_z = float(self.target_internal_world_pos[2] + 0.10)
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            
            self.target_depth_mm = float(abs(skin_z - self.target_internal_world_pos[2]) * 1000.0)
            self.target_surface_world_pos = np.array([
                self.target_internal_world_pos[0],
                self.target_internal_world_pos[1],
                skin_z
            ], dtype=np.float64)
            
            # For SonoGym Task 1 probe navigation, the target is the surface acoustic window (with 2mm nominal contact indentation):
            self.target_world_pos = self.target_surface_world_pos - 0.002 * normal
            
            # Acoustic entry vector into tissue is perpendicular to the skin surface normal:
            self.target_world_normal = normal
        else:
            self.target_internal_world_pos = np.array([self.body_center[0], self.body_center[1], self.bed_top_z + 0.05], dtype=np.float64)
            self.target_surface_world_pos = np.array([self.body_center[0], self.body_center[1], self.bed_top_z + 0.15], dtype=np.float64)
            self.target_world_pos = self.target_surface_world_pos - 0.002 * np.array([0, 0, 1], dtype=np.float64)
            self.target_depth_mm = 100.0
            self.target_world_normal = np.array([0, 0, 1], dtype=np.float64)

    def reset(self, seed=None, options=None):
        """Reset the environment matching SonoGym 130x130 mm bounded search space."""
        super().reset(seed=seed)
        self.step_counter = 0
        self.visited_voxels = set()
        self.has_succeeded = False

        # Make the model-based speckle noise reproducible when an explicit seed is given
        if seed is not None:
            self.mb_sim.rng = np.random.default_rng(seed)

        # Multi-Patient Domain Randomization (Zero-leak cached mesh translation)
        if self.multi_patient:
            new_subj_id = str(self.np_random.choice(TRAIN_PATIENTS))
            new_subj_dir = _PROJECT_ROOT_DIR / "totalseg_patients" / new_subj_id
            if new_subj_dir.resolve() != self.subject_dir.resolve() and (new_subj_dir / "ct.nii.gz").exists():
                self._load_subject_data(new_subj_dir)
                self._spawn_patient_mesh()

        # Compute Ground Truth 6-DoF Landmark
        self._compute_current_target()

        # SonoGym Section C.1: Initialize within 130 x 130 mm bounding box on patient skin
        # Centered over the anatomical target, ensuring probe starts at least 25 mm away on valid skin
        valid_found = False
        for _ in range(30):
            angle = float(self.np_random.uniform(0, 2 * np.pi))
            dist = float(self.np_random.uniform(0.025, 0.065))
            cand_x = float(self.target_world_pos[0] + dist * np.cos(angle))
            cand_y = float(self.target_world_pos[1] + dist * np.sin(angle))
            _, _, is_valid = self.surface_planner.query(cand_x, cand_y)
            if is_valid:
                init_x = cand_x
                init_y = cand_y
                valid_found = True
                break

        if not valid_found:
            for dist in [0.020, 0.015, 0.010]:
                for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    cand_x = float(self.target_world_pos[0] + dist * np.cos(ang))
                    cand_y = float(self.target_world_pos[1] + dist * np.sin(ang))
                    _, _, is_valid = self.surface_planner.query(cand_x, cand_y)
                    if is_valid:
                        init_x = cand_x
                        init_y = cand_y
                        valid_found = True
                        break
                if valid_found:
                    break
            if not valid_found:
                init_x = float(self.target_world_pos[0])
                init_y = float(self.target_world_pos[1])

        # Transverse acoustic plane reference (0.0 rad) with randomized perturbation in [-0.5, 0.5] rad (~28 deg)
        rand_yaw = float(self.np_random.uniform(-0.5, 0.5))

        self.current_surface_x = init_x
        self.current_surface_y = init_y
        self.current_surface_yaw = rand_yaw
        self.current_indentation = 0.002  # nominal 2 mm clinical acoustic coupling indentation

        if self.task == "reconstruction":
            # SonoGym Section 4.2: Randomize initial probe position in 30x30 mm^2 region around target
            offset_x = float(self.np_random.uniform(-0.015, 0.015))
            offset_y = float(self.np_random.uniform(-0.015, 0.015))
            init_yaw = float(self.np_random.uniform(-np.pi / 6.0, np.pi / 6.0))
            self.current_surface_x = float(self.target_surface_world_pos[0] + offset_x)
            self.current_surface_y = float(self.target_surface_world_pos[1] + offset_y)
            self.reconstructed_voxels_world = set()
            self.all_reconstructed_voxels_world = set()
            self.trajectory_positions = []

        # Low-level surface controller: compute exact surface poses & contact force
        (probe_tip_pos, probe_quat, ee_pos, ee_quat,
         normal, force, is_valid) = self.surface_planner.compute_surface_pose(
            self.current_surface_x, self.current_surface_y, self.current_surface_yaw,
            indentation=self.current_indentation
        )
        self.current_probe_tip_pos = probe_tip_pos
        self.current_probe_quat = probe_quat
        self.current_ee_pos = ee_pos
        self.current_ee_quat = ee_quat
        self.current_force = float(force)
        self.current_is_valid = bool(is_valid)
        self.current_surface_normal = normal
        self.home_pos = ee_pos.copy()
        self.home_orn = ee_quat.copy()

        # Snap Panda arm directly to the surface pose via nullspace IK
        snap_panda_to_pose(self.panda_id, ee_pos, ee_quat)
        update_probe_model(self.probe_body_id, probe_tip_pos, probe_quat)
        for _ in range(10):
            p.stepSimulation()

        # Compute initial errors
        self.prev_pos_err = float(np.linalg.norm(self.current_probe_tip_pos - self.target_world_pos) * 1000.0)
        
        R_probe = np.array(p.getMatrixFromQuaternion(self.current_probe_quat.tolist())).reshape(3, 3)
        probe_beam = -R_probe[:, 2]
        target_beam = -self.target_world_normal
        dot = np.clip(np.dot(probe_beam, target_beam), -1.0, 1.0)
        self.prev_rot_err = float(np.arccos(dot))

        obs = self._get_obs()
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        info = {
            "pos_error_mm": self.prev_pos_err,
            "rot_error_deg": float(np.degrees(self.prev_rot_err)),
            "quality_score": self.current_quality,
            "target_depth_mm": self.target_depth_mm,
            "binary_contact": 1.0 if (self.current_is_valid and self.current_force >= 1.0) else 0.0,
            "penetration_depth_mm": float(self.current_indentation * 1000.0) if self.current_is_valid else 0.0,
            "is_success": False
        }
        return obs, info

    def _get_obs(self):
        """Synthesize B-mode ultrasound image (MB or LB) and return gym observation (Zero Raycasts)."""
        ee_state = p.getLinkState(self.panda_id, PANDA_EE_LINK)
        ee_pos = np.array(ee_state[4], dtype=np.float32)
        ee_orn = np.array(ee_state[5], dtype=np.float32)

        # Ultrasound synthesis: acquired whenever probe is in acoustic contact on the body
        if self.current_is_valid and self.current_force >= 1.0:
            ct_center = compute_registered_ct_center(
                self.current_probe_tip_pos, self.reg_body_position, self.reg_body_orientation_matrix,
                self.reg_meta['inv_affine'], self.reg_meta['mesh_centering_offset'],
                self.ct_volume.shape, mesh_scale=self.mesh_scale
            )

            # Extract internal CT & label slices using probe transducer orientation
            ct_slice = extract_slice(
                self.ct_volume, center=ct_center, quaternion=self.current_probe_quat,
                spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                body_orientation_matrix=self.reg_body_orientation_matrix
            )
            seg_slice = extract_slice(
                self.label_volume, center=ct_center, quaternion=self.current_probe_quat,
                spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                body_orientation_matrix=self.reg_body_orientation_matrix,
                order=0
            )
            seg_binary = (seg_slice > 0.5).astype(np.float32)

            vessel_slice = None
            if self.anatomy == "liver" and getattr(self, "vessel_volume", None) is not None:
                v_raw = extract_slice(
                    self.vessel_volume, center=ct_center, quaternion=self.current_probe_quat,
                    spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                    inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                    body_orientation_matrix=self.reg_body_orientation_matrix,
                    order=0
                )
                vessel_slice = (v_raw > 0.5).astype(np.float32)

            bone_slice = None
            if getattr(self, "bone_volume", None) is not None:
                b_raw = extract_slice(
                    self.bone_volume, center=ct_center, quaternion=self.current_probe_quat,
                    spacing=self.spacing, size=self.slice_size, pixel_spacing=self.pixel_spacing,
                    inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                    body_orientation_matrix=self.reg_body_orientation_matrix,
                    order=0
                )
                bone_slice = (b_raw > 0.5).astype(np.float32)

            # Synthesize B-mode Ultrasound (Model-Based vs Learning-Based)
            if self.skip_unet:
                us_image_uint8 = (seg_binary * 255).astype(np.uint8)
            elif self.sim_mode in ["conv", "ray"]:
                # Model-Based (MB) Acoustic Simulation
                lbl_map = make_label_map(ct_slice, seg_binary, anatomy=self.anatomy, vessel_slice=vessel_slice, bone_slice=bone_slice)
                if self.sim_mode == "ray":
                    us_img = self.mb_sim.simulate_raytrace(lbl_map, ct_slice=ct_slice)
                else:
                    us_img = self.mb_sim.simulate(lbl_map, ct_slice=ct_slice)
                us_image_uint8 = np.clip(us_img * 255.0, 0, 255).astype(np.uint8)
            else:
                # Learning-Based (LB) U-Net / Pix2Pix Simulation
                is_p2p = (self.sim_mode == "pix2pix")
                if self.slice_size == 256:
                    ct_infer = ct_slice
                    seg_infer = seg_binary
                else:
                    ct_infer = extract_slice(
                        self.ct_volume, center=ct_center, quaternion=self.current_probe_quat,
                        spacing=self.spacing, size=256, pixel_spacing=self.pixel_spacing,
                        inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                        body_orientation_matrix=self.reg_body_orientation_matrix
                    )
                    seg_infer = extract_slice(
                        self.label_volume, center=ct_center, quaternion=self.current_probe_quat,
                        spacing=self.spacing, size=256, pixel_spacing=self.pixel_spacing,
                        inv_affine=self.reg_meta['inv_affine'], mesh_scale=self.mesh_scale,
                        body_orientation_matrix=self.reg_body_orientation_matrix,
                        order=0
                    )
                    seg_infer = (seg_infer > 0.5).astype(np.float32)

                us_img = predict_ultrasound(
                    self.model, ct_infer, seg_infer, self.device, is_pix2pix=is_p2p, enhance=True
                )
                if self.slice_size != 256:
                    us_img = cv2.resize(us_img, (self.slice_size, self.slice_size), interpolation=cv2.INTER_AREA)
                us_image_uint8 = np.clip(us_img * 255.0, 0, 255).astype(np.uint8)

            self.last_seg_slice = seg_binary
            self.last_ct_slice = ct_slice
        else:
            us_image_uint8 = np.zeros((self.slice_size, self.slice_size), dtype=np.uint8)
            self.last_seg_slice = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)
            self.last_ct_slice = np.zeros((self.slice_size, self.slice_size), dtype=np.float32)

        # Evaluate MICCAI 2025 Foundation Model Image Quality
        self.current_quality = compute_ultrasound_quality(us_image_uint8, anatomy=self.anatomy)

        if self.goal_conditioned:
            # Relative displacement vector to target scaled by 20.0 (i.e. / 0.05m = 50mm)
            # so that 50mm = 1.0, 25mm = 0.5, 5mm = 0.1, matching [-1, 1] range of action and quaternion
            delta_p = np.clip((self.target_world_pos - self.current_probe_tip_pos) / 0.05, -2.0, 2.0).astype(np.float32)
            pose_obs = np.concatenate([delta_p, self.current_probe_quat], dtype=np.float32)
        else:
            # Absolute robot end-effector pose [pos, orn] in R^7
            pose_obs = np.concatenate([ee_pos, ee_orn], dtype=np.float32)

        obs = {
            "image": us_image_uint8,
            "force": np.array([self.current_force / 10.0], dtype=np.float32),
            "pose": pose_obs
        }
        if self.task == "reconstruction":
            obs["occupancy"] = self._get_probe_occupancy_grid()
        return obs

    def _get_probe_occupancy_grid(self):
        """Computes 2-channel 40x40 spatial memory map in local probe frame (Ch 0: Top-Down X-Y, Ch 1: Frontal X-Z)."""
        grid = np.zeros((2, 40, 40), dtype=np.float32)
        if not hasattr(self, 'reconstructed_voxels_world') or len(self.reconstructed_voxels_world) == 0:
            return grid
        R_probe = np.array(p.getMatrixFromQuaternion(self.current_probe_quat.tolist())).reshape(3, 3)
        pts_world = np.array(list(self.reconstructed_voxels_world), dtype=np.float64) * 0.0015  # 1.5mm voxels to meters
        pts_local = (R_probe.T @ (pts_world - self.current_probe_tip_pos).T).T
        # Local coordinate range: 4mm (spine, aorta -> 160mm window), 5mm (liver -> 200mm window)
        VOX_SZ = 0.005 if self.anatomy == "liver" else 0.004
        gx = np.round(pts_local[:, 0] / VOX_SZ + 20).astype(int)
        gy = np.round(pts_local[:, 1] / VOX_SZ + 20).astype(int)
        gz = np.round(-pts_local[:, 2] / VOX_SZ).astype(int)
        # Top-down projection (X vs Y)
        val_xy = (gx >= 0) & (gx < 40) & (gy >= 0) & (gy < 40)
        grid[0, gx[val_xy], gy[val_xy]] = 1.0
        # Frontal projection (X vs Z)
        val_xz = (gx >= 0) & (gx < 40) & (gz >= 0) & (gz < 40)
        grid[1, gx[val_xz], gz[val_xz]] = 1.0
        return grid

    def step(self, action):
        """Advance simulation along the patient skin surface (SonoGym Admittance Control)."""
        self.step_counter += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if len(action) == 3:
            # SonoGym 3-DoF: [dx, dy, dyaw]
            dx = float(action[0] * 0.005)
            dy = float(action[1] * 0.005)
            dyaw = float(action[2] * 0.05)
            dindent = 0.0
        else:
            # 6-DoF compatibility: [dx, dy, dz/dindent, droll, dpitch, dyaw]
            dx = float(action[0] * 0.005)
            dy = float(action[1] * 0.005)
            dindent = float(-action[2] * 0.0005)
            dyaw = float(action[5] * 0.05)

        self.current_surface_x += dx
        self.current_surface_y += dy
        self.current_surface_yaw += dyaw
        self.current_indentation = float(np.clip(self.current_indentation + dindent, 0.0005, 0.003))

        # Clamp within valid patient surface grid bounds
        self.current_surface_x = float(np.clip(
            self.current_surface_x, self.surface_planner.x_min + 0.02, self.surface_planner.x_max - 0.02
        ))
        self.current_surface_y = float(np.clip(
            self.current_surface_y, self.surface_planner.y_min + 0.02, self.surface_planner.y_max - 0.02
        ))

        # Evaluate SonoGym low-level surface admittance pose
        (probe_tip_pos, probe_quat, ee_pos, ee_quat,
         normal, force, is_valid) = self.surface_planner.compute_surface_pose(
            self.current_surface_x, self.current_surface_y, self.current_surface_yaw,
            indentation=self.current_indentation
        )
        self.current_probe_tip_pos = probe_tip_pos
        self.current_probe_quat = probe_quat
        self.current_ee_pos = ee_pos
        self.current_ee_quat = ee_quat
        self.current_force = float(force)
        self.current_is_valid = bool(is_valid)
        self.current_surface_normal = normal

        snap_panda_to_pose(self.panda_id, ee_pos, ee_quat)
        update_probe_model(self.probe_body_id, probe_tip_pos, probe_quat)
        for _ in range(self.substeps):
            p.stepSimulation()

        obs = self._get_obs()
        reward, terminated, truncated, info = self._compute_reward(action)

        return obs, reward, terminated, truncated, info

    def _compute_reward(self, action):
        """Compute the SonoGym and MICCAI 2025 reward."""
        curr_pos_err = float(np.linalg.norm(self.current_probe_tip_pos - self.target_world_pos) * 1000.0)  # mm
        R_probe = np.array(p.getMatrixFromQuaternion(self.current_probe_quat.tolist())).reshape(3, 3)
        probe_beam = -R_probe[:, 2]
        target_beam = -self.target_world_normal
        dot = np.clip(np.dot(probe_beam, target_beam), -1.0, 1.0)
        curr_rot_err = float(np.arccos(dot))  # rad

        terminated = False
        truncated = (self.step_counter >= self.max_episode_steps)
        info = {
            "pos_error_mm": curr_pos_err,
            "rot_error_deg": float(np.degrees(curr_rot_err)),
            "quality_score": self.current_quality,
            "target_depth_mm": self.target_depth_mm,
            "binary_contact": 1.0 if (self.current_is_valid and self.current_force >= 1.0) else 0.0,
            "penetration_depth_mm": float(self.current_indentation * 1000.0) if self.current_is_valid else 0.0,
            "is_success": False
        }

        if self.task == "navigation":
            # 1. Out-of-bounds termination: prevent probe from loitering in air with blank images
            if not self.current_is_valid:
                terminated = True
                return -5.0, terminated, truncated, info

            # 2. SonoGym Appendix C.1 Differential Progress: positive for moving closer, negative for moving away
            R_pos = 0.5 * (self.prev_pos_err - curr_pos_err)
            R_rot = 1.0 * (self.prev_rot_err - curr_rot_err)

            # 3. Success condition: < 5 mm, surface normal aligned (< 10 deg), Q >= min_q
            R_terminal = 0.0
            is_at_goal = (curr_pos_err < 5.0 and curr_rot_err < np.radians(10.0))
            min_q = 0.40  # Unified across spine, aorta, and liver
            if is_at_goal and self.current_quality >= min_q:
                info["is_success"] = True
                if not self.has_succeeded:
                    R_terminal = 50.0
                    self.has_succeeded = True
                if self.early_termination:
                    terminated = True

            reward = R_pos + R_rot + R_terminal

        elif self.task == "reconstruction":
            # True 3D Ultrasound Anatomy Surface Reconstruction
            from scipy.ndimage import binary_erosion
            marginal_gain = 0.0
            new_discovered = 0
            if self.current_force >= 1.0 and self.current_is_valid and hasattr(self, 'last_seg_slice'):
                seg = self.last_seg_slice
                if self.anatomy == "spine":
                    # Dorsal bone surface: top-most hyperechoic reflection per column
                    bone_cols = np.where(seg > 0.5)
                    if len(bone_cols[0]) > 0:
                        col_indices = np.unique(bone_cols[1])
                        row_indices = [np.min(bone_cols[0][bone_cols[1] == c]) for c in col_indices]
                        y_pix, x_pix = np.array(row_indices), np.array(col_indices)
                    else:
                        y_pix, x_pix = np.array([]), np.array([])
                else:
                    # Vascular wall boundary: 2D morphological binary erosion
                    m = (seg > 0.5)
                    edge = m & ~binary_erosion(m)
                    y_pix, x_pix = np.where(edge)
                if len(x_pix) > 0:
                    # Convert pixel coordinates to physical local probe frame (meters)
                    x_local = (x_pix - self.slice_size / 2.0) * (self.pixel_spacing / 1000.0)
                    y_local = np.zeros_like(x_local)
                    z_local = -y_pix * (self.pixel_spacing / 1000.0)
                    # Transform to 3D PyBullet world coordinates via forward probe kinematics
                    R_probe = np.array(p.getMatrixFromQuaternion(self.current_probe_quat.tolist())).reshape(3, 3)
                    P_local = np.stack([x_local, y_local, z_local], axis=0)
                    P_world = self.current_probe_tip_pos[:, None] + R_probe @ P_local
                    # Accumulate into 1.5mm 3D world voxel grid
                    VOXEL_RES = 0.0015
                    vox_coords = np.round(P_world / VOXEL_RES).astype(int)
                    for i in range(vox_coords.shape[1]):
                        v_key = (int(vox_coords[0, i]), int(vox_coords[1, i]), int(vox_coords[2, i]))
                        if hasattr(self, "all_reconstructed_voxels_world"):
                            self.all_reconstructed_voxels_world.add(v_key)
                        
                        in_roi = True
                        if self.roi_radius is not None and not np.isinf(self.roi_radius):
                            dist = np.linalg.norm(P_world[:, i] - self.target_internal_world_pos)
                            if dist > self.roi_radius:
                                in_roi = False
                                
                        if in_roi:
                            if v_key not in self.reconstructed_voxels_world:
                                self.reconstructed_voxels_world.add(v_key)
                                new_discovered += 1
                    marginal_gain = float(new_discovered)
            
            if hasattr(self, "trajectory_positions"):
                self.trajectory_positions.append(self.current_probe_tip_pos.copy())

            # Submodular reward: marginal discovered point gain minus action effort
            R_coverage = marginal_gain * 0.05
            act_y = action[1] if len(action) > 1 else 0.0
            act_r1 = action[4] if len(action) > 4 else 0.0
            act_r2 = action[2] if len(action) == 3 else (action[5] if len(action) > 5 else 0.0)
            R_effort = -0.01 * (abs(action[0]) + abs(act_y) + 0.5 * abs(act_r1) + 0.5 * abs(act_r2))
            
            reward = R_coverage + R_effort
            info["coverage_points"] = len(self.reconstructed_voxels_world)
            info["all_coverage_points"] = len(self.all_reconstructed_voxels_world) if hasattr(self, "all_reconstructed_voxels_world") else len(self.reconstructed_voxels_world)

        else:
            # Legacy reward preservation
            F = self.current_force
            R_f = 2.0 if (2.0 <= F <= 8.0) else (-1.0 if F == 0 else 0.5)
            R_b = 1.0 if np.sum(self.last_seg_slice) > 5.0 else 0.0
            R_act = -0.05 * float(np.sum(np.square(action)))
            reward = R_f + R_b + R_act

        self.prev_pos_err = curr_pos_err
        self.prev_rot_err = curr_rot_err
        self.last_action = action.copy()

        return float(reward), terminated, truncated, info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
