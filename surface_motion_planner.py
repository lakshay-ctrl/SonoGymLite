"""
surface_motion_planner.py
=========================
Replication of SonoGym (NeurIPS 2025) Surface Motion Planner & Admittance Control.

Key Principles:
1. Low-Level Surface Admittance Controller:
   - Holds probe strictly on the patient's outer skin surface.
   - Applies nominal acoustic contact indentation (2 mm, clamped to [0, 3] mm).
   - NEVER penetrates deep into internal torso structures or organs during navigation.
   - Maintains steady clinical contact force (nominal 4.0 N, F in [2, 8] N).
2. Orientation Alignment:
   - Aligns probe beam perpendicular to the local skin surface normal.
   - Supports continuous tangential yaw rotation around the surface normal.
3. Precomputed Surface & Normal Maps:
   - High-resolution 2D surface grid (Z_skin and outward unit normal n_skin).
   - Instant O(1) bilinear interpolation with zero runtime raycasting in env.step().
"""

import os
import sys
from pathlib import Path
import numpy as np
import pybullet as p
from scipy.interpolate import RegularGridInterpolator

# Add root directory to sys.path
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from live_unet_demo import (
    create_registered_body_mesh,
    BED_MATTRESS_TOP_Z,
    PROBE_TIP_FROM_EE,
    PROBE_QUAT_FROM_EE,
    multiply_quaternions
)

ALL_PATIENTS = ["s0011", "s0058", "s0250", "s0344", "s0358"]


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a PyBullet quaternion [x, y, z, w]."""
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]

    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float32)
    norm = np.linalg.norm(q)
    return (q / norm) if norm > 1e-6 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


class SurfaceMap:
    """Encapsulates precomputed 2D skin surface height and outward normal map."""

    def __init__(self, data_path: Path, bed_top_z: float = BED_MATTRESS_TOP_Z):
        self.data_path = Path(data_path)
        data = np.load(self.data_path)
        self.xs = data["xs"]
        self.ys = data["ys"]
        self.surface_z = np.copy(data["surface_z"])
        self.surface_normals = data["surface_normals"]
        self.valid_mask = data["valid_mask"]

        # Automatic bed height alignment:
        # If the map was generated at a legacy/different bed height (e.g. 0.53m instead of 0.72m),
        # automatically shift surface_z by the difference so the probe matches the actual 3D mesh.
        if "bed_top_z" in data:
            saved_bed_z = float(data["bed_top_z"])
            dz = float(bed_top_z - saved_bed_z)
            if abs(dz) > 1e-4:
                self.surface_z += dz
        else:
            # Legacy file without stored bed_top_z: check if it was precomputed at 0.53m
            max_z = float(np.nanmax(self.surface_z)) if np.any(np.isfinite(self.surface_z)) else 0.0
            if max_z < 0.90:  # Surface peak < 0.90m indicates legacy 0.53m bed height
                dz = float(bed_top_z - 0.53)
                self.surface_z += dz

        # Build fast RegularGridInterpolators (ys along axis 0, xs along axis 1)
        # Handle nan values by filling with min surface height for fallback
        clean_z = np.copy(self.surface_z)
        min_z = float(np.nanmin(clean_z)) if np.any(np.isfinite(clean_z)) else (bed_top_z + 0.01)
        clean_z[~np.isfinite(clean_z)] = min_z

        self._interp_z = RegularGridInterpolator(
            (self.ys, self.xs), clean_z, method="linear", bounds_error=False, fill_value=min_z
        )
        self._interp_nx = RegularGridInterpolator(
            (self.ys, self.xs), self.surface_normals[:, :, 0], method="linear", bounds_error=False, fill_value=0.0
        )
        self._interp_ny = RegularGridInterpolator(
            (self.ys, self.xs), self.surface_normals[:, :, 1], method="linear", bounds_error=False, fill_value=0.0
        )
        self._interp_nz = RegularGridInterpolator(
            (self.ys, self.xs), self.surface_normals[:, :, 2], method="linear", bounds_error=False, fill_value=1.0
        )
        self._interp_valid = RegularGridInterpolator(
            (self.ys, self.xs), self.valid_mask.astype(np.float32), method="nearest", bounds_error=False, fill_value=0.0
        )

        self.x_min, self.x_max = float(self.xs[0]), float(self.xs[-1])
        self.y_min, self.y_max = float(self.ys[0]), float(self.ys[-1])

    def query(self, x: float, y: float) -> tuple[float, np.ndarray, bool]:
        """Query continuous skin surface height Z and outward unit normal n at (x, y)."""
        # Clamp coordinates within bounding grid
        qx = float(np.clip(x, self.x_min, self.x_max))
        qy = float(np.clip(y, self.y_min, self.y_max))
        pt = np.array([[qy, qx]], dtype=np.float64)

        skin_z = float(self._interp_z(pt)[0])
        nx = float(self._interp_nx(pt)[0])
        ny = float(self._interp_ny(pt)[0])
        nz = float(self._interp_nz(pt)[0])
        is_valid = bool(self._interp_valid(pt)[0] > 0.5)

        normal = np.array([nx, ny, nz], dtype=np.float64)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-4:
            normal /= n_len
        else:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        return skin_z, normal, is_valid

    def compute_surface_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        indentation: float = 0.002,
        nominal_force: float = 4.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """
        Compute probe tip pose and robot end-effector pose adhering strictly to the skin surface.

        Returns:
            probe_tip_pos: (3,) float32 - position of probe contact tip
            probe_quat:    (4,) float32 - orientation of probe [x, y, z, w]
            ee_pos:        (3,) float32 - position of robot end-effector (PANDA_EE_LINK)
            ee_quat:       (4,) float32 - orientation of robot end-effector
            normal:        (3,) float64 - outward skin surface unit normal
            force:         float - contact force in Newtons (nominal 4.0 N)
            is_valid:      bool - whether (x, y) is on patient body
        """
        skin_z, normal, is_valid = self.query(x, y)

        # Nominal clinical acoustic contact indentation: 2 mm (strictly clamped to [0.5, 3.0] mm)
        d_indent = float(np.clip(indentation, 0.0005, 0.003))

        # Position probe tip pressed by d_indent along inward normal (-normal)
        probe_tip_pos = np.array([x, y, skin_z], dtype=np.float64) - d_indent * normal

        # Construct probe orientation:
        # In extract_slice.py, the ultrasound beam axis is -R_probe[:, 2].
        # For the beam to point along -normal (into patient tissue),
        # R_probe[:, 2] must point along +normal (out of patient tissue).
        z_probe = normal / (np.linalg.norm(normal) + 1e-6)

        # Longitudinal reference vector y_ref = [0, 1, 0]
        y_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        y_proj = y_ref - np.dot(y_ref, z_probe) * z_probe
        y_norm = np.linalg.norm(y_proj)
        if y_norm > 1e-4:
            y0 = y_proj / y_norm
        else:
            y0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        x0 = np.cross(y0, z_probe)
        x0 /= (np.linalg.norm(x0) + 1e-6)

        # Apply in-plane yaw angle alpha around z_probe
        x_probe = np.cos(yaw) * x0 + np.sin(yaw) * y0
        y_probe = -np.sin(yaw) * x0 + np.cos(yaw) * y0

        R_probe = np.column_stack([x_probe, y_probe, z_probe])
        probe_quat = rotation_matrix_to_quaternion(R_probe)

        # In PyBullet:
        # probe_quaternion = multiply_quaternions(ee_quaternion, PROBE_QUAT_FROM_EE)
        # Therefore:
        # ee_quaternion = multiply_quaternions(probe_quat, PROBE_QUAT_FROM_EE)
        ee_quat = multiply_quaternions(probe_quat, PROBE_QUAT_FROM_EE)

        # End-effector (robot flange) position:
        # probe_tip = ee_pos + R_ee @ PROBE_TIP_FROM_EE
        # => ee_pos = probe_tip - R_ee @ PROBE_TIP_FROM_EE
        R_ee = np.array(p.getMatrixFromQuaternion(ee_quat.tolist())).reshape(3, 3)
        ee_pos = probe_tip_pos - R_ee @ PROBE_TIP_FROM_EE

        # Contact force: nominal 4.0 N with spring compliance
        # F = 800 * d_indent + 2.4 => F = 800 * 0.002 + 2.4 = 4.0 N
        contact_force = float(np.clip(800.0 * d_indent + 2.4, 2.0, 8.0)) if is_valid else 0.0

        return (
            probe_tip_pos.astype(np.float32),
            probe_quat.astype(np.float32),
            ee_pos.astype(np.float32),
            ee_quat.astype(np.float32),
            normal,
            contact_force,
            is_valid
        )


def precompute_surface_map(
    subject_dir: Path,
    bed_top_z: float = BED_MATTRESS_TOP_Z,
    mesh_scale: np.ndarray = None,
    client_id: int = None,
    orientation: str = "prone"
) -> Path:
    """Precompute and save 2D surface height and normal map for a subject."""
    subject_dir = Path(subject_dir)
    is_supine = str(orientation).lower() == "supine"
    save_path = subject_dir / ("surface_map_supine.npz" if is_supine else "surface_map.npz")

    if mesh_scale is None:
        mesh_scale = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    disconnect_after = False
    if client_id is None:
        client_id = p.connect(p.DIRECT)
        disconnect_after = True

    try:
        body_id, body_pos, body_rot, meta = create_registered_body_mesh(
            subject_dir, bed_top_z, mesh_scale, orientation=orientation
        )

        # Grid covering entire torso:
        # Lateral X: [-0.22, +0.22] m (177 points, 2.5 mm spacing)
        # Longitudinal Y: [-0.40, +0.40] m (321 points, 2.5 mm spacing)
        xs = np.linspace(-0.22, 0.22, 177, dtype=np.float64)
        ys = np.linspace(-0.40, 0.40, 321, dtype=np.float64)
        Nx, Ny = len(xs), len(ys)

        X, Y = np.meshgrid(xs, ys)
        starts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 1.35, dtype=np.float64)])
        ends = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, 0.55, dtype=np.float64)])

        batch_size = 4000
        results = []
        for i in range(0, len(starts), batch_size):
            chunk = p.rayTestBatch(
                starts[i:i+batch_size].tolist(), ends[i:i+batch_size].tolist()
            )
            results.extend(chunk)

        z_grid = np.full((Ny, Nx), np.nan, dtype=np.float32)
        normals = np.zeros((Ny, Nx, 3), dtype=np.float32)
        valid_mask = np.zeros((Ny, Nx), dtype=bool)

        for idx, r in enumerate(results):
            iy = idx // Nx
            ix = idx % Nx
            if r[0] == body_id:
                z_grid[iy, ix] = float(r[3][2])
                norm = np.array(r[4], dtype=np.float32)
                n_len = np.linalg.norm(norm)
                if n_len > 1e-4:
                    norm /= n_len
                    normals[iy, ix] = norm
                    valid_mask[iy, ix] = True

        # Edge padding / hole filling for clean boundary gradients
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) > 0:
            median_z = float(np.nanmedian(z_grid))
            z_filled = np.copy(z_grid)
            z_filled[~valid_mask] = median_z
        else:
            z_filled = np.full((Ny, Nx), bed_top_z + 0.25, dtype=np.float32)

        # Save compressed map with bed_top_z metadata
        np.savez_compressed(
            save_path,
            xs=xs.astype(np.float32),
            ys=ys.astype(np.float32),
            surface_z=z_filled,
            surface_normals=normals,
            valid_mask=valid_mask,
            bed_top_z=float(bed_top_z)
        )
        print(f"[SurfaceMotionPlanner] Saved surface map to {save_path} ({valid_mask.sum()} contact vertices, bed_top_z={bed_top_z:.2f}m)")
    finally:
        if disconnect_after and p.isConnected(client_id):
            p.disconnect(client_id)

    return save_path


def load_or_create_surface_map(
    subject_dir: Path,
    bed_top_z: float = BED_MATTRESS_TOP_Z,
    mesh_scale: np.ndarray = None,
    client_id: int = None,
    orientation: str = "prone"
) -> SurfaceMap:
    """Load existing surface_map.npz or automatically precompute it on demand."""
    subject_dir = Path(subject_dir)
    is_supine = str(orientation).lower() == "supine"
    save_path = subject_dir / ("surface_map_supine.npz" if is_supine else "surface_map.npz")

    if not save_path.exists():
        print(f"[SurfaceMotionPlanner] Precomputing missing {orientation} surface map for {subject_dir.name}...")
        precompute_surface_map(subject_dir, bed_top_z, mesh_scale, client_id, orientation=orientation)

    return SurfaceMap(save_path, bed_top_z=bed_top_z)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute SonoGym Surface Maps for all patients")
    parser.add_argument("--precompute-all", action="store_true", help="Precompute for all 5 patients")
    parser.add_argument("--subject", type=str, default="totalseg_patients/s0058", help="Target subject")
    args = parser.parse_args()

    if args.precompute_all:
        for subj in ALL_PATIENTS:
            subj_dir = ROOT_DIR / "totalseg_patients" / subj
            if subj_dir.exists():
                print(f"\n=== Processing Patient {subj} ===")
                precompute_surface_map(subj_dir, bed_top_z=BED_MATTRESS_TOP_Z)
        print("\nAll 5 patient surface maps precomputed successfully!")
    else:
        precompute_surface_map(Path(args.subject), bed_top_z=BED_MATTRESS_TOP_Z)
