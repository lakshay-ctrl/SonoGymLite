"""
compute_qmin.py — Compute median frame Q over raster_full acquisition for each target anatomy.
"""
import sys
import json
from pathlib import Path
import numpy as np

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from robotic_us_env import RoboticUltrasoundGymEnv
TRAIN_PATIENTS = ["s0011", "s0058", "s0250"]
TARGETS = ["spine", "aorta", "liver"]
HORIZON = 150

def compute_qmins():
    qmin_dict = {}
    config_dir = PROJECT_DIR / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        print(f"\n=======================================================")
        print(f"  Acquiring Raster Quality Distribution for: {target.upper()}")
        print(f"=======================================================")
        all_q_vals = []

        for pid in TRAIN_PATIENTS:
            subj_dir = PROJECT_DIR / "totalseg_patients" / pid
            env = RoboticUltrasoundGymEnv(
                subject_dir=str(subj_dir),
                task="reconstruction",
                anatomy=target,
                sim_mode="conv",
                size=128,
                pixel_spacing=1.0,
                max_episode_steps=HORIZON,
                multi_patient=False
            )

            obs, info = env.reset(seed=42)
            patient_q = []

            for step in range(HORIZON):
                dx = 1.0 if (step // 20) % 2 == 0 else -1.0
                dy = 0.6 if step % 20 == 0 else 0.0
                action = np.array([dx, dy, 0.0], dtype=np.float32)

                obs, r, term, trunc, info = env.step(action)
                q_val = float(env.current_quality)
                # Only consider acquired frames in contact (current_force >= 1.0 and valid)
                if env.current_force >= 1.0 and env.current_is_valid:
                    patient_q.append(q_val)
                    all_q_vals.append(q_val)

            env.close()
            med_p = float(np.median(patient_q)) if patient_q else 0.0
            print(f"  [{target}] Patient {pid}: Acquired {len(patient_q)} frames, Median Q = {med_p:.4f}")

        overall_med = float(np.median(all_q_vals)) if all_q_vals else 0.50
        print(f"  ==> [{target.upper()}] Overall Median Q = {overall_med:.4f} (from {len(all_q_vals)} frames)")
        qmin_dict[target] = round(overall_med, 4)

    out_file = config_dir / "qmin.json"
    with open(out_file, "w") as f:
        json.dump(qmin_dict, f, indent=2)
    print(f"\n[DONE] Saved q_min thresholds to {out_file}: {qmin_dict}")
    return qmin_dict

if __name__ == "__main__":
    compute_qmins()
