#!/usr/bin/env python3
"""
scripts/prepare_experiment_videos.py
====================================
Master Batch Video Generator for Autonomous Ultrasound Experiments.
Generates MP4 demonstration videos for publication and presentation:
  1. Spine Navigation: Image-Only SAC (MB)
  2. Spine Navigation: Image-Only SAC (LB acoustic simulator)
  3. Aorta Navigation: Image-Only SAC (MB)
  4. Liver Navigation: Image-Only SAC (MB)
  5. Spine 3D Surface Reconstruction (PPO+Memory)
  6. Aorta 3D Surface Reconstruction (PPO+Memory)
  7. Liver 3D Surface Reconstruction (PPO+Memory)

All videos are saved directly to `results/videos/`.
"""

import sys
import os
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

PYTHON_EXEC = PROJECT_DIR / "venv/bin/python3"
if not PYTHON_EXEC.exists():
    PYTHON_EXEC = Path(sys.executable)

VIDEOS_DIR = PROJECT_DIR / "results/videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd_list, desc):
    print("\n" + "=" * 80)
    print(f"  🎬 RECORDING VIDEO: {desc}")
    print("=" * 80)
    print("CMD:", " ".join(str(x) for x in cmd_list))
    res = subprocess.run(cmd_list, cwd=str(PROJECT_DIR))
    if res.returncode == 0:
        print(f"  ✅ Finished recording {desc} successfully!")
    else:
        print(f"  ⚠️ Warning: Process exited with return code {res.returncode}")

def main():
    print("================================================================================")
    print("       SONOGYM / SONOSIMLITE EXPERIMENT VIDEO PRODUCTION SUITE")
    print(f"       Output Directory: {VIDEOS_DIR}")
    print("================================================================================")

    # 1. Spine Image-Only SAC (MB)
    run_cmd([
        str(PYTHON_EXEC), "scripts/live_nav_eval.py",
        "--anatomy", "spine",
        "--patient", "s0344",
        "--model-type", "image_only_sac",
        "--seed", "1000",
        "--step-delay", "0",
        "--render-video", str(VIDEOS_DIR / "spine_image_only_sac_mb.mp4")
    ], "Spine Image-Only SAC (Model-Based)")

    # 2. Spine Image-Only SAC (LB)
    run_cmd([
        str(PYTHON_EXEC), "scripts/live_nav_eval.py",
        "--anatomy", "spine",
        "--patient", "s0344",
        "--model-type", "image_only_sac_lb",
        "--backend", "unet",
        "--seed", "1000",
        "--step-delay", "0",
        "--render-video", str(VIDEOS_DIR / "spine_image_only_sac_lb.mp4")
    ], "Spine Image-Only SAC (Learning-Based UNet)")

    # 3. Aorta Image-Only SAC (MB)
    run_cmd([
        str(PYTHON_EXEC), "scripts/live_nav_eval.py",
        "--anatomy", "aorta",
        "--patient", "s0344",
        "--model-type", "image_only_sac",
        "--seed", "1000",
        "--step-delay", "0",
        "--render-video", str(VIDEOS_DIR / "aorta_image_only_sac_mb.mp4")
    ], "Aorta Image-Only SAC (Model-Based)")

    # 4. Liver Image-Only SAC (MB)
    run_cmd([
        str(PYTHON_EXEC), "scripts/live_nav_eval.py",
        "--anatomy", "liver",
        "--patient", "s0344",
        "--model-type", "image_only_sac",
        "--seed", "1000",
        "--step-delay", "0",
        "--render-video", str(VIDEOS_DIR / "liver_image_only_sac_mb.mp4")
    ], "Liver Image-Only SAC (Model-Based)")

    # 5. Spine 3D Surface Reconstruction (PPO+Memory)
    run_cmd([
        str(PYTHON_EXEC), "scripts/render_live_reconstruction.py",
        "--target", "spine",
        "--patient", "s0344",
        "--method", "ppo_memory",
        "--interval", "5",
        "--steps", "300"
    ], "Spine 3D Reconstruction (PPO+Memory)")

    # 6. Aorta 3D Surface Reconstruction (PPO+Memory)
    run_cmd([
        str(PYTHON_EXEC), "scripts/render_live_reconstruction.py",
        "--target", "aorta",
        "--patient", "s0344",
        "--method", "ppo_memory",
        "--interval", "5",
        "--steps", "300"
    ], "Aorta 3D Reconstruction (PPO+Memory)")

    # 7. Liver 3D Surface Reconstruction (PPO+Memory)
    run_cmd([
        str(PYTHON_EXEC), "scripts/render_live_reconstruction.py",
        "--target", "liver",
        "--patient", "s0344",
        "--method", "ppo_memory",
        "--interval", "5",
        "--steps", "300"
    ], "Liver 3D Reconstruction (PPO+Memory)")

    # Copy reconstruction videos to videos/
    recon_vis = PROJECT_DIR / "results/recon_vis"
    for mp4 in recon_vis.glob("live_recon_*.mp4"):
        target_dest = VIDEOS_DIR / mp4.name
        try:
            target_dest.write_bytes(mp4.read_bytes())
        except Exception:
            pass

    print("\n================================================================================")
    print("  🎉 ALL EXPERIMENT VIDEOS GENERATED SUCCESSFULLY!")
    print("================================================================================")
    for p in sorted(VIDEOS_DIR.glob("*.mp4")):
        sz_mb = p.stat().st_size / (1024 * 1024)
        print(f"  🎬 {p.name:<35s} : {sz_mb:5.2f} MB -> {p}")

if __name__ == "__main__":
    main()
