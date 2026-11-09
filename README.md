# SonoGymLite: Lightweight Simulator for Learning Autonomous Robotic Ultrasound

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29%2B-brightgreen.svg)](https://gymnasium.farama.org/)
[![PyBullet](https://img.shields.io/badge/PyBullet-v3.2%2B-orange.svg)](https://pybullet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Lakshay Batta and Deepak Raina**

<br>

<img src="assets/sonogymlite_overview.jpg" alt="SonoGymLite Simulation Loop & Architecture" width="750">

</div>

---

## Overview

**SonoGymLite** is a lightweight, accessible, CPU-friendly robotic ultrasound simulation and benchmarking environment designed for developing and evaluating autonomous ultrasound scanning policies. Built on **PyBullet** physics and **OpenAI Gymnasium**, SonoGymLite couples Franka Emika Panda manipulator kinematics with ultrasound image synthesis from patient CT and a surface-following probe controller.

Unlike traditional GPU-intensive wave solvers, SonoGymLite executes at high throughput on standard commodity hardware:
- **Workstation:** $120.1\,\text{steps/s}$ ($8.3\,\text{ms/step}$).
- **Laptop (Apple Silicon M1):** $48.4 \pm 18.6\,\text{steps/s}$.

```text
                                       ┌────────────────────────────────┐
                                       │     Patient CT Volumes         │
                                       │    (TotalSegmentator N=5)      │
                                       └──────────────┬─────────────────┘
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
        ┌─────────────────────────────┐                               ┌─────────────────────────────┐
        │   Physics Ultrasound Engine │                               │   Learning-Based Generator  │
        │  Vectorized Ray-Convolution │                               │  2-Channel Deep Spine U-Net │
        │     (50–120 FPS on CPU)     │                               │   (Cavalcanti Clinical US)  │
        └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                       │                                                             │
                       └──────────────────────────────┬──────────────────────────────┘
                                                      │
                                                      ▼
                                       ┌────────────────────────────────┐
                                       │    Gymnasium Environment       │
                                       │     (RoboticUltrasound-v0)     │
                                       └──────────────┬─────────────────┘
                                                      │
         ┌────────────────────────────────────────────┼────────────────────────────────────────────┐
         ▼                                            ▼                                            ▼
┌──────────────────────────────┐       ┌──────────────────────────────┐       ┌──────────────────────────────┐
│   Task 1: Navigation         │       │   Task 2: Reconstruction     │       │   Task 3: Needle Guidance    │
│ • Goal-Conditioned Alignment │       │ • Trajectory Coverage Sweep  │       │ • Probe-Mounted Guide (URDF) │
│ • Surface Admittance Contact │       │ • Downstream 3D Compounding  │       │ • Specular Visibility Law    │
│ • PPO, SAC, A2C, ACT, BC     │       │ • Spine Mesh Reconstruction  │       │ • CT Bone Corridor Planning  │
└──────────────────────────────┘       └──────────────────────────────┘       └──────────────────────────────┘
```

---

## Key Features

1. **High-Throughput CPU Simulation:**
   - Evaluates robotic ultrasound scanning rollouts on standard multi-core CPUs without requiring dedicated GPU clusters.
   - Benchmark throughput: $120.1\,\text{steps/s}$ on standard workstation, $48.4 \pm 18.6\,\text{steps/s}$ on Apple Silicon M1 laptop.

2. **Dual Ultrasound Synthesis Engines:**
   - **Model-Based (MB):** Pure NumPy vectorized ray-tracing and convolution simulating acoustic impedance boundaries, Rayleigh backscatter speckle, bone cortical reflection, acoustic shadowing, and anechoic vascular lumen (`US_LABEL_FLUID = 5`).
   - **Learning-Based (LB):** 2-channel deep convolutional U-Net generator trained on clinical volunteer sweeps (Cavalcanti Spine Dataset).

3. **Three Clinical Benchmark Tasks:**
   - **Task 1 (Target Navigation):** Autonomous alignment of the ultrasound probe with the landmark of a target anatomy for three anatomies (Lumbar Spine L4, Liver Portal Vein, Abdominal Aorta Bifurcation) evaluated over in-domain (`s0011`, `s0058`, `s0250`) and held-out test patients (`s0344`, `s0358`).
   - **Task 2 (Volumetric Reconstruction):** Autonomous scanning of the surface of the target anatomy. PPO with a spatial memory covers about 60% of the upper spine surface on unseen patients.
   - **Task 3 (In-Plane Needle Guidance):** Biopsy guide attachment (`robotic_probe_needle_guide.urdf`) with in-plane ultrasound needle visualization, specular reflection physics ($V = \cos\theta$), and 24-azimuth bone corridor planning for obstacle-free trajectory selection.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/lakshay-ctrl/SonoGymLite.git
cd SonoGymLite
```

### 2. Set Up Environment
We recommend Python 3.10, 3.11, or 3.12:
```bash
conda create -n sonogymlite python=3.11 -y
conda activate sonogymlite

pip install -r requirements.txt
```

---

## Patient Data & Attribution

### Reference Patient
Subject `s0058` is included directly in this repository under `totalseg_patients/s0058/` (containing `ct.nii.gz`, segmentation masks, `patient_skin.obj`, `bone_label.nii.gz`, and precomputed surface height maps) to provide an immediately runnable reference patient out of the box.

### Full 5-Patient Cohort
The full evaluation cohort comprises five subjects:
- **In-Domain Training:** `s0011`, `s0058`, `s0250`
- **Held-Out Test:** `s0344`, `s0358`

Precomputed surface maps, registration metadata, and segmentation masks for all five subjects are provided in `totalseg_patients/`. To run multi-patient training across the remaining four subjects, download their corresponding full-body CT scans from the TotalSegmentator dataset on Zenodo:
- **Zenodo Repository:** [https://zenodo.org/records/6802614](https://zenodo.org/records/6802614) (DOI: `10.5281/zenodo.6802614`)
- Place the downloaded CT files as `ct.nii.gz` in each corresponding `totalseg_patients/<subject_id>/` directory.

### Dataset Attribution & License
The anatomical CT volumes and segmentations used in this platform are sourced from the **TotalSegmentator** dataset:
> Wasserthal, J., Breit, H. C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., & Segeroth, M. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images. *Radiology: Artificial Intelligence*, 5(5), e230024.

TotalSegmentator is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

---

## Quickstart & Interactive Demos

### 1. Environment Verification
Verify that the simulation environment, robot kinematics, and physics step cleanly:
```bash
python3 test_gym_env.py --headless
```

### 2. Interactive Ultrasound Scanning Simulator
Manually control the robotic probe over the patient body with real-time B-mode ultrasound feedback:
```bash
python3 live_unet_demo.py --sim-mode conv
```

### 3. Interactive Needle Guidance Visualizer (Task 3)
Launch the multi-viewport visualizer showing the 3D PyBullet environment, live CT cross-section with bone corridors, and real-time synthesized B-mode ultrasound:
```bash
python3 live_needle_experiment.py --anatomy aorta --angle 45
```
**Keyboard Controls:**
- `[1]` / `[2]` / `[3]` : Switch guide angle (30° / 45° / 60°)
- `[SPACE]` : Toggle automatic cannula insertion to target
- `[E]` / `[Q]` : Advance / retract cannula manually
- `[C]` : Toggle visual servoing
- `[O]` : Toggle target organ (Aorta Bifurcation $\leftrightarrow$ Liver Portal Vein)
- `[P]` : Cycle patient cohort (`s0058` $\to$ `s0011` $\to$ `s0250` $\to$ `s0344` $\to$ `s0358`)
- `[ESC]` : Exit visualizer

### 4. Automated Video Showcase Generator
Generate an automated multi-panel video demonstrating safe acoustic corridor selection, smooth cannula advance, and target localization:
```bash
python3 record_needle_showcase.py --output needle_guidance_showcase.mp4
```

---

## Policy Training

### Train Navigation Agents (Task 1)
```bash
# Train PPO on Spine with Multi-Patient Domain Randomization
python3 train_ppo.py --timesteps 100000 --task navigation --anatomy spine --sim-mode conv --multi-patient

# Train SAC on Liver Portal Vein
python3 train_sac.py --timesteps 50000 --task navigation --anatomy liver --sim-mode conv --buffer-size 10000

# Train on Abdominal Aorta (Supine Orientation)
python3 train_aorta_gpu.py --algo ppo --timesteps 100000 --multi-patient
```

### Train Imitation Learning Agents
```bash
# Train Action Chunking with Transformers (ACT) policy
python3 train_act.py --anatomy spine --epochs 50

# Train Behavioral Cloning (BC) baseline
python3 train_bc.py --anatomy spine --epochs 50
```

---

## Reproducing Paper Results

The evaluation commands below use 100 trials and 300-step episodes, as in the paper. Pretrained checkpoints are not provided, so first train the policies with the commands in the Policy Training section.

```bash
# Table 1: Goal-Conditioned Navigation Benchmark (100 trials across algorithms and anatomies)
python3 run_full_benchmark.py --episodes 100 --max-steps 300

# Table 2: Image-Only Navigation Benchmark
python3 eval_sonogym_benchmark.py --episodes-per-patient 100 --max-steps 300 --anatomy spine --sim-mode conv

# Table 3: Volumetric Bone Reconstruction Coverage Comparison
python3 compare_reconstruction_sweep.py --max-steps 300

# Supplementary Material: Needle Geometric Feasibility & Acoustic Visibility Verification
python3 scripts/step5_needle_verify.py --all

# Supplementary Material: Navigation-to-Needle Hit Rate Coupling Benchmark
python3 scripts/step5b_needle_nav_hit.py
```

All raw outputs, machine-readable JSON logs, and summary tables are logged to `results/`.

---

## Repository Structure

```text
SonoGymLite/
├── robotic_us_env.py                       # Core Gymnasium environment (RoboticUltrasound-v0)
├── live_unet_demo.py                       # Simulation hub & Model-Based US synthesizer
├── surface_motion_planner.py               # Compliant surface controller & precomputed height maps
├── extract_slice.py                        # Oblique/orthogonal CT and segmentation reslicer
├── registration.py                         # Voxel-to-world coordinate transformations
├── image_quality_metric.py                 # Speckle statistics & acoustic shadow detection metric
├── robotic_probe_needle_guide.urdf         # Robot probe with integrated needle guide assembly
├── live_needle_experiment.py               # Interactive visualizer & video recorder
├── record_needle_showcase.py               # Automated video showcase generator
├── test_gym_env.py                         # Environment sanity test & throughput benchmark
├── train_ppo.py, train_sac.py, train_act.py# Reinforcement and imitation learning training scripts
├── eval_sonogym_benchmark.py               # Multi-algorithm benchmark evaluation engine
├── run_full_benchmark.py                   # Master evaluation runner (Table 1)
├── evaluate_needle_study.py                # Multi-patient needle insertion benchmark
├── scripts/
│   ├── step5_needle_verify.py              # Geometric & acoustic feasibility verification (N1-N5)
│   ├── step5b_needle_nav_hit.py            # Navigation-to-needle hit rate coupling benchmark
│   └── generate_publication_reconstruction_figure.py
├── results/                                # Empirical benchmark JSONs, tables, and reports
├── aorta_bifurcation_targets.json          # 6-DoF target database for Aorta Bifurcation
├── portal_vein_targets.json                # 6-DoF target database for Liver Portal Vein
├── l4_svp_targets.json                     # 6-DoF target database for Lumbar Spine L4
├── totalseg_patients/                      # 5-patient cohort (s0011, s0058, s0250, s0344, s0358)
│   └── s0058/                              # Runnable reference patient assets (CT, mesh, bone, maps)
├── requirements.txt                        # Python dependencies
└── LICENSE                                 # MIT License
```

---

## Citation

If you use SonoGymLite in your research, please cite:

```bibtex
@article{batta2026sonogymlite,
  title={SonoGymLite: Lightweight Simulator for Learning Autonomous Robotic Ultrasound},
  author={Batta, Lakshay and Raina, Deepak},
  journal={Journal of Medical Robotics Research (under review)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
Anatomical patient data is redistributed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.
