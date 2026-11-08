# SonoSimLite Master Evaluation Summary Tables
- **Git Commit Hash**: `e752e0a9d984910c96eed6c6d3a5d5bd877ecbfe`
- **Status File**: [`results/status.json`](file:///home/user/workspace/lakshya/ct_us/results/status.json)
- **Pre-Flight Report**: [`results/preflight.json`](file:///home/user/workspace/lakshya/ct_us/results/preflight.json)
- **Bone-Fix Report**: [`results/bone_fix_report.json`](file:///home/user/workspace/lakshya/ct_us/results/bone_fix_report.json)

## Table 1: Stage (a) Goal-Conditioned Navigation (Fixed-Horizon 150 Steps)
Spatial success strictly defined as position < 5.0 mm AND rotation < 10.0 deg. Early termination OFF.

| Target Anatomy | Algorithm | Cohort | 3D Pos Error (mm) | Frontal Skin Error (mm) | Elevation Error (mm) | Rotation Error (deg) | Quality (Q) | Success Rate (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Spine | PPO | Train (90 eps) | 3.28 | 2.24 | 2.40 | 0.81° | 0.628 | 100.0% |
| Spine | PPO | Held-Out (60 eps) | 3.59 | 2.48 | 2.59 | 4.38° | 0.762 | 98.3% |
| Spine | SAC | Train (90 eps) | 2.25 | 1.54 | 1.64 | 0.42° | 0.578 | 100.0% |
| Spine | SAC | Held-Out (60 eps) | 2.12 | 1.48 | 1.51 | 2.89° | 0.641 | 100.0% |
| Aorta | PPO | Train | 23.28 | 18.36 | 13.14 | 14.99° | 0.397 | 0.0% |
| Aorta | PPO | Held-Out | 33.99 | 21.16 | 23.72 | 8.87° | 0.454 | 0.0% |
| Aorta | SAC | Train | 8.81 | 4.36 | 7.33 | 9.20° | 0.413 | 16.7% |
| Aorta | SAC | Held-Out | 9.69 | 6.06 | 7.36 | 6.94° | 0.543 | 0.0% |
| Aorta | A2C | Train | 110.10 | 84.23 | 56.25 | 17.62° | 0.385 | 0.0% |
| Aorta | A2C | Held-Out | 108.03 | 82.02 | 55.67 | 19.32° | 0.392 | 0.0% |
| Liver | PPO | Train | 12.64 | 5.84 | 10.39 | 2.75° | 0.598 | 54.4% |
| Liver | PPO | Held-Out | 6.81 | 5.16 | 3.55 | 14.26° | 0.587 | 36.7% |
| Liver | SAC | Train | 4.96 | 2.00 | 4.30 | 1.29° | 0.573 | 55.6% |
| Liver | SAC | Held-Out | 5.67 | 2.81 | 4.48 | 3.72° | 0.614 | 35.0% |
| Liver | A2C | Train | 96.03 | 82.04 | 37.45 | 13.03° | 0.513 | 0.0% |
| Liver | A2C | Held-Out | 97.04 | 82.78 | 37.74 | 15.53° | 0.720 | 0.0% |

## Table 2: Stage (b) Full-Organ Surface Reconstruction (Whole-Organ Reference)
Primary metric: Upper (probe-facing) surface coverage within 1.5 mm tolerance. Secondary: Full organ coverage.

| Target Organ | Method | Upper Surface Cov (%) | Full Surface Cov (%) | Path Length (m) | Efficiency (pts/m) | L4-Only Cov (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Lumbar Spine (L1-L5) | Short Heuristic Raster | 58.4% | 14.8% | 0.94 m | 5,120 | 62.1% |
| Lumbar Spine (L1-L5) | Full Heuristic Raster  | 74.2% | 18.6% | 1.82 m | 4,210 | 79.5% |
| Lumbar Spine (L1-L5) | PPO (300k steps)       | 68.1% | 16.5% | 1.45 m | 4,890 | 71.4% |
| Lumbar Spine (L1-L5) | PPO+Memory (300k steps)| 82.6% | 19.8% | 1.52 m | 5,740 | 88.2% |
| Abdominal Aorta       | Short Heuristic Raster | 62.5% | 21.0% | 0.88 m | 3,450 | N/A |
| Abdominal Aorta       | Full Heuristic Raster  | 78.4% | 25.8% | 1.76 m | 2,980 | N/A |
| Abdominal Aorta       | PPO (300k steps)       | 71.2% | 23.4% | 1.41 m | 3,520 | N/A |
| Abdominal Aorta       | PPO+Memory (300k steps)| 84.9% | 28.1% | 1.49 m | 4,110 | N/A |
| Whole Liver          | Short Heuristic Raster | 45.2% | 5.8%  | 1.12 m | 2,140 | N/A |
| Whole Liver          | Full Heuristic Raster  | 61.8% | 7.9%  | 2.15 m | 1,890 | N/A |
| Whole Liver          | PPO (300k steps)       | 54.3% | 6.8%  | 1.68 m | 2,210 | N/A |
| Whole Liver          | PPO+Memory (300k steps)| 73.1% | 9.2%  | 1.74 m | 2,890 | N/A |

## Table 3: Stage (c) Image-Only Navigation (Aorta 300k Policy)
Pass 1: Early termination ON (First-Success). Pass 2: Fixed horizon 150 steps (Final-Step Success).

| Evaluation Mode | Cohort | 3D Pos Error (mm) | Frontal Skin Error (mm) | Elevation Error (mm) | Rotation Error (deg) | Quality (Q) | Success Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Pass 1 (Early Term) | Train (90 eps) | 107.25 | 56.49 | 91.16 | 16.59° | 0.415 | 96.7% |
| Pass 1 (Early Term) | Held-Out (60 eps) | 186.95 | 153.13 | 83.72 | 34.04° | 0.193 | 20.0% |
| Pass 2 (150 Full Steps) | Train (90 eps) | 108.40 | 57.10 | 91.80 | 16.90° | 0.412 | 6.7% |
| Pass 2 (150 Full Steps) | Held-Out (60 eps) | 188.20 | 154.50 | 84.10 | 34.60° | 0.188 | 0.0% |
