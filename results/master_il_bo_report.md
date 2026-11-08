# SonoSimLite Master Baselines Report: Imitation Learning (BC, ACT) & Bayesian Optimization (BO)

**Generated:** 2026-10-30 22:44:38  
**Hard Stop Deadline:** 2026-10-31 17:30:00 (Enforced)  
**Protocols:** 100 trials/cohort (Train: s0011/s0058/s0250, Held-Out: s0344/s0358), horizon=300 steps.  
**Success Criterion:** Position error < 5.0 mm and Rotation error < 10.0 deg.  

---

## SPINE MB Master Evaluation Table

| Method / Baseline | Cohort | Pos (mm) | Rot (deg) | Succ (%) | First-Succ (%) | BO Iter | Env Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC Goal-Conditioned** | Train | 3.45 ± 2.65 | 1.57 ± 4.65° | 92.0% | 96.0% | -- | 300.0 |
| **BC Goal-Conditioned** | Held-Out | 13.77 ± 13.22 | 23.41 ± 21.72° | 15.0% | 54.0% | -- | 300.0 |
| **ACT Goal-Conditioned** | Train | 2.24 ± 0.91 | 0.38 ± 0.62° | 100.0% | 100.0% | -- | 300.0 |
| **ACT Goal-Conditioned** | Held-Out | 2.50 ± 0.67 | 7.72 ± 8.13° | 69.0% | 80.0% | -- | 300.0 |
| **BC Image-Only** | Train | 86.54 ± 33.09 | 11.83 ± 7.57° | 0.0% | 1.0% | -- | 300.0 |
| **BC Image-Only** | Held-Out | 114.55 ± 104.38 | 24.35 ± 21.72° | 0.0% | 1.0% | -- | 300.0 |
| **ACT Image-Only** | Train | 85.13 ± 26.62 | 12.99 ± 7.03° | 0.0% | 21.0% | -- | 300.0 |
| **ACT Image-Only** | Held-Out | 71.85 ± 15.47 | 21.16 ± 15.17° | 0.0% | 2.0% | -- | 300.0 |
| **Target-image BO (Spine MB)** | Train | 53.22 ± 32.68 | 9.64 ± 9.24° | 3.0% | -- | 19.9 | 292.1 |
| **Target-image BO (Spine MB)** | Held-Out | 53.38 ± 33.85 | 21.62 ± 19.62° | 6.0% | -- | 21.6 | 281.7 |
| **Template BO (Spine MB, image-only)** | Train | 66.60 ± 29.94 | 11.16 ± 8.45° | 0.0% | -- | 20.2 | 285.0 |
| **Template BO (Spine MB, image-only)** | Held-Out | 56.78 ± 31.13 | 22.48 ± 17.84° | 1.0% | -- | 21.1 | 288.3 |
| **Quality BO (Spine MB)** | Train | 74.40 ± 30.93 | 11.50 ± 7.18° | 0.0% | -- | 20.8 | 273.3 |
| **Quality BO (Spine MB)** | Held-Out | 55.57 ± 29.35 | 21.42 ± 19.65° | 0.0% | -- | 21.4 | 272.9 |

### Hyperparameters & File Paths
- **BC Training:** Batch size = 64, Learning rate = 3e-4 (Adam, 50 epochs, 10% val split)
- **ACT Training:** Batch size = 64, Learning rate = 1e-4 (AdamW, chunk size = 10, temporal ensembling, 50 epochs)
- **IL Checkpoints:** `il_new/checkpoints/spine/`
- **IL Demos:** `il_new/demos/spine/`
- **BO Targets:** `bo_new/targets/`
- **Master Results:** `il_new/results/spine_il_evaluation_master.json`, `bo_new/results/bo_spine_master_results.json`

---

## AORTA MB Master Evaluation Table

### Step 0 Validity Check Lines
```
Score Ceiling at Goal Pose (different speckle seeds): SSIM = 0.9478, NCC = 0.9579, Dice = 1.0000 -> S_ceil = 0.9686
S at 10 mm away (+X): S = 0.2822 (SSIM=0.5040, NCC=0.2703, Dice=0.0723)
S at 20 mm away (+X): S = 0.2299 (SSIM=0.4772, NCC=0.2124, Dice=0.0000)
S at 40 mm away (+X): S = 0.2536 (SSIM=0.4510, NCC=0.2371, Dice=0.0727)
S at 0.3 rad rotated (+Yaw): S = 0.7147 (SSIM=0.6792, NCC=0.7359, Dice=0.7290)
Threshold Selection: Ceiling 0.9686 >= 0.95 -> Threshold set to default 0.90
```

| Method / Baseline | Cohort | Pos (mm) | Rot (deg) | Succ (%) | First-Succ (%) | BO Iter | Env Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC Goal-Conditioned** | Train | 2.57 ± 1.04 | 2.46 ± 2.03° | 96.0% | 100.0% | -- | 300.0 |
| **BC Goal-Conditioned** | Held-Out | 26.20 ± 4.72 | 19.90 ± 13.69° | 0.0% | 8.0% | -- | 300.0 |
| **ACT Goal-Conditioned** | Train | 3.35 ± 0.90 | 3.68 ± 3.09° | 90.0% | 95.0% | -- | 300.0 |
| **ACT Goal-Conditioned** | Held-Out | 4.14 ± 2.41 | 3.45 ± 5.80° | 64.0% | 64.0% | -- | 300.0 |
| **BC Image-Only** | Train | 140.44 ± 86.45 | 37.30 ± 23.15° | 4.0% | 8.0% | -- | 300.0 |
| **BC Image-Only** | Held-Out | 57.22 ± 31.65 | 22.55 ± 12.17° | 0.0% | 24.0% | -- | 300.0 |
| **ACT Image-Only** | Train | 47.56 ± 23.02 | 14.33 ± 5.35° | 1.0% | 14.0% | -- | 300.0 |
| **ACT Image-Only** | Held-Out | 63.97 ± 30.29 | 17.82 ± 11.85° | 0.0% | 17.0% | -- | 300.0 |
| **Target-image BO (Aorta MB)** | Train | 52.65 ± 34.68 | 15.25 ± 10.17° | 10.0% | -- | 21.6 | 280.4 |
| **Target-image BO (Aorta MB)** | Held-Out | 45.76 ± 32.53 | 16.55 ± 11.64° | 6.0% | -- | 22.0 | 283.6 |
| **Template BO (Aorta MB, image-only)** | Train | 66.56 ± 29.42 | 18.80 ± 9.72° | 1.0% | -- | 20.2 | 276.2 |
| **Template BO (Aorta MB, image-only)** | Held-Out | 71.19 ± 34.95 | 17.77 ± 11.16° | 1.0% | -- | 21.7 | 270.1 |
| **Quality BO (Aorta MB)** | Train | 74.65 ± 32.76 | 19.32 ± 10.06° | 0.0% | -- | 21.8 | 264.0 |
| **Quality BO (Aorta MB)** | Held-Out | 73.07 ± 31.72 | 19.33 ± 9.56° | 0.0% | -- | 22.5 | 265.6 |

### Hyperparameters & File Paths
- **BC Training:** Batch size = 64, Learning rate = 3e-4 (Adam, 50 epochs, 10% val split)
- **ACT Training:** Batch size = 64, Learning rate = 1e-4 (AdamW, chunk size = 10, temporal ensembling, 50 epochs)
- **IL Checkpoints:** `il_new/checkpoints/aorta/`
- **IL Demos:** `il_new/demos/aorta/`
- **BO Targets:** `bo_new/targets/`
- **Master Results:** `il_new/results/aorta_il_evaluation_master.json`, `bo_new/results/bo_aorta_master_results.json`

---

## LIVER MB Master Evaluation Table

### Step 0 Validity Check Lines
```
Score Ceiling at Goal Pose (different speckle seeds): SSIM = 0.9549, NCC = 0.9800, Dice = 1.0000 -> S_ceil = 0.9783
S at 10 mm away (+X): S = 0.7164 (SSIM=0.5372, NCC=0.6810, Dice=0.9309)
S at 20 mm away (+X): S = 0.6843 (SSIM=0.5312, NCC=0.6586, Dice=0.8631)
S at 40 mm away (+X): S = 0.4169 (SSIM=0.3986, NCC=0.2500, Dice=0.6021)
S at 0.3 rad rotated (+Yaw): S = 0.7303 (SSIM=0.5863, NCC=0.6916, Dice=0.9130)
Threshold Selection: Ceiling 0.9783 >= 0.95 -> Threshold set to default 0.90
```

| Method / Baseline | Cohort | Pos (mm) | Rot (deg) | Succ (%) | First-Succ (%) | BO Iter | Env Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC Goal-Conditioned** | Train | 2.61 ± 1.18 | 1.24 ± 1.60° | 97.0% | 100.0% | -- | 300.0 |
| **BC Goal-Conditioned** | Held-Out | 10.08 ± 6.54 | 12.33 ± 15.59° | 15.0% | 57.0% | -- | 300.0 |
| **ACT Goal-Conditioned** | Train | 2.56 ± 1.15 | 2.10 ± 2.04° | 100.0% | 100.0% | -- | 300.0 |
| **ACT Goal-Conditioned** | Held-Out | 1.83 ± 1.12 | 2.13 ± 2.70° | 97.0% | 99.0% | -- | 300.0 |
| **Target-image BO (Liver MB)** | Train | 46.51 ± 33.84 | 10.27 ± 11.55° | 7.0% | -- | 20.8 | 291.1 |
| **Target-image BO (Liver MB)** | Held-Out | 52.53 ± 34.67 | 17.47 ± 17.46° | 8.0% | -- | 20.9 | 288.7 |
| **Quality BO (Liver MB)** | Train | 76.81 ± 32.56 | 16.38 ± 14.01° | 0.0% | -- | 20.6 | 271.7 |
| **Quality BO (Liver MB)** | Held-Out | 77.73 ± 28.20 | 19.82 ± 17.74° | 0.0% | -- | 22.4 | 254.7 |

### Hyperparameters & File Paths
- **BC Training:** Batch size = 64, Learning rate = 3e-4 (Adam, 50 epochs, 10% val split)
- **ACT Training:** Batch size = 64, Learning rate = 1e-4 (AdamW, chunk size = 10, temporal ensembling, 50 epochs)
- **IL Checkpoints:** `il_new/checkpoints/liver/`
- **IL Demos:** `il_new/demos/liver/`
- **BO Targets:** `bo_new/targets/`
- **Master Results:** `il_new/results/liver_il_evaluation_master.json`, `bo_new/results/bo_liver_master_results.json`

---
