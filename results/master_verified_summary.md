# Autonomous Robotic Ultrasound: Master Verified Results Summary

> **Verification Status**: PASSED & FINALIZED
> **Project Directory**: `lakshya/ct_us`

---
## 1. Verified Model Checkpoints

| Category | Model / Target | Status | Size (MB) | File |
| :--- | :--- | :---: | :---: | :--- |
| Nav GC PPO | Spine MB | ✅ Verified | 76.7 | `ppo_mb_spine_final_model.zip` |
| Nav GC SAC | Spine MB | ✅ Verified | 39.7 | `sac_mb_spine_final_model.zip` |
| Nav GC PPO | Aorta MB | ✅ Verified | 76.7 | `ppo_mb_aorta_final_model.zip` |
| Nav GC SAC | Aorta MB | ✅ Verified | 39.7 | `sac_mb_aorta_final_model.zip` |
| Nav GC A2C | Aorta MB | ✅ Verified | 10.3 | `a2c_mb_aorta_final_model.zip` |
| Nav GC PPO | Liver MB | ✅ Verified | 76.7 | `ppo_mb_liver_final_model.zip` |
| Nav GC SAC | Liver MB | ✅ Verified | 39.7 | `sac_mb_liver_final_model.zip` |
| Nav GC A2C | Liver MB | ✅ Verified | 10.3 | `a2c_mb_liver_final_model.zip` |
| Recon PPO+Mem | Spine MB | ✅ Verified | 18.7 | `final_model.zip` |
| Recon PPO | Spine MB | ✅ Verified | 20.3 | `final_model.zip` |
| Recon PPO+Mem | Aorta MB | ✅ Verified | 18.7 | `final_model.zip` |
| Recon PPO | Aorta MB | ✅ Verified | 20.3 | `final_model.zip` |
| Recon PPO+Mem | Liver MB | ✅ Verified | 18.7 | `final_model.zip` |
| Recon PPO | Liver MB | ✅ Verified | 20.3 | `final_model.zip` |
| Image-Only PPO | Spine MB (s0) | ✅ Verified | 30.9 | `ppo_scale_spine_s0_final.zip` |
| Image-Only PPO | Aorta MB (s0) | ✅ Verified | 30.9 | `ppo_scale_aorta_s0_final.zip` |
| Image-Only PPO | Aorta MB (s1) | ✅ Verified | 30.9 | `ppo_scale_aorta_s1_final.zip` |
| Image-Only PPO | Aorta MB (s2) | ✅ Verified | 30.9 | `ppo_scale_aorta_s2_final.zip` |
| Image-Only PPO | Liver MB (300k) | ✅ Verified | 14.9 | `ppo_scale_liver_300000_steps.zip` |
| Image-Only SAC | Spine MB | ✅ Verified | 39.7 | `sac_image_only_spine_mb_final.zip` |
| Image-Only SAC | Spine LB | ✅ Verified | 39.7 | `sac_image_only_spine_lb_final.zip` |
| Image-Only SAC | Aorta MB | ✅ Verified | 39.7 | `sac_image_only_aorta_mb_final.zip` |
| Image-Only SAC | Liver MB | ✅ Verified | 39.7 | `sac_image_only_liver_mb_final.zip` |

---
## 2. Table I: Goal-Conditioned Navigation Benchmark
Fixed-horizon 150 steps evaluation. Success criterion: Position error < 5.0 mm AND Rotation error < 10.0°.

| Target Anatomy | Algorithm | Cohort | 3D Pos Error (mm) | Rot Error (deg) | Success Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Spine | GC_PPO | Train | 1.46 ± 0.90 | 0.19 ± 0.45° | 100.0% |
| Spine | GC_PPO | Held-Out | 6.03 ± 2.84 | 9.37 ± 11.39° | 50.0% |
| Spine | GC_SAC | Train | 2.49 ± 1.05 | 0.45 ± 1.12° | 100.0% |
| Spine | GC_SAC | Held-Out | 2.98 ± 2.15 | 9.36 ± 12.60° | 72.0% |
| Spine | IMAGE_ONLY_PPO | Train | 135.48 ± 137.02 | 14.39 ± 20.39° | 14.0% |
| Spine | IMAGE_ONLY_PPO | Held-Out | 240.37 ± 135.59 | 33.34 ± 29.41° | 0.0% |
| Aorta | GC_PPO | Train | 1.18 ± 0.40 | 1.50 ± 1.44° | 100.0% |
| Aorta | GC_PPO | Held-Out | 1.22 ± 0.78 | 0.59 ± 0.30° | 100.0% |
| Aorta | GC_SAC | Train | 5.49 ± 3.46 | 5.07 ± 3.70° | 57.0% |
| Aorta | GC_SAC | Held-Out | 4.16 ± 1.50 | 3.31 ± 2.76° | 53.0% |
| Aorta | IMAGE_ONLY_PPO | Train | 5.52 ± 2.87 | 4.88 ± 3.68° | 55.0% |
| Aorta | IMAGE_ONLY_PPO | Held-Out | 176.88 ± 54.95 | 32.88 ± 21.64° | 0.0% |
| Aorta | IMAGE_ONLY_PPO_S0 | Train | 14.40 ± 32.40 | 7.87 ± 8.11° | 30.0% |
| Aorta | IMAGE_ONLY_PPO_S0 | Held-Out | 158.42 ± 43.98 | 32.28 ± 25.47° | 0.0% |
| Aorta | IMAGE_ONLY_PPO_S2 | Train | 12.93 ± 25.19 | 10.78 ± 9.65° | 36.0% |
| Aorta | IMAGE_ONLY_PPO_S2 | Held-Out | 146.54 ± 68.59 | 29.88 ± 22.24° | 0.0% |
| Liver | GC_PPO | Train | 1.02 ± 0.45 | 0.65 ± 0.72° | 100.0% |
| Liver | GC_PPO | Held-Out | 1.95 ± 0.89 | 6.75 ± 7.47° | 62.0% |
| Liver | GC_SAC | Train | 4.41 ± 1.46 | 1.89 ± 2.35° | 70.0% |
| Liver | GC_SAC | Held-Out | 4.79 ± 1.54 | 8.34 ± 9.11° | 33.0% |

---
## 3. Table II: 3D Surface Reconstruction with Yaw Rotation
300 steps evaluation (10 random starts per patient, 3 train, 2 held-out).

| Target Organ | Method | Cohort | Upper Coverage (%) | Path Length (m) | Total Yaw Rotation (deg) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Aorta | ppo_memory | Train | 12.4 ± 7.0% | 0.41 ± 0.27 m | **137.6 ± 104.1°** |
| Aorta | ppo_memory | Held-Out | 23.5 ± 3.1% | 0.12 ± 0.03 m | **57.4 ± 14.6°** |
| Aorta | ppo_nomem | Train | 15.0 ± 9.1% | 0.42 ± 0.05 m | **88.9 ± 50.8°** |
| Aorta | ppo_nomem | Held-Out | 33.6 ± 4.3% | 0.32 ± 0.08 m | **68.2 ± 39.6°** |
| Liver | ppo_memory | Train | 42.3 ± 5.6% | 1.01 ± 0.54 m | **242.8 ± 127.8°** |
| Liver | ppo_memory | Held-Out | 54.5 ± 5.7% | 1.47 ± 1.13 m | **181.0 ± 32.2°** |
| Liver | ppo_nomem | Train | 48.7 ± 7.4% | 1.11 ± 0.34 m | **1519.2 ± 1064.5°** |
| Liver | ppo_nomem | Held-Out | 67.4 ± 16.6% | 0.81 ± 0.61 m | **1061.0 ± 726.8°** |

---
## 4. Table III: Image-Only PPO Navigation (Fixed Horizon 300 Steps)

| Target | Model | Cohort | Pos Error (mm) | Rot Error (deg) | Success Rate (%) | Steps |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Aorta | Aorta Image-Only PPO Seed 0 | Train | 12.02 ± 28.25 | 6.81 ± 7.42° | 25.0% | 300.0 |
| Aorta | Aorta Image-Only PPO Seed 0 | Held-Out | 148.62 ± 48.47 | 38.54 ± 29.67° | 0.0% | 222.6 |
| Aorta | Aorta Image-Only PPO Seed 1 | Train | 8.17 ± 21.90 | 5.62 ± 5.25° | 53.0% | 297.9 |
| Aorta | Aorta Image-Only PPO Seed 1 | Held-Out | 176.85 ± 59.34 | 33.99 ± 20.84° | 0.0% | 232.8 |
| Aorta | Aorta Image-Only PPO Seed 2 | Train | 17.16 ± 35.71 | 11.31 ± 12.44° | 39.0% | 298.8 |
| Aorta | Aorta Image-Only PPO Seed 2 | Held-Out | 155.85 ± 61.92 | 23.22 ± 16.96° | 0.0% | 267.4 |
| Liver | Liver Image-Only PPO (300k) | Train | 22.02 ± 16.12 | 5.72 ± 5.73° | 2.0% | 300.0 |
| Liver | Liver Image-Only PPO (300k) | Held-Out | 46.62 ± 32.27 | 19.47 ± 19.05° | 0.0% | 300.0 |

---
## 5. Table IV: Image-Only SAC Suite (300k Steps, 100 Trials Each)

| Model Setting | Eval Mode | Cohort | Pos Error (mm) | Rot Error (deg) | Success Rate (%) | Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Spine MB Image-Only SAC | fixed | Train | 6.16 ± 3.73 | 2.45 ± 4.39° | 41.0% | 300.0 |
| Spine MB Image-Only SAC | fixed | Held-Out | 115.92 ± 94.45 | 31.13 ± 27.49° | 1.0% | 267.9 |
| Spine MB Image-Only SAC | early_term | Train | 3.04 ± 1.28 | 0.75 ± 1.42° | 100.0% | 10.3 |
| Spine MB Image-Only SAC | early_term | Held-Out | 65.15 ± 90.90 | 16.21 ± 23.29° | 53.0% | 150.2 |
| Spine LB Image-Only SAC | fixed | Train | 14.47 ± 28.17 | 2.92 ± 4.50° | 53.0% | 300.0 |
| Spine LB Image-Only SAC | fixed | Held-Out | 151.62 ± 88.93 | 32.40 ± 23.48° | 0.0% | 296.3 |
| Spine LB Image-Only SAC | early_term | Train | 8.62 ± 26.55 | 1.34 ± 2.59° | 95.0% | 29.0 |
| Spine LB Image-Only SAC | early_term | Held-Out | 134.72 ± 98.62 | 30.65 ± 24.38° | 16.0% | 261.6 |
| Aorta MB Image-Only SAC | fixed | Train | 4.74 ± 2.47 | 5.53 ± 5.30° | 56.0% | 300.0 |
| Aorta MB Image-Only SAC | fixed | Held-Out | 62.86 ± 47.54 | 12.58 ± 9.47° | 2.0% | 300.0 |
| Aorta MB Image-Only SAC | early_term | Train | 3.14 ± 1.23 | 2.81 ± 2.36° | 100.0% | 10.9 |
| Aorta MB Image-Only SAC | early_term | Held-Out | 36.32 ± 45.47 | 6.78 ± 7.36° | 46.0% | 189.5 |
| Liver MB Image-Only SAC | fixed | Train | 24.41 ± 40.13 | 9.63 ± 17.02° | 34.0% | 298.9 |
| Liver MB Image-Only SAC | fixed | Held-Out | 140.58 ± 96.57 | 17.44 ± 15.01° | 0.0% | 299.5 |
| Liver MB Image-Only SAC | early_term | Train | 2.90 ± 1.26 | 1.43 ± 1.65° | 100.0% | 8.2 |
| Liver MB Image-Only SAC | early_term | Held-Out | 113.58 ± 103.37 | 15.08 ± 15.24° | 19.0% | 252.1 |
