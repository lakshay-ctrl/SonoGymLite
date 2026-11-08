# Navigation Gaps Benchmark Report

Evaluated on 100 trials per cohort (Train: seeds 1000-1099, Held-Out: seeds 2000-2099, Horizon: 300 steps).
Success criterion: `pos_error < 5.0 mm` and `rot_error < 10.0 deg`.

| Model / Baseline | Mode | Cohort | Pos Error (mm) | Rot Error (deg) | Final Succ (%) | First-Succ (%) | Steps |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spine MB Goal-A2C (100k)** | Fixed | Train | 82.26 ± 29.61 | 8.01 ± 6.63° | 0.0% | 3.0% | 300.0 |
| **Spine MB Goal-A2C (100k)** | Fixed | Held-Out | 79.14 ± 31.61 | 20.73 ± 17.22° | 0.0% | 4.0% | 300.0 |
| **Spine LB Goal-PPO (100k)** | Fixed | Train | 0.85 ± 0.80 | 0.17 ± 0.52° | 100.0% | 100.0% | 300.0 |
| **Spine LB Goal-PPO (100k)** | Fixed | Held-Out | 2.27 ± 1.42 | 11.58 ± 11.42° | 51.0% | 100.0% | 300.0 |
| **Spine LB Goal-SAC (100k)** | Fixed | Train | 2.28 ± 0.50 | 0.29 ± 0.39° | 100.0% | 100.0% | 300.0 |
| **Spine LB Goal-SAC (100k)** | Fixed | Held-Out | 4.13 ± 1.90 | 5.35 ± 3.71° | 77.0% | 100.0% | 300.0 |
| **Spine LB Image-Only PPO (Run 4b, 300k)** | Fixed | Train | 304.00 ± 85.16 | 19.96 ± 28.65° | 0.0% | 7.0% | 300.0 |
| **Spine LB Image-Only PPO (Run 4b, 300k)** | Fixed | Held-Out | 344.62 ± 59.87 | 44.17 ± 41.03° | 0.0% | 5.0% | 300.0 |
| **Spine LB Image-Only PPO (Run 4b, 300k)** | Early Term | Train | 304.00 ± 85.16 | 19.96 ± 28.65° | 0.0% | -- | 78.4 ± 25.1 |
| **Spine LB Image-Only PPO (Run 4b, 300k)** | Early Term | Held-Out | 344.62 ± 59.87 | 44.17 ± 41.03° | 0.0% | -- | 103.4 ± 43.7 |