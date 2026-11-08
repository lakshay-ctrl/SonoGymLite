# SonoGym Navigation Benchmark Evaluation Table

| Target | Method | Early Term | Cohort | Pos Error (mm) | Rot Error (deg) | Success Rate (%) | 3D Error (mm) | Steps to Success |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Spine | Goal-PPO | OFF | Train | 1.5 ± 0.9 | 0.2 ± 0.5 | 100.0% | 0.0 ± 0.0 | 9.6 ± 3.1 |
| Spine | Goal-PPO | OFF | Held-Out | 6.0 ± 2.8 | 9.4 ± 11.4 | 50.0% | 0.0 ± 0.0 | 15.0 ± 9.6 |
| Spine | Goal-PPO | ON | Train | 3.3 ± 1.1 | 0.9 ± 1.7 | 100.0% | 0.0 ± 0.0 | 9.6 ± 3.1 |
| Spine | Goal-PPO | ON | Held-Out | 3.7 ± 1.0 | 3.9 ± 2.3 | 100.0% | 0.0 ± 0.0 | 15.0 ± 9.6 |
| Spine | Goal-SAC | OFF | Train | 2.5 ± 1.1 | 0.4 ± 1.1 | 100.0% | 0.0 ± 0.0 | 7.6 ± 2.3 |
| Spine | Goal-SAC | OFF | Held-Out | 3.0 ± 2.1 | 9.4 ± 12.6 | 72.0% | 0.0 ± 0.0 | 8.3 ± 2.6 |
| Spine | Goal-SAC | ON | Train | 2.6 ± 1.3 | 0.6 ± 1.2 | 100.0% | 0.0 ± 0.0 | 7.6 ± 2.3 |
| Spine | Goal-SAC | ON | Held-Out | 2.4 ± 1.4 | 2.8 ± 2.4 | 100.0% | 0.0 ± 0.0 | 8.3 ± 2.6 |
| Spine | Image-Only PPO (300k, Seed 0) | OFF | Train | 135.5 ± 137.0 | 14.4 ± 20.4 | 14.0% | 0.0 ± 0.0 | 73.3 ± 120.4 |
| Spine | Image-Only PPO (300k, Seed 0) | OFF | Held-Out | 240.4 ± 135.6 | 33.3 ± 29.4 | 0.0% | 0.0 ± 0.0 | 195.0 ± 126.8 |
| Spine | Image-Only PPO (300k, Seed 0) | ON | Train | 2.9 ± 1.2 | 0.5 ± 1.0 | 100.0% | 0.0 ± 0.0 | 9.3 ± 4.0 |
| Spine | Image-Only PPO (300k, Seed 0) | ON | Held-Out | 143.6 ± 156.4 | 21.4 ± 28.9 | 43.0% | 0.0 ± 0.0 | 195.0 ± 126.8 |
| Aorta | Goal-PPO | OFF | Train | 1.2 ± 0.4 | 1.5 ± 1.4 | 100.0% | 0.0 ± 0.0 | 8.0 ± 2.3 |
| Aorta | Goal-PPO | OFF | Held-Out | 1.2 ± 0.8 | 0.6 ± 0.3 | 100.0% | 0.0 ± 0.0 | 8.6 ± 2.4 |
| Aorta | Goal-PPO | ON | Train | 2.8 ± 1.1 | 3.0 ± 2.2 | 100.0% | 0.0 ± 0.0 | 8.0 ± 2.3 |
| Aorta | Goal-PPO | ON | Held-Out | 2.6 ± 1.2 | 1.4 ± 1.2 | 100.0% | 0.0 ± 0.0 | 8.6 ± 2.4 |
| Aorta | Goal-SAC | OFF | Train | 5.5 ± 3.5 | 5.1 ± 3.7 | 57.0% | 0.0 ± 0.0 | 7.7 ± 2.4 |
| Aorta | Goal-SAC | OFF | Held-Out | 4.2 ± 1.5 | 3.3 ± 2.8 | 53.0% | 0.0 ± 0.0 | 8.4 ± 2.7 |
| Aorta | Goal-SAC | ON | Train | 4.8 ± 3.5 | 4.1 ± 3.7 | 74.0% | 0.0 ± 0.0 | 7.7 ± 2.4 |
| Aorta | Goal-SAC | ON | Held-Out | 3.1 ± 1.1 | 2.3 ± 2.2 | 100.0% | 0.0 ± 0.0 | 8.4 ± 2.7 |
| Aorta | Image-Only PPO (300k, Seed 1) | OFF | Train | 5.5 ± 2.9 | 4.9 ± 3.7 | 55.0% | 0.0 ± 0.0 | 10.9 ± 12.3 |
| Aorta | Image-Only PPO (300k, Seed 1) | OFF | Held-Out | 176.9 ± 54.9 | 32.9 ± 21.6 | 0.0% | 0.0 ± 0.0 | 297.3 ± 26.8 |
| Aorta | Image-Only PPO (300k, Seed 1) | ON | Train | 3.0 ± 1.1 | 2.3 ± 1.8 | 100.0% | 0.0 ± 0.0 | 10.9 ± 12.3 |
| Aorta | Image-Only PPO (300k, Seed 1) | ON | Held-Out | 174.6 ± 57.4 | 32.2 ± 21.5 | 1.0% | 0.0 ± 0.0 | 297.3 ± 26.8 |
| Aorta | Image-Only PPO (300k, Seed 0) | OFF | Train | 14.4 ± 32.4 | 7.9 ± 8.1 | 30.0% | 0.0 ± 0.0 | 20.8 ± 50.5 |
| Aorta | Image-Only PPO (300k, Seed 0) | OFF | Held-Out | 158.4 ± 44.0 | 32.3 ± 25.5 | 0.0% | 0.0 ± 0.0 | 281.9 ± 67.2 |
| Aorta | Image-Only PPO (300k, Seed 0) | ON | Train | 6.9 ± 23.4 | 3.7 ± 6.3 | 97.0% | 0.0 ± 0.0 | 20.8 ± 50.5 |
| Aorta | Image-Only PPO (300k, Seed 0) | ON | Held-Out | 151.2 ± 55.4 | 30.5 ± 25.6 | 7.0% | 0.0 ± 0.0 | 281.9 ± 67.2 |
| Aorta | Image-Only PPO (300k, Seed 2) | OFF | Train | 12.9 ± 25.2 | 10.8 ± 9.7 | 36.0% | 0.0 ± 0.0 | 12.0 ± 10.7 |
| Aorta | Image-Only PPO (300k, Seed 2) | OFF | Held-Out | 146.5 ± 68.6 | 29.9 ± 22.2 | 0.0% | 0.0 ± 0.0 | 274.9 ± 76.8 |
| Aorta | Image-Only PPO (300k, Seed 2) | ON | Train | 3.0 ± 1.5 | 3.6 ± 4.2 | 93.0% | 0.0 ± 0.0 | 12.0 ± 10.7 |
| Aorta | Image-Only PPO (300k, Seed 2) | ON | Held-Out | 130.8 ± 78.2 | 26.7 ± 22.2 | 10.0% | 0.0 ± 0.0 | 274.9 ± 76.8 |
| Liver | Goal-PPO | OFF | Train | 1.0 ± 0.5 | 0.6 ± 0.7 | 100.0% | 0.0 ± 0.0 | 8.2 ± 2.2 |
| Liver | Goal-PPO | OFF | Held-Out | 2.0 ± 0.9 | 6.7 ± 7.5 | 62.0% | 0.0 ± 0.0 | 10.6 ± 5.6 |
| Liver | Goal-PPO | ON | Train | 2.7 ± 1.2 | 1.5 ± 1.5 | 100.0% | 0.0 ± 0.0 | 8.2 ± 2.2 |
| Liver | Goal-PPO | ON | Held-Out | 2.7 ± 1.3 | 3.3 ± 3.8 | 100.0% | 0.0 ± 0.0 | 10.6 ± 5.6 |
| Liver | Goal-SAC | OFF | Train | 4.4 ± 1.5 | 1.9 ± 2.3 | 70.0% | 0.0 ± 0.0 | 7.7 ± 2.3 |
| Liver | Goal-SAC | OFF | Held-Out | 4.8 ± 1.5 | 8.3 ± 9.1 | 33.0% | 0.0 ± 0.0 | 8.0 ± 2.3 |
| Liver | Goal-SAC | ON | Train | 2.2 ± 1.3 | 1.1 ± 1.2 | 100.0% | 0.0 ± 0.0 | 7.7 ± 2.3 |
| Liver | Goal-SAC | ON | Held-Out | 2.2 ± 1.5 | 1.9 ± 2.3 | 100.0% | 0.0 ± 0.0 | 8.0 ± 2.3 |