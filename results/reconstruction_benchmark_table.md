# SonoGym Whole-Organ 3D Reconstruction Benchmark Table

| Target | Cohort | Method | Upper Coverage (%) | Whole Coverage (%) | Path Length (m) | Total Rotation (deg) | Footprint Reach (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Spine | Train | PPO+Memory | 58.5 ± 5.2 | 14.7 ± 1.4 | 0.21 ± 0.01 | 325.5 ± 8.5 | - |
| Spine | Train | PPO (No Memory) | 51.2 ± 4.2 | 12.4 ± 1.2 | 0.37 ± 0.03 | 424.5 ± 20.0 | - |
| Spine | Train | Footprint Raster | 38.5 ± 5.0 | 10.8 ± 1.4 | 1.51 ± 0.01 | 31.0 ± 18.5 | 39.4 ± 2.8 |
| Spine | Train | Short Raster | 23.8 ± 5.1 | 8.0 ± 1.6 | 1.53 ± 0.01 | 66.9 ± 48.8 | - |
| Spine | Held-Out | PPO+Memory | 60.0 ± 5.6 | 16.7 ± 2.5 | 0.21 ± 0.01 | 360.7 ± 37.7 | - |
| Spine | Held-Out | PPO (No Memory) | 48.0 ± 4.8 | 13.1 ± 1.5 | 0.36 ± 0.03 | 477.8 ± 41.7 | - |
| Spine | Held-Out | Footprint Raster | 37.4 ± 3.9 | 13.6 ± 1.1 | 1.55 ± 0.04 | 138.7 ± 135.1 | 36.2 ± 2.5 |
| Spine | Held-Out | Short Raster | 23.0 ± 3.6 | 10.1 ± 1.6 | 1.57 ± 0.05 | 180.7 ± 176.3 | - |
| Aorta | Train | PPO+Memory | 11.9 ± 6.8 | 9.6 ± 5.2 | 0.40 ± 0.28 | 0.0 ± 0.0 | - |
| Aorta | Train | PPO (No Memory) | 14.8 ± 8.9 | 12.4 ± 7.4 | 0.42 ± 0.06 | 0.0 ± 0.0 | - |
| Aorta | Train | Footprint Raster | 10.4 ± 7.8 | 8.6 ± 6.7 | 1.56 ± 0.02 | 0.0 ± 0.0 | 31.1 ± 1.5 |
| Aorta | Train | Short Raster | 2.4 ± 3.4 | 1.1 ± 1.5 | 1.57 ± 0.03 | 0.0 ± 0.0 | - |
| Aorta | Held-Out | PPO+Memory | 23.7 ± 3.3 | 23.0 ± 3.6 | 0.12 ± 0.03 | 0.0 ± 0.0 | - |
| Aorta | Held-Out | PPO (No Memory) | 33.5 ± 4.3 | 33.2 ± 4.2 | 0.33 ± 0.08 | 0.0 ± 0.0 | - |
| Aorta | Held-Out | Footprint Raster | 39.3 ± 2.4 | 38.2 ± 2.8 | 1.56 ± 0.05 | 0.0 ± 0.0 | 34.9 ± 1.8 |
| Aorta | Held-Out | Short Raster | 30.7 ± 3.5 | 30.0 ± 4.7 | 1.58 ± 0.07 | 0.0 ± 0.0 | - |
| Liver | Train | PPO+Memory | 42.1 ± 5.7 | 18.8 ± 2.9 | 1.29 ± 0.85 | 0.0 ± 0.0 | - |
| Liver | Train | PPO (No Memory) | 47.7 ± 7.9 | 23.0 ± 5.4 | 1.38 ± 0.65 | 0.0 ± 0.0 | - |
| Liver | Train | Footprint Raster | 41.0 ± 5.2 | 15.3 ± 2.1 | 1.61 ± 0.08 | 0.0 ± 0.0 | 51.0 ± 5.3 |
| Liver | Train | Short Raster | 26.6 ± 4.2 | 9.7 ± 1.8 | 1.57 ± 0.03 | 0.0 ± 0.0 | - |
| Liver | Held-Out | PPO+Memory | 54.2 ± 6.5 | 31.7 ± 3.8 | 1.59 ± 1.25 | 0.0 ± 0.0 | - |
| Liver | Held-Out | PPO (No Memory) | 66.0 ± 15.1 | 35.4 ± 10.0 | 0.79 ± 0.53 | 0.0 ± 0.0 | - |
| Liver | Held-Out | Footprint Raster | 65.3 ± 15.7 | 32.6 ± 10.2 | 1.62 ± 0.03 | 0.0 ± 0.0 | 57.6 ± 8.4 |
| Liver | Held-Out | Short Raster | 41.1 ± 7.2 | 19.5 ± 4.9 | 1.58 ± 0.02 | 0.0 ± 0.0 | - |