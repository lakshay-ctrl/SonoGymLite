# Aorta and Liver Reconstruction Re-Evaluation (with Total Rotation Logged)

Protocol: 10 random starts per patient (seeds: `42 + 17*i`), 300 steps.
Checkpoints evaluated: identical existing checkpoints from `ppo_checkpoints/block_b/`.

| Target | Method | Cohort | Upper Coverage (%) | Path Length (m) | Total Rotation (deg) | Existing Table Coverage (%) | Reproduces? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Aorta | PPO+Memory | Train | 12.4 ± 7.0 | 0.41 ± 0.27 | **137.6 ± 104.1** | 11.9% | Diff (12.4% vs 11.9%) |
| Aorta | PPO+Memory | Held-Out | 23.5 ± 3.1 | 0.12 ± 0.03 | **57.4 ± 14.6** | 23.7% | Diff (23.5% vs 23.7%) |
| Aorta | PPO (No Memory) | Train | 15.0 ± 9.1 | 0.42 ± 0.05 | **88.9 ± 50.8** | 14.8% | Diff (15.0% vs 14.8%) |
| Aorta | PPO (No Memory) | Held-Out | 33.6 ± 4.3 | 0.32 ± 0.08 | **68.2 ± 39.6** | 33.5% | YES |
| Liver | PPO+Memory | Train | 42.3 ± 5.6 | 1.01 ± 0.54 | **242.8 ± 127.8** | 42.1% | Diff (42.3% vs 42.1%) |
| Liver | PPO+Memory | Held-Out | 54.5 ± 5.7 | 1.47 ± 1.13 | **181.0 ± 32.2** | 54.2% | Diff (54.5% vs 54.2%) |
| Liver | PPO (No Memory) | Train | 48.7 ± 7.4 | 1.11 ± 0.34 | **1519.2 ± 1064.5** | 47.7% | Diff (48.7% vs 47.7%) |
| Liver | PPO (No Memory) | Held-Out | 67.4 ± 16.6 | 0.81 ± 0.61 | **1061.0 ± 726.8** | 66.0% | Diff (67.4% vs 66.0%) |