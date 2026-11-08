# Image-Only PPO Fixed-Horizon Evaluation Report

Evaluation Protocol: Fixed-Horizon mode (300 steps with NO early termination on goal success).
Out-of-bounds safety termination: probe leaves skin boundary (identical to spine evaluation).
100 trials per cohort: Train seeds 1000-1099, Unseen/Held-Out seeds 2000-2099.
Success criterion at final step: `pos_error < 5.0 mm` and `rot_error < 10.0 deg`.

| Target | Model | Cohort | Succ (%) | Pos Error (mm) | Rot Error (deg) | Steps | Checkpoint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Aorta | Aorta Image-Only PPO Seed 0 | Train | 25.0% | 12.02 ± 28.25 | 6.81 ± 7.42° | 300.0 ± 0.0 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip` |
| Aorta | Aorta Image-Only PPO Seed 0 | Held-Out | 0.0% | 148.62 ± 48.47 | 38.54 ± 29.67° | 222.6 ± 94.9 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip` |
| Aorta | Aorta Image-Only PPO Seed 1 | Train | 53.0% | 8.17 ± 21.90 | 5.62 ± 5.25° | 297.9 ± 20.8 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip` |
| Aorta | Aorta Image-Only PPO Seed 1 | Held-Out | 0.0% | 176.85 ± 59.34 | 33.99 ± 20.84° | 232.8 ± 96.4 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip` |
| Aorta | Aorta Image-Only PPO Seed 2 | Train | 39.0% | 17.16 ± 35.71 | 11.31 ± 12.44° | 298.8 ± 12.4 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip` |
| Aorta | Aorta Image-Only PPO Seed 2 | Held-Out | 0.0% | 155.85 ± 61.92 | 23.22 ± 16.96° | 267.4 ± 66.3 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip` |
| Liver | Liver Image-Only PPO (300k) | Train | 2.0% | 22.02 ± 16.12 | 5.72 ± 5.73° | 300.0 ± 0.0 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/liver_centre/ppo_scale_liver_300000_steps.zip` |
| Liver | Liver Image-Only PPO (300k) | Held-Out | 0.0% | 46.62 ± 32.27 | 19.47 ± 19.05° | 300.0 ± 0.0 | `/home/user/workspace/lakshya/ct_us/ppo_checkpoints/liver_centre/ppo_scale_liver_300000_steps.zip` |

### Summary: Aorta (Mean ± Std across 3 Seeds)

| Target | Summary Metric | Cohort | Succ (%) | Pos Error (mm) | Rot Error (deg) | Steps |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Aorta | **Mean ± Std (3 Seeds)** | Train | **39.0 ± 11.4%** | **12.45 ± 3.68** | **7.91 ± 2.45°** | **298.9 ± 0.9** |
| Aorta | **Mean ± Std (3 Seeds)** | Held-Out | **0.0 ± 0.0%** | **160.44 ± 11.97** | **31.92 ± 6.42°** | **241.0 ± 19.2** |