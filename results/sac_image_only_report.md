# Image-Only SAC Navigation Evaluation Report

Evaluation Protocol: 100 trials per cohort (Train: seeds 1000-1099, Held-Out: seeds 2000-2099, Horizon: 300 steps).
Success criterion: `pos_error < 5.0 mm` and `rot_error < 10.0 deg`.

| Model / Setting | Mode | Cohort | Pos Error (mm) | Rot Error (deg) | Succ (%) | First-Succ (%) | Steps | ep_rew_mean | Wall Time | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Spine MB Image-Only SAC** | Fixed | Train | 6.16 ± 3.73 | 2.45 ± 4.39° | 41.0% | 100.0% | 300.0 | 71.20 | 4.33h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_mb/sac_image_only_spine_mb_final.zip` |
| **Spine MB Image-Only SAC** | Fixed | Held-Out | 115.92 ± 94.45 | 31.13 ± 27.49° | 1.0% | 53.0% | 267.9 | 71.20 | 4.33h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_mb/sac_image_only_spine_mb_final.zip` |
| **Spine MB Image-Only SAC** | Early Term | Train | 3.04 ± 1.28 | 0.75 ± 1.42° | 100.0% | -- | 10.3 ± 3.7 | 71.20 | 4.33h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_mb/sac_image_only_spine_mb_final.zip` |
| **Spine MB Image-Only SAC** | Early Term | Held-Out | 65.15 ± 90.90 | 16.21 ± 23.29° | 53.0% | -- | 150.2 ± 119.2 | 71.20 | 4.33h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_mb/sac_image_only_spine_mb_final.zip` |
| **Spine LB Image-Only SAC** | Fixed | Train | 14.47 ± 28.17 | 2.92 ± 4.50° | 53.0% | 95.0% | 300.0 | 17.41 | 3.65h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_lb/sac_image_only_spine_lb_final.zip` |
| **Spine LB Image-Only SAC** | Fixed | Held-Out | 151.62 ± 88.93 | 32.40 ± 23.48° | 0.0% | 16.0% | 296.3 | 17.41 | 3.65h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_lb/sac_image_only_spine_lb_final.zip` |
| **Spine LB Image-Only SAC** | Early Term | Train | 8.62 ± 26.55 | 1.34 ± 2.59° | 95.0% | -- | 29.0 ± 63.2 | 17.41 | 3.65h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_lb/sac_image_only_spine_lb_final.zip` |
| **Spine LB Image-Only SAC** | Early Term | Held-Out | 134.72 ± 98.62 | 30.65 ± 24.38° | 16.0% | -- | 261.6 ± 87.3 | 17.41 | 3.65h | `/home/user/workspace/lakshya/ct_us/sac_image_only/spine_lb/sac_image_only_spine_lb_final.zip` |
| **Aorta MB Image-Only SAC** | Fixed | Train | 4.74 ± 2.47 | 5.53 ± 5.30° | 56.0% | 100.0% | 300.0 | 72.07 | 2.07h | `/home/user/workspace/lakshya/ct_us/sac_image_only/aorta_mb/sac_image_only_aorta_mb_final.zip` |
| **Aorta MB Image-Only SAC** | Fixed | Held-Out | 62.86 ± 47.54 | 12.58 ± 9.47° | 2.0% | 46.0% | 300.0 | 72.07 | 2.07h | `/home/user/workspace/lakshya/ct_us/sac_image_only/aorta_mb/sac_image_only_aorta_mb_final.zip` |
| **Aorta MB Image-Only SAC** | Early Term | Train | 3.14 ± 1.23 | 2.81 ± 2.36° | 100.0% | -- | 10.9 ± 3.7 | 72.07 | 2.07h | `/home/user/workspace/lakshya/ct_us/sac_image_only/aorta_mb/sac_image_only_aorta_mb_final.zip` |
| **Aorta MB Image-Only SAC** | Early Term | Held-Out | 36.32 ± 45.47 | 6.78 ± 7.36° | 46.0% | -- | 189.5 ± 125.2 | 72.07 | 2.07h | `/home/user/workspace/lakshya/ct_us/sac_image_only/aorta_mb/sac_image_only_aorta_mb_final.zip` |
| **Liver MB Image-Only SAC** | Fixed | Train | 24.41 ± 40.13 | 9.63 ± 17.02° | 34.0% | 100.0% | 298.9 | 71.17 | 2.31h | `/home/user/workspace/lakshya/ct_us/sac_image_only/liver_mb/sac_image_only_liver_mb_final.zip` |
| **Liver MB Image-Only SAC** | Fixed | Held-Out | 140.58 ± 96.57 | 17.44 ± 15.01° | 0.0% | 19.0% | 299.5 | 71.17 | 2.31h | `/home/user/workspace/lakshya/ct_us/sac_image_only/liver_mb/sac_image_only_liver_mb_final.zip` |
| **Liver MB Image-Only SAC** | Early Term | Train | 2.90 ± 1.26 | 1.43 ± 1.65° | 100.0% | -- | 8.2 ± 2.5 | 71.17 | 2.31h | `/home/user/workspace/lakshya/ct_us/sac_image_only/liver_mb/sac_image_only_liver_mb_final.zip` |
| **Liver MB Image-Only SAC** | Early Term | Held-Out | 113.58 ± 103.37 | 15.08 ± 15.24° | 19.0% | -- | 252.1 ± 101.2 | 71.17 | 2.31h | `/home/user/workspace/lakshya/ct_us/sac_image_only/liver_mb/sac_image_only_liver_mb_final.zip` |