# Step 5b: Navigation-to-Needle Hit Test Benchmark

## Reproduction Check & Policy Verification
- **Verified Reproduction of Step 3 Table:** Using ground-truth `robotic_us_env.py`, all navigation metrics reproduce Step 3 exact table targets (Aorta Goal-PPO: Train pos error $1.18 \pm 0.40\,\text{mm}$, rot error $1.50 \pm 1.44^\circ$, $100\%$ success; Held-Out: pos error $1.22 \pm 0.78\,\text{mm}$, rot error $0.59 \pm 0.30^\circ$, $100\%$ success).
- **Main Needle Protocol:** Evaluated at the **First-Success Pose** (the pose where clinical navigation success criteria $\le 5\,\text{mm}$ and $\le 10^\circ$ are first satisfied).
- **Needle Hit Definition:** Needle tip $\le 5.0\,\text{mm}$ of target center AND straight path does not cross bone (`bone_label.nii.gz` at 0.5 mm steps) AND required stroke $s \le L$.
- **Feasible Hit Rate:** Evaluated strictly among feasible cases where the reference perfect pose achieves HIT.

---

## Reference Ceiling: Perfect Pose Hit/Miss per Patient

| Target | Patient | $\theta$ | Required Stroke (mm) | 100 mm Needle | 150 mm Needle | 200 mm Needle |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Aorta Bifurcation | `s0011` | 30° | 287.1 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0011` | 45° | 203.0 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0011` | 60° | 165.8 mm | MISS | MISS | HIT |
| Aorta Bifurcation | `s0058` | 30° | 349.9 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0058` | 45° | 247.4 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0058` | 60° | 202.0 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0250` | 30° | 300.7 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0250` | 45° | 212.7 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0250` | 60° | 173.6 mm | MISS | MISS | HIT |
| Aorta Bifurcation | `s0344` | 30° | 217.6 mm | MISS | MISS | MISS |
| Aorta Bifurcation | `s0344` | 45° | 153.8 mm | MISS | MISS | HIT |
| Aorta Bifurcation | `s0344` | 60° | 125.6 mm | MISS | HIT | HIT |
| Aorta Bifurcation | `s0358` | 30° | 193.7 mm | MISS | MISS | HIT |
| Aorta Bifurcation | `s0358` | 45° | 137.0 mm | MISS | HIT | HIT |
| Aorta Bifurcation | `s0358` | 60° | 111.9 mm | MISS | HIT | HIT |
| Liver Portal Vein | `s0011` | 30° | 307.5 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0011` | 45° | 217.4 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0011` | 60° | 177.5 mm | MISS | MISS | HIT |
| Liver Portal Vein | `s0058` | 30° | 378.3 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0058` | 45° | 267.5 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0058` | 60° | 218.4 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0250` | 30° | 354.7 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0250` | 45° | 250.8 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0250` | 60° | 204.8 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0344` | 30° | 303.2 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0344` | 45° | 214.4 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0344` | 60° | 175.0 mm | MISS | MISS | HIT |
| Liver Portal Vein | `s0358` | 30° | 282.2 mm | MISS | MISS | MISS |
| Liver Portal Vein | `s0358` | 45° | 199.6 mm | MISS | MISS | HIT |
| Liver Portal Vein | `s0358` | 60° | 163.0 mm | MISS | MISS | MISS |

---

## Needle Hit Benchmark Results Table (Policy Type Breakdown)

| Target | Cohort | Policy Type | Checkpoint File | Trained Steps | $\theta$ | Needle Length | Perfect Hits | Feasible Cases ($N_{\text{feas}}$) | Feasible Hit Rate (%) [95% CI] | Median Tip Dist (mm) | Median Pos Err (mm) | Median Tilt Err (deg) |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 30° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 200.9 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 30° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 150.9 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 30° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 101.4 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 45° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 113.1 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 45° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 63.7 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 45° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 17.0 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 60° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 74.4 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 60° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 25.8 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 60° | 200 mm | 2/3 | 67 | 17.9% [10.6, 28.7] (12/67) | 8.5 mm | 3.0 mm | 2.6° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 30° | 100 mm | 0/2 | 0 | N/A (0 feasible) | 105.7 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 30° | 150 mm | 0/2 | 0 | N/A (0 feasible) | 55.8 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 30° | 200 mm | 1/2 | 50 | 76.0% [62.6, 85.7] (38/50) | 13.8 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 45° | 100 mm | 0/2 | 0 | N/A (0 feasible) | 45.7 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 45° | 150 mm | 1/2 | 50 | 76.0% [62.6, 85.7] (38/50) | 4.4 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 45° | 200 mm | 2/2 | 100 | 69.0% [59.4, 77.2] (69/100) | 3.8 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 60° | 100 mm | 0/2 | 0 | N/A (0 feasible) | 20.1 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 60° | 150 mm | 2/2 | 100 | 69.0% [59.4, 77.2] (69/100) | 3.5 mm | 2.6 mm | 1.1° |
| Aorta Bifurcation | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_aorta_final_model.zip` | 100,352 | 60° | 200 mm | 2/2 | 100 | 69.0% [59.4, 77.2] (69/100) | 3.5 mm | 2.6 mm | 1.1° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 30° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 255.3 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 30° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 205.3 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 30° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 155.3 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 45° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 151.3 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 45° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 101.3 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 45° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 51.3 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 60° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 105.1 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 60° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 55.2 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | In-Domain Train | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 60° | 200 mm | 1/3 | 34 | 58.8% [42.2, 73.6] (20/34) | 7.1 mm | 3.0 mm | 0.9° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 30° | 100 mm | 0/2 | 0 | N/A (0 feasible) | 192.7 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 30° | 150 mm | 0/2 | 0 | N/A (0 feasible) | 143.9 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 30° | 200 mm | 0/2 | 0 | N/A (0 feasible) | 96.3 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 45° | 100 mm | 0/2 | 0 | N/A (0 feasible) | 107.6 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 45° | 150 mm | 0/2 | 0 | N/A (0 feasible) | 60.1 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 45° | 200 mm | 1/2 | 50 | 8.0% [3.2, 18.8] (4/50) | 16.2 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 60° | 100 mm | 0/2 | 0 | N/A (0 feasible) | 70.3 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 60° | 150 mm | 0/2 | 0 | N/A (0 feasible) | 24.2 mm | 2.8 mm | 0.7° |
| Liver Portal Vein | Held-Out Test | Goal-Conditioned PPO | `ppo_mb_liver_final_model.zip` | 100,352 | 60° | 200 mm | 1/2 | 50 | 100.0% [92.9, 100.0] (50/50) | 4.7 mm | 2.8 mm | 0.7° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 30° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 201.1 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 30° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 151.1 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 30° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 101.4 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 45° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 112.9 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 45° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 63.6 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 45° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 19.9 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 60° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 74.5 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 60° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 27.0 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 0) | `ppo_scale_aorta_s0_final.zip` | 300,032 | 60° | 200 mm | 2/3 | 67 | 13.4% [7.2, 23.6] (9/67) | 7.9 mm | 3.2 mm | 2.8° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 30° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 200.8 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 30° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 150.9 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 30° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 101.5 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 45° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 113.0 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 45° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 63.7 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 45° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 19.2 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 60° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 74.8 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 60° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 26.1 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 1) | `ppo_scale_aorta_s1_final.zip` | 300,032 | 60° | 200 mm | 2/3 | 67 | 19.4% [11.7, 30.4] (13/67) | 7.9 mm | 3.1 mm | 2.3° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 30° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 200.6 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 30° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 150.7 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 30° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 100.9 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 45° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 112.8 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 45° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 63.1 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 45° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 18.4 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 60° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 74.1 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 60° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 24.9 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | Image-Only PPO (Seed 2) | `ppo_scale_aorta_s2_final.zip` | 300,032 | 60° | 200 mm | 2/3 | 67 | 20.9% [12.9, 32.1] (14/67) | 7.2 mm | 2.7 mm | 2.1° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 30° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 200.8 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 30° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 150.9 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 30° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 101.2 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 45° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 112.9 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 45° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 63.5 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 45° | 200 mm | 0/3 | 0 | N/A (0 feasible) | 19.2 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 60° | 100 mm | 0/3 | 0 | N/A (0 feasible) | 74.4 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 60° | 150 mm | 0/3 | 0 | N/A (0 feasible) | 26.0 mm | 3.0 mm | 2.4° |
| Aorta Bifurcation | In-Domain Train | **Image-Only PPO (Mean ± Std)** | `ppo_scale_aorta_s{0,1,2}_final.zip` | 300,032 | 60° | 200 mm | 2/3 | 67 | **17.9 ± 3.2%** | 7.7 mm | 3.0 mm | 2.4° |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 30° | 100 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 30° | 150 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 30° | 200 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 45° | 100 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 45° | 150 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 45° | 200 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 60° | 100 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 60° | 150 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | In-Domain Train | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 60° | 200 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 30° | 100 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 30° | 150 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 30° | 200 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 45° | 100 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 45° | 150 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 45° | 200 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 60° | 100 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 60° | 150 mm | - | - | *Not Evaluated* | - | - | - |
| Liver Portal Vein | Held-Out Test | *Image-Only PPO* | `None (4c trains liver-centre)` | Not Evaluated | 60° | 200 mm | - | - | *Not Evaluated* | - | - | - |