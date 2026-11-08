# Image-Only Navigation Policies: Evaluation at Terminating Step (Early Termination ON)

This report presents the position error, rotation error, episode steps, and success rate evaluated **at the terminating step** (the step where the episode terminates due to meeting the goal criteria `pos < 5 mm` and `rot < 10°` with $Q \ge 0.40$, or at the horizon step $T=300$ if unreached).

### Protocol & Cohorts
- **Task**: 6-DoF probe surface navigation to ultrasound landmark plane (SonoGym physics, `conv` MB simulation, 1.0 mm pixel spacing, image size $128 \times 128$ for image-only policies).
- **Early Termination**: ON (`early_termination=True`). Episode stops immediately upon success; otherwise runs to maximum horizon ($T = 300$).
- **Cohorts**:
  - **Training**: 100 trials (`s0011`: 34, `s0058`: 33, `s0250`: 33; seeds 1000–1099).
  - **Held-Out**: 100 trials (`s0344`: 50, `s0358`: 50; seeds 2000–2099).

---

## Benchmark Results Table

| Target | Policy / Method | Checkpoint & Step Count | Early Term | Cohort | Pos Error (mm) | Rot Error (deg) | Steps to Term. | Success Rate (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Spine** | **PPO (Image-Only)** | `ppo_checkpoints/image_only_scale/ppo_scale_spine_s0_final.zip`<br>*(Exact steps: 1,000,448)* | **ON** | Train | 2.92 ± 1.18 | 0.46 ± 1.00 | 9.7 ± 4.0 | 100.0% |
| **Spine** | **PPO (Image-Only)** | *(Same checkpoint as reported row)* | **ON** | Held-Out | 143.57 ± 156.38 | 21.45 ± 28.89 | 142.2 ± 104.8 | 43.0% |
| **Aorta** | **PPO (Image-Only, 3-Seed Mean)** | Mean ± std across Seeds 0, 1, 2<br>*(300,032 steps each)* | **ON** | Train | 4.32 ± 2.23 | 3.19 ± 0.78 | 37.9 ± 15.3 | 96.7 ± 3.5% |
| **Aorta** | **PPO (Image-Only, 3-Seed Mean)** | Mean ± std across Seeds 0, 1, 2<br>*(300,032 steps each)* | **ON** | Held-Out | 152.20 ± 21.94 | 29.78 ± 2.82 | 223.6 ± 16.3 | 6.0 ± 4.6% |
| **Spine** | **Behavioral Cloning (BC)** | `il_new/checkpoints/spine/bc_spine_image_only.pt`<br>*(Trained on 200 expert demos)* | **ON** | Train | 85.70 ± 34.09 | 11.74 ± 7.65 | 295.4 ± 33.5 | 1.0% |
| **Spine** | **Behavioral Cloning (BC)** | `il_new/checkpoints/spine/bc_spine_image_only.pt` | **ON** | Held-Out | 113.96 ± 104.84 | 24.22 ± 21.81 | 291.1 ± 42.7 | 1.0% |
| **Spine** | **ACT (Chunking Transformer)** | `il_new/checkpoints/spine/act_spine_image_only.pt`<br>*(10-step temporal ensemble chunking)* | **ON** | Train | 73.54 ± 39.12 | 10.57 ± 7.74 | 251.7 ± 107.0 | 16.0% |
| **Spine** | **ACT (Chunking Transformer)** | `il_new/checkpoints/spine/act_spine_image_only.pt` | **ON** | Held-Out | 70.25 ± 17.85 | 21.08 ± 15.24 | 294.1 ± 41.2 | 2.0% |
| **Spine** | **Template BO** | GP-UCB on $128 \times 128$ template image<br>*(No access to pose error feedback)* | **N/A\*** | Train | 66.60 ± 29.94 | 11.16 ± 8.45 | 285.0 ± 35.8 | 0.0% |
| **Spine** | **Template BO** | GP-UCB on $128 \times 128$ template image | **N/A\*** | Held-Out | 56.78 ± 31.13 | 22.48 ± 17.84 | 288.3 ± 31.6 | 1.0% |
| **Spine** | **Quality BO** | GP-UCB on no-reference quality score $Q$<br>*(No access to pose error feedback)* | **N/A\*** | Train | 74.40 ± 30.93 | 11.50 ± 7.18 | 273.3 ± 44.2 | 0.0% |
| **Spine** | **Quality BO** | GP-UCB on no-reference quality score $Q$ | **N/A\*** | Held-Out | 55.57 ± 29.35 | 21.42 ± 19.65 | 272.9 ± 46.1 | 0.0% |

\* **Note on Bayesian Optimization (BO)**: Early termination based on ground-truth pose error does **not** apply to BO. In black-box optimization (`run_bo_nav.py`), the agent observes only image similarity / image quality without oracle pose feedback. The episode runs for its full allotted budget (up to 25 GP iterations / 300 environment steps), with termination occurring only if internal similarity $\ge 0.90$ across consecutive samples (0.0% occurrence).

---

### Aorta PPO (Image-Only): Individual Seed Breakdown

All three aorta models were trained for 300,032 steps and evaluated with early termination ON on the exact identical cohorts:

| Seed | Checkpoint Path | Cohort | Pos Error (mm) | Rot Error (deg) | Steps to Term. | Success Rate (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Seed 0** | `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip` | Train | 6.90 ± 23.41 | 3.72 ± 6.29 | 39.5 ± 56.3 | 97.0% |
| **Seed 0** | `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s0_final.zip` | Held-Out | 151.17 ± 55.41 | 30.48 ± 25.62 | 212.8 ± 102.3 | 7.0% |
| **Seed 1** | `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip` | Train | 3.01 ± 1.13 | 2.29 ± 1.85 | 21.9 ± 29.2 | 100.0% |
| **Seed 1** | `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s1_final.zip` | Held-Out | 174.64 ± 57.40 | 32.18 ± 21.52 | 215.7 ± 99.0 | 1.0% |
| **Seed 2** | `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip` | Train | 3.05 ± 1.48 | 3.56 ± 4.20 | 52.4 ± 87.9 | 93.0% |
| **Seed 2** | `ppo_checkpoints/image_only_scale/ppo_scale_aorta_s2_final.zip` | Held-Out | 130.79 ± 78.23 | 26.68 ± 22.20 | 242.4 ± 93.6 | 10.0% |
| **All Seeds** | **Mean ± Std Over Seeds** | **Train** | **4.32 ± 2.23** | **3.19 ± 0.78** | **37.9 ± 15.3** | **96.7 ± 3.5%** |
| **All Seeds** | **Mean ± Std Over Seeds** | **Held-Out** | **152.20 ± 21.94** | **29.78 ± 2.82** | **223.6 ± 16.3** | **6.0 ± 4.6%** |

---

### Key Observations
1. **Spine PPO (Image-Only)**: On training patients, reaches the goal within $9.7 \pm 4.0$ steps with $100\%$ early success ($2.92 \pm 1.18$ mm, $0.46 \pm 1.00^\circ$). Under fixed-horizon evaluation without early termination, lack of hovering caused it to wander away ($135.5 \pm 137.0$ mm), which explains why the early-termination metric is drastically different from the fixed-horizon metric.
2. **Aorta PPO (Image-Only)**: Consistently reaches the goal across all 3 seeds on training patients ($96.7 \pm 3.5\%$ success, terminating in $37.9 \pm 15.3$ steps with $4.32 \pm 2.23$ mm error), but shows steep out-of-distribution degradation on held-out patients ($6.0 \pm 4.6\%$ success, terminating at $223.6 \pm 16.3$ steps with $152.20 \pm 21.94$ mm error).
3. **Imitation Learning (BC & ACT)**: Without goal vector conditioning, passive behavioral cloning on raw image observations fails to generalize to 6-DoF navigation ($1.0\%$ success). ACT achieves early success in $16.0\%$ of training episodes ($73.54 \pm 39.12$ mm at termination, $251.7 \pm 107.0$ steps), but drops to $2.0\%$ on held-out patients.
4. **Bayesian Optimization**: Because BO operates strictly via black-box acquisition on image features without pose error feedback, early stopping on ground-truth pose error is not applicable.
