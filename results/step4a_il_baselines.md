# Step 4a: Spine Imitation Learning & Optimization Baselines

| Method | Cohort | Pos Error (mm) | Rot Error (deg) | Success Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Behavioral Cloning (BC)** | Train | 130.6 | 24.7° | 0.0% |
| **Behavioral Cloning (BC)** | Held-Out | 131.2 | 24.2° | 0.0% |
| **GP-UCB Optimization** | Train | 171.2 | 33.7° | 0.0% |
| **GP-UCB Optimization** | Held-Out | 120.7 | 37.8° | 0.0% |

> **Key Finding**: Consistent with SonoGym (NeurIPS 2025 Table 1), passive imitation learning (BC) and derivative-free Gaussian Process optimization (GP-UCB) fail to solve high-dimensional 6-DoF robotic acoustic probe contact navigation on non-planar human torso geometry without closed-loop reinforcement learning.