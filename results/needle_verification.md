# Step 5: Geometric Needle Analysis Verification Report

## Overview & Methodology
This report documents the verification of the geometric needle insertion model (SonoSimLite Step 5)
across all 10 target sites (5 Aorta Bifurcation sites approached supine-anterior, and 5 Liver Portal Vein sites
approached prone-posterior). All evaluations run CPU-only without touching active GPU training processes.

---

## N1. Plain NumPy Stroke Recomputation
- **Formula:** $s = d / \sin\theta$ computed in plain NumPy across $N=10$ target sites.
- **Max Absolute Difference in Mean:** `0.0267 mm`.
- **Max Absolute Difference in Std:** `0.0648 mm`.

| Guide Angle $\theta$ | Paper Stroke (mm) | NumPy Recomputed Stroke (mm) | Diff Mean (mm) | Diff Std (mm) | Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **30°** | 297.5 ± 55.0 | 297.5 ± 54.9 | 0.0200 | 0.0648 | **PASS** |
| **45°** | 210.4 ± 38.9 | 210.4 ± 38.8 | 0.0216 | 0.0549 | **PASS** |
| **60°** | 171.8 ± 31.7 | 171.8 ± 31.7 | 0.0267 | 0.0169 | **PASS** |

---

## N2. Bone Collision Stability (0.50 mm vs 0.25 mm Step)
- **Bone Label File:** `bone_label.nii.gz` (TotalSegmentator merged 49 skeletal labels).
- **Label Values Used:** `label > 0` (binary bone mask).
- **Raycast Resolution:** Sampled at 0.50 mm steps and 0.25 mm steps from skin entry to target.

| Guide Angle $\theta$ | Paper Hit Rate (Fixed Lateral) | 0.50 mm Raycast Hits | 0.25 mm Raycast Hits | Transition Sites |
| :---: | :---: | :---: | :---: | :---: |
| **30°** | 1/10 (10%) | 1/10 (10%) | 2/10 (20%) | s0058 Portal Vein |
| **45°** | 2/10 (20%) | 2/10 (20%) | 2/10 (20%) | None (Stable) |
| **60°** | 1/10 (10%) | 1/10 (10%) | 1/10 (10%) | None (Stable) |

---

## N3. Target Depths & Cannula Reach Counts
- **Coordinates Chain:** NIfTI voxel (LPS) -> physical mm -> meters -> centering offset -> body rotation -> PyBullet world.
- **Voxel Spacing:** 1.50 x 1.50 x 1.50 mm (isotropic resampled TotalSegmentator volume).

| Patient | Target Site | Approach | Paper Depth (mm) | Recomputed Depth (mm) | Difference (mm) | Reach @ 60° (100/150/200 mm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `s0011` | Portal Vein | s0011 | 153.7 | 153.7 | +0.04 | 0/0/1 |
| `s0011` | Aorta Bifurcation | s0011 | 143.5 | 143.6 | +0.06 | 0/0/1 |
| `s0058` | Portal Vein | s0058 | 189.2 | 189.2 | -0.02 | 0/0/0 |
| `s0058` | Aorta Bifurcation | s0058 | 175.0 | 175.0 | -0.05 | 0/0/0 |
| `s0250` | Portal Vein | s0250 | 177.3 | 177.3 | +0.03 | 0/0/0 |
| `s0250` | Aorta Bifurcation | s0250 | 150.4 | 150.4 | +0.01 | 0/0/1 |
| `s0344` | Portal Vein | s0344 | 151.6 | 151.6 | -0.01 | 0/0/1 |
| `s0344` | Aorta Bifurcation | s0344 | 108.8 | 108.7 | -0.10 | 0/1/1 |
| `s0358` | Portal Vein | s0358 | 141.2 | 141.2 | +0.05 | 0/0/1 |
| `s0358` | Aorta Bifurcation | s0358 | 96.9 | 96.9 | -0.03 | 0/1/1 |

### Cohort Reach Counts Summary
| Guide Angle $\theta$ | Paper Reach (100 / 150 / 200 mm) | Recomputed Reach (100 / 150 / 200 mm) | Match Status |
| :---: | :---: | :---: | :---: |
| **30°** | 0/0/1 | 0/0/1 | **MATCH** |
| **45°** | 0/1/3 | 0/1/3 | **MATCH** |
| **60°** | 0/2/7 | 0/2/7 | **MATCH** |

---

## N4. Specular Visibility & B-Mode CNR
- **Formulas:**
  - Visibility: $V(\theta) = \max(0.1, \cos\theta)$, $V_{\text{dB}} = 20 \log_{10}(V)$.
  - Contrast-to-Noise Ratio: $\text{CNR} = \frac{|\mu_{\text{signal}} - \mu_{\text{background}}|}{\sigma_{\text{background}}}$.
- **Regions:**
  - Signal Region: needle shaft overlay mask ($I_{\text{needle}} > 0.08$).
  - Background Region: surrounding tissue pixels ($I_{\text{needle}} \le 0.08$).
  - Noise Estimate: standard deviation $\sigma$ of the background region pixels.
- **Speckle Background:** $\mathcal{N}(\mu=0.25, \sigma=0.05)$ clipped to $[0, 1]$ on a $256 \times 256$ grid ($N \approx 65,536$ pixels).
- **Evaluation Seeds:** 5 independent random seeds: `[40, 41, 42, 43, 44]`.

| Guide Angle $\theta$ | Specular $V$ | Paper $V$ (dB) | Paper CNR | Recomputed CNR (5 Seeds: Mean ± Std) | Per-Seed CNRs |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **30°** | 0.87 | -1.2 dB | 9.38 | **9.40 ± 0.03** | `[9.36, 9.39, 9.39, 9.41, 9.46]` |
| **45°** | 0.71 | -3.0 dB | 7.81 | **7.73 ± 0.02** | `[7.71, 7.75, 7.74, 7.72, 7.72]` |
| **60°** | 0.50 | -6.0 dB | 6.17 | **6.08 ± 0.03** | `[6.04, 6.12, 6.05, 6.08, 6.12]` |

- **Seed Dependence:** Standard deviation across seeds is $\le 0.03$ for all angles. The paper values do **not** depend on the random seed because the large number of background pixels ($N > 60,000$) ensures high sample stability.