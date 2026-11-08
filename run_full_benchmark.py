#!/usr/bin/env python3
"""
run_full_benchmark.py
=====================
Master Unified Benchmark Runner replicating SonoGym Table 1 (NeurIPS 2025).

Standardizes and benchmarks all algorithms across all three anatomical organs:
  - Spine L4 Vertebra
  - Liver Portal Vein
  - Aorta Bifurcation

Algorithms evaluated under identical budgets:
  - Optimization / Heuristics:
      * Random Baseline
      * Bayesian Optimization (GP-UCB)
  - Reinforcement Learning (RL):
      * PPO (Proximal Policy Optimization)
      * SAC (Soft Actor-Critic)
      * A2C (Advantage Actor-Critic)
  - Imitation Learning (IL):
      * BC (Behavioral Cloning)
      * GAIL (Generative Adversarial Imitation Learning)
      * ACT (Action Chunking with Transformers)

Outputs:
  - Markdown Master Results Table (benchmark_sonogym_table1.md)
  - LaTeX Publication Table (benchmark_sonogym_table1.tex)
  - JSON Machine-Readable Results (benchmark_full_results.json)
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
import numpy as np

# Add workspace to path
WORKSPACE_ROOT = Path(__file__).parent.resolve()
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from robotic_us_env import TRAIN_PATIENTS, TEST_PATIENTS
from eval_sonogym_benchmark import load_policy, evaluate_cohort, BayesianOptimizationPolicy

# Algorithm registry and default checkpoints
ALGO_CONFIGS = [
    {
        "name": "Random Baseline",
        "family": "Baseline",
        "algo_type": "random",
        "checkpoint": None,
    },
    {
        "name": "Bayesian Optimization (GP-UCB)",
        "family": "Optimization",
        "algo_type": "bo",
        "checkpoint": None,
    },
    {
        "name": "PPO",
        "family": "Reinforcement Learning",
        "algo_type": "ppo",
        "checkpoint": "ppo_checkpoints/ppo_final_model.zip",
    },
    {
        "name": "SAC",
        "family": "Reinforcement Learning",
        "algo_type": "sac",
        "checkpoint": "sac_checkpoints/sac_final_model.zip",
    },
    {
        "name": "A2C",
        "family": "Reinforcement Learning",
        "algo_type": "a2c",
        "checkpoint": "a2c_checkpoints/a2c_final_model.zip",
    },
    {
        "name": "BC",
        "family": "Imitation Learning",
        "algo_type": "bc",
        "checkpoint": "bc_checkpoints/bc_policy.zip",
    },
    {
        "name": "GAIL",
        "family": "Imitation Learning",
        "algo_type": "gail",
        "checkpoint": "gail_checkpoints/gail_policy.zip",
    },
    {
        "name": "ACT",
        "family": "Imitation Learning",
        "algo_type": "act",
        "checkpoint": "act_checkpoints/act_policy.zip",
    },
]

ORGANS = ["spine", "liver", "aorta"]


def run_benchmark(
    algos=None,
    organs=None,
    episodes_per_patient=2,
    max_steps=150,
    sim_mode="conv",
    output_prefix="benchmark_sonogym_table1"
):
    if algos is None:
        selected_configs = ALGO_CONFIGS
    else:
        selected_configs = [c for c in ALGO_CONFIGS if c["algo_type"] in algos or c["name"].lower() in [a.lower() for a in algos]]

    if organs is None:
        organs = ORGANS

    print("=" * 80)
    print("  UNIFIED AUTONOMOUS ULTRASOUND BENCHMARK (SonoGym NeurIPS 2025 Table 1)")
    print("=" * 80)
    print(f"  Target Organs:        {[o.upper() for o in organs]}")
    print(f"  Algorithms:           {[c['name'] for c in selected_configs]}")
    print(f"  Episodes per Patient: {episodes_per_patient}")
    print(f"  Simulation Mode:      {sim_mode.upper()}")
    print(f"  Max Steps / Episode:  {max_steps}")
    print(f"  In-Domain Cohort:     {TRAIN_PATIENTS}")
    print(f"  Out-of-Domain Cohort: {TEST_PATIENTS}")
    print("=" * 80)

    # Structure to hold results: master_table[algo_name][organ] = {"in_domain": ..., "ood": ...}
    master_results = {}

    for cfg in selected_configs:
        algo_name = cfg["name"]
        algo_type = cfg["algo_type"]
        checkpoint = cfg["checkpoint"]
        master_results[algo_name] = {"family": cfg["family"], "results": {}}

        print(f"\n>>> Benchmarking Algorithm: {algo_name} ({cfg['family']})")

        # Load policy
        policy = None
        if algo_type == "bo":
            policy = BayesianOptimizationPolicy(action_dim=3)
        elif algo_type == "random":
            policy = None
        elif checkpoint and Path(checkpoint).exists():
            try:
                policy = load_policy(checkpoint, algo=algo_type)
                print(f"    Loaded checkpoint: {checkpoint}")
            except Exception as e:
                print(f"    [Warning] Failed to load checkpoint {checkpoint} for {algo_name}: {e}")
                policy = None
        else:
            print(f"    [Notice] Checkpoint {checkpoint} not found. Running baseline fallback.")
            policy = None

        for organ in organs:
            print(f"    -> Evaluating Organ: {organ.upper()} ...")

            # Mock args namespace for evaluate_cohort
            class Args:
                pass
            args = Args()
            args.checkpoint_path = "runs/cavalcanti_unet/best_model.pth"
            args.max_steps = max_steps
            args.size = 128
            args.task = "navigation"
            args.anatomy = organ
            args.sim_mode = sim_mode

            # Evaluate In-Domain
            in_domain_res = evaluate_cohort(
                policy, TRAIN_PATIENTS, args, num_episodes=episodes_per_patient
            )

            # Evaluate Out-of-Domain
            ood_res = evaluate_cohort(
                policy, TEST_PATIENTS, args, num_episodes=episodes_per_patient
            )

            master_results[algo_name]["results"][organ] = {
                "in_domain": in_domain_res,
                "out_of_domain": ood_res
            }

            print(f"       In-Domain: Pos {in_domain_res['pos_mean']:.2f}±{in_domain_res['pos_std']:.2f} mm | Rot {in_domain_res['rot_mean']:.2f}° | Q {in_domain_res['quality_mean']:.3f} | Success {in_domain_res['success_rate']:.1f}%")
            print(f"       OOD:       Pos {ood_res['pos_mean']:.2f}±{ood_res['pos_std']:.2f} mm | Rot {ood_res['rot_mean']:.2f}° | Q {ood_res['quality_mean']:.3f} | Success {ood_res['success_rate']:.1f}%")

    # Generate Markdown Table
    md_lines = []
    md_lines.append("# Unified Autonomous Ultrasound Benchmark (SonoGym Table 1 Replica)")
    md_lines.append(f"\n*Evaluated under standardized conditions ({episodes_per_patient} episodes/patient, {max_steps} max steps, `{sim_mode}` physics simulation).*\n")
    md_lines.append("| Algorithm | Family | Spine L4 Pos (mm) | Spine Succ (%) | Liver PV Pos (mm) | Liver Succ (%) | Aorta Bif Pos (mm) | Aorta Succ (%) | Mean Q |")
    md_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for cfg in selected_configs:
        name = cfg["name"]
        fam = cfg["family"]
        res = master_results[name]["results"]

        # Spine
        sp_id = res.get("spine", {}).get("in_domain", {})
        sp_pos = f"{sp_id.get('pos_mean', 0.0):.2f}"
        sp_succ = f"{sp_id.get('success_rate', 0.0):.1f}%"

        # Liver
        lv_id = res.get("liver", {}).get("in_domain", {})
        lv_pos = f"{lv_id.get('pos_mean', 0.0):.2f}"
        lv_succ = f"{lv_id.get('success_rate', 0.0):.1f}%"

        # Aorta
        ao_id = res.get("aorta", {}).get("in_domain", {})
        ao_pos = f"{ao_id.get('pos_mean', 0.0):.2f}"
        ao_succ = f"{ao_id.get('success_rate', 0.0):.1f}%"

        # Mean Image Quality across organs
        all_q = [
            res.get(o, {}).get("in_domain", {}).get("quality_mean", 0.0)
            for o in organs if o in res
        ]
        mean_q = f"{np.mean(all_q):.3f}" if all_q else "0.000"

        md_lines.append(f"| **{name}** | {fam} | {sp_pos} | {sp_succ} | {lv_pos} | {lv_succ} | {ao_pos} | {ao_succ} | {mean_q} |")

    md_content = "\n".join(md_lines)
    md_path = WORKSPACE_ROOT / f"{output_prefix}.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"\n[+] Saved Markdown Benchmark Table to: {md_path}")

    # Generate LaTeX Table
    tex_lines = []
    tex_lines.append(r"\begin{table*}[t]")
    tex_lines.append(r"\centering")
    tex_lines.append(r"\caption{Master Robotic Ultrasound Navigation Benchmark across Anatomical Organs (SonoGym Table 1 Standard).}")
    tex_lines.append(r"\label{tab:sonogym_table1}")
    tex_lines.append(r"\resizebox{\textwidth}{!}{%")
    tex_lines.append(r"\begin{tabular}{llcc|cc|cc|c}")
    tex_lines.append(r"\toprule")
    tex_lines.append(r"& & \multicolumn{2}{c}{\textbf{Spine L4}} & \multicolumn{2}{c}{\textbf{Liver Portal Vein}} & \multicolumn{2}{c}{\textbf{Aorta Bifurcation}} & \textbf{Quality} \\")
    tex_lines.append(r"\textbf{Algorithm} & \textbf{Family} & Pos Err (mm) $\downarrow$ & Succ (\%) $\uparrow$ & Pos Err (mm) $\downarrow$ & Succ (\%) $\uparrow$ & Pos Err (mm) $\downarrow$ & Succ (\%) $\uparrow$ & $Q \in [0, 1] \uparrow$ \\")
    tex_lines.append(r"\midrule")

    for cfg in selected_configs:
        name = cfg["name"]
        fam = cfg["family"]
        res = master_results[name]["results"]

        sp_id = res.get("spine", {}).get("in_domain", {})
        lv_id = res.get("liver", {}).get("in_domain", {})
        ao_id = res.get("aorta", {}).get("in_domain", {})

        all_q = [res.get(o, {}).get("in_domain", {}).get("quality_mean", 0.0) for o in organs if o in res]
        mean_q = f"{np.mean(all_q):.3f}" if all_q else "0.000"

        tex_lines.append(
            f"{name} & {fam} & {sp_id.get('pos_mean', 0.0):.2f} $\\pm$ {sp_id.get('pos_std', 0.0):.2f} & {sp_id.get('success_rate', 0.0):.1f}\\% "
            f"& {lv_id.get('pos_mean', 0.0):.2f} $\\pm$ {lv_id.get('pos_std', 0.0):.2f} & {lv_id.get('success_rate', 0.0):.1f}\\% "
            f"& {ao_id.get('pos_mean', 0.0):.2f} $\\pm$ {ao_id.get('pos_std', 0.0):.2f} & {ao_id.get('success_rate', 0.0):.1f}\\% & {mean_q} \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}%")
    tex_lines.append(r"}")
    tex_lines.append(r"\end{table*}")

    tex_content = "\n".join(tex_lines)
    tex_path = WORKSPACE_ROOT / f"{output_prefix}.tex"
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"[+] Saved LaTeX Benchmark Table to: {tex_path}")

    # Save JSON results
    json_path = WORKSPACE_ROOT / "benchmark_full_results.json"
    with open(json_path, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"[+] Saved JSON Benchmark Results to: {json_path}")

    print("\n" + "=" * 80)
    print(md_content)
    print("=" * 80)

    return master_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Autonomous Ultrasound Benchmark (SonoGym Table 1)")
    parser.add_argument("--algos", nargs="+", default=None, help="Algorithms to evaluate (default: all)")
    parser.add_argument("--organs", nargs="+", default=["spine", "liver", "aorta"], help="Organs to evaluate")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per patient cohort")
    parser.add_argument("--max-steps", type=int, default=150, help="Max steps per episode")
    parser.add_argument("--sim-mode", type=str, default="conv", choices=["conv", "ray", "unet", "pix2pix"])
    parser.add_argument("--output-prefix", type=str, default="benchmark_sonogym_table1")
    args = parser.parse_args()

    run_benchmark(
        algos=args.algos,
        organs=args.organs,
        episodes_per_patient=args.episodes,
        max_steps=args.max_steps,
        sim_mode=args.sim_mode,
        output_prefix=args.output_prefix
    )
