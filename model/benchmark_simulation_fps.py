import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from robotic_us_env_v2 import RoboticUltrasoundGymEnvV2


def sync_cuda():
    """Synchronize CUDA so GPU work is included in timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def create_env(patient, skip_unet):
    return RoboticUltrasoundGymEnvV2(
        subject_dir=f"totalseg_patients/{patient}",
        render_mode="rgb_array",
        size=256,
        skip_unet=skip_unet,
    )


def benchmark_env(env, warmup_steps=200, timed_steps=1000):
    """Benchmark complete env.step() execution."""

    obs, info = env.reset()

    zero_action = np.zeros(6, dtype=np.float32)

    # Warm-up
    for _ in range(warmup_steps):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if terminated or truncated:
            obs, info = env.reset()

    # Synchronize before timing
    sync_cuda()

    start = time.perf_counter()

    completed = 0

    while completed < timed_steps:
        obs, reward, terminated, truncated, info = env.step(zero_action)
        completed += 1

        if terminated or truncated:
            obs, info = env.reset()

    sync_cuda()

    elapsed = time.perf_counter() - start

    fps = completed / elapsed
    ms_per_step = 1000.0 * elapsed / completed

    return fps, ms_per_step


def run_backend(name, patient, skip_unet, repeats, warmup, steps):

    print("\n" + "=" * 80)
    print(f"BACKEND: {name}")
    print("=" * 80)

    fps_values = []
    ms_values = []

    env = create_env(
        patient=patient,
        skip_unet=skip_unet,
    )

    try:
        for i in range(repeats):

            print(f"Run {i + 1}/{repeats}...", flush=True)

            fps, ms = benchmark_env(
                env,
                warmup_steps=warmup,
                timed_steps=steps,
            )

            fps_values.append(fps)
            ms_values.append(ms)

            print(
                f"  FPS = {fps:.3f} | "
                f"time/step = {ms:.3f} ms"
            )

    finally:
        env.close()

    fps_values = np.asarray(fps_values)
    ms_values = np.asarray(ms_values)

    result = {
        "backend": name,
        "patient": patient,
        "skip_unet": skip_unet,
        "repeats": repeats,
        "warmup_steps": warmup,
        "timed_steps": steps,
        "fps_mean": float(np.mean(fps_values)),
        "fps_std": float(
            np.std(fps_values, ddof=1)
        ) if repeats > 1 else 0.0,
        "fps_median": float(np.median(fps_values)),
        "fps_min": float(np.min(fps_values)),
        "fps_max": float(np.max(fps_values)),
        "ms_mean": float(np.mean(ms_values)),
        "ms_std": float(
            np.std(ms_values, ddof=1)
        ) if repeats > 1 else 0.0,
        "raw_fps": fps_values.tolist(),
        "raw_ms_per_step": ms_values.tolist(),
    }

    print("\nResult:")
    print(
        f"{name}: "
        f"{result['fps_mean']:.2f} ± "
        f"{result['fps_std']:.2f} FPS"
    )

    return result


def print_system_info():

    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)

    print("OS:", platform.platform())
    print("Python:", platform.python_version())
    print("CPU:", platform.processor())

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--patient",
        default="s0058",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    print_system_info()

    results = []

    # -------------------------------------------------------
    # U-Net backend
    # -------------------------------------------------------
    results.append(
        run_backend(
            name="U-Net",
            patient=args.patient,
            skip_unet=False,
            repeats=args.repeats,
            warmup=args.warmup,
            steps=args.steps,
        )
    )

    # -------------------------------------------------------
    # Non-U-Net / model-based path
    # -------------------------------------------------------
    results.append(
        run_backend(
            name="Non-U-Net",
            patient=args.patient,
            skip_unet=True,
            repeats=args.repeats,
            warmup=args.warmup,
            steps=args.steps,
        )
    )

    output = {
        "protocol": {
            "patient": args.patient,
            "repeats": args.repeats,
            "warmup_steps": args.warmup,
            "timed_steps": args.steps,
            "metric": "complete env.step() throughput",
        },
        "results": results,
    }

    output_file = Path(
        "fps_benchmark_results.json"
    )

    with open(output_file, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("PAPER-READY RESULTS")
    print("=" * 80)

    for r in results:

        print(
            f"{r['backend']}: "
            f"{r['fps_mean']:.2f} ± "
            f"{r['fps_std']:.2f} FPS"
        )

    print("\nSaved:")
    print(output_file.resolve())


if __name__ == "__main__":
    main()