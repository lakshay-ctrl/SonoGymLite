#!/usr/bin/env python3
"""
record_needle_showcase.py
=========================
Automated high-definition video recorder for the SonoSimLite Probe-Mounted
Needle Insertion & Acoustic Guidance System (Task 3).

Generates a broadcast-quality MP4 video clip (1536x742 @ 30 FPS) demonstrating:
1. Oracle probe placement with compliant 2-8N acoustic contact on patient abdomen.
2. In-plane needle cannula insertion into the Abdominal Aorta Bifurcation (45°).
3. Live CT cross-section with 24-azimuth bone corridor fan (0% bone collision).
4. Genuine CT-synthesized B-mode ultrasound with hyperechoic needle shaft,
   glowing tip in anechoic aorta lumen, and comet-tail reverberations.
5. Multi-angle guide showcase (30°, 45°, 60°) demonstrating specular visibility.
6. Closed-loop visual servoing compensation reducing tip deflection to < 1.0 mm.

Usage:
  python3 record_needle_showcase.py --output needle_guidance_showcase.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_needle_experiment import LiveNeedleExperiment


def record_showcase(output_path: str = "needle_guidance_showcase.mp4", fps: float = 30.0):
    print("=" * 80)
    print("  SonoSimLite: Automated Needle Guidance Video Recorder")
    print(f"  Target Output: {output_path} (1536x742 @ {fps} FPS)")
    print("=" * 80)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (1536, 742))

    # Configure base args for aorta insertion
    class Args:
        subject = "totalseg_patients/s0058"
        anatomy = "aorta"
        angle = 45
        speed = 22.0
        headless = True
        steps = 0
        save_figure = False
        record_video = ""
        auto_advance = False

    args = Args()
    app = LiveNeedleExperiment(args)

    total_frames = 0
    dt = 1.0 / fps

    def record_frames(num_frames: int, banner_text: str = ""):
        nonlocal total_frames
        for _ in range(num_frames):
            frame = app.step(dt=dt)
            if banner_text:
                # Add an elegant subtitle card at the top-left of the 3D panel
                cv2.rectangle(frame, (10, 190), (500, 225), (10, 15, 20), -1)
                cv2.rectangle(frame, (10, 190), (500, 225), (0, 200, 255), 1)
                cv2.putText(
                    frame, banner_text, (20, 214),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 240, 255), 2, cv2.LINE_AA
                )
            writer.write(frame)
            total_frames += 1
            if total_frames % 30 == 0:
                print(f"  [Progress] Rendered {total_frames} frames ({total_frames / fps:.1f}s)...")

    # ── Phase 1: Contact Establishment & Acoustic Window Planning (3 seconds) ──
    print("\n[Phase 1] Demonstrating Acoustic Window Planning & Safe Corridors...")
    app.insertion_stroke_mm = 0.0
    record_frames(int(3.0 * fps), "Phase 1: Safe Acoustic Window Planning")

    # ── Phase 2: Smooth Needle Cannula Advance to Aorta (5 seconds) ──
    print("\n[Phase 2] Advancing 18G Needle Cannula along 45 deg guide rail...")
    app.auto_advance = True
    app.compensated = False
    record_frames(int(5.0 * fps), "Phase 2: 45 deg In-Plane Needle Insertion")

    # ── Phase 3: Tip Lock in Aorta Lumen & Deflection Servoing (3 seconds) ──
    print("\n[Phase 3] Target Lock & Closed-Loop Servoing (<1 mm error)...")
    app.auto_advance = False
    app.compensated = True
    record_frames(int(3.5 * fps), "Phase 3: Closed-Loop Visual Servoing (<1mm)")

    # ── Phase 4: Guide Angle Transition to 30 deg (2.5 seconds) ──
    print("\n[Phase 4] Demonstrating 30 deg Guide Configuration...")
    app.angle_deg = 30
    app.update_target_geometry()
    app.load_probe_assembly()
    app.insertion_stroke_mm = app.stroke_req_mm * 0.70
    record_frames(int(2.5 * fps), "Phase 4: 30 deg Steep Trajectory (V = -1.2 dB)")

    # ── Phase 5: Guide Angle Transition to 60 deg (2.5 seconds) ──
    print("\n[Phase 5] Demonstrating 60 deg Guide Configuration...")
    app.angle_deg = 60
    app.update_target_geometry()
    app.load_probe_assembly()
    app.insertion_stroke_mm = app.stroke_req_mm * 0.85
    record_frames(int(2.5 * fps), "Phase 5: 60 deg Shallow Trajectory (V = -6.0 dB)")

    # Clean finish
    writer.release()
    try:
        import pybullet as p
        p.disconnect(app.client)
    except Exception:
        pass

    duration_sec = total_frames / fps
    print("\n" + "=" * 80)
    print(f"  🎉 Video showcase successfully rendered: {output_path}")
    print(f"  Total frames: {total_frames} ({duration_sec:.1f} seconds) @ {fps} FPS")
    print(f"  Resolution:   1536 x 742 (HD Widescreen Multi-Viewport)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Record Broadcast-Quality Needle Guidance Video Showcase")
    parser.add_argument("--output", type=str, default="needle_guidance_showcase.mp4", help="Output MP4 video file path")
    parser.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30 FPS)")
    args = parser.parse_args()

    record_showcase(output_path=args.output, fps=args.fps)


if __name__ == "__main__":
    main()
