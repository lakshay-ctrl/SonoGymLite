#!/usr/bin/env bash
# ==============================================================================
# SonoGymLite: Pretrained Model & Checkpoint Downloader
# ==============================================================================
# This script downloads the official pretrained reinforcement learning checkpoints,
# Action Chunking Transformer (ACT) policies, and Cavalcanti U-Net generative
# ultrasound synthesis models for SonoGymLite.
#
# Checkpoints included:
#   - PPO & SAC models for Spine, Liver, and Aorta navigation
#   - ACT imitation learning policy weights and normalization statistics
#   - 2-channel U-Net generator weights for Cavalcanti spine simulation
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "===================================================================="
echo "  SonoGymLite Checkpoint Downloader"
echo "===================================================================="

# Remote archive hosting URL (Update with official Zenodo / GitHub Release URL upon camera-ready publication)
WEIGHTS_URL="${SONOSIMLITE_WEIGHTS_URL:-https://github.com/lakshaybatta/ct_us/releases/download/v1.0.0/sonogymlite_checkpoints.tar.gz}"
WEIGHTS_FILE="${PROJECT_ROOT}/sonogymlite_checkpoints.tar.gz"

echo "[1/3] Downloading checkpoint bundle from: ${WEIGHTS_URL}"
if command -v curl >/dev/null 2>&1; then
    curl -L "${WEIGHTS_URL}" -o "${WEIGHTS_FILE}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${WEIGHTS_FILE}" "${WEIGHTS_URL}"
else
    echo "Error: neither curl nor wget found. Please install either to download weights."
    exit 1
fi

echo "[2/3] Extracting checkpoints into project root..."
tar -xzvf "${WEIGHTS_FILE}" -C "${PROJECT_ROOT}"

echo "[3/3] Cleaning up temporary archive..."
rm -f "${WEIGHTS_FILE}"

echo "===================================================================="
echo "  Checkpoints successfully installed!"
echo "  You can now evaluate trained policies with:"
echo "    python3 enjoy_policy.py --algo sac --anatomy spine"
echo "    python3 enjoy_act.py"
echo "===================================================================="
