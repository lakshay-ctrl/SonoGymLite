#!/usr/bin/env bash
set -e

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

log() {
    echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "======================================================================"
log "  STARTING COMPLETE MULTI-PATIENT MULTI-ORGAN RL OVERNIGHT PIPELINE"
log "  Algorithms:        PPO, SAC, A2C, TD3"
log "  Organs:            Spine, Liver, Aorta"
log "  Simulation Engine Separation:"
log "    - Spine L4:          Learning-Based (LB) 2-channel U-Net on GPU (--sim-mode unet)"
log "    - Liver & Aorta:     Model-Based (MB) Physics Engine (--sim-mode conv, SonoGym benchmark)"
log "  Training Patients: s0011, s0058, s0250 (Domain Randomized per episode)"
log "  Total:             12 Trained Production Checkpoints"
log "======================================================================"

python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# --- 1. PPO (Multi-Patient) ---
log ">>> [1/12] Training PPO on Spine L4 Vertebra (100k timesteps, 3 patients, U-Net GPU)..."
python3 train_ppo.py --timesteps 100000 --task navigation --anatomy spine --sim-mode unet --multi-patient --save-dir ./ppo_checkpoints/spine/ --tb-log ./ppo_tensorboard/spine/
mkdir -p ./ppo_checkpoints && cp ./ppo_checkpoints/spine/ppo_final_model.zip ./ppo_checkpoints/ppo_final_model.zip

log ">>> [2/12] Training PPO on Liver Portal Vein (100k timesteps, 3 patients, Physics MB)..."
python3 train_ppo.py --timesteps 100000 --task navigation --anatomy liver --sim-mode conv --multi-patient --save-dir ./ppo_checkpoints/liver/ --tb-log ./ppo_tensorboard/liver/

log ">>> [3/12] Training PPO on Aorta Bifurcation (100k timesteps, 3 patients, Physics MB)..."
python3 train_ppo.py --timesteps 100000 --task navigation --anatomy aorta --sim-mode conv --multi-patient --save-dir ./ppo_checkpoints/aorta/ --tb-log ./ppo_tensorboard/aorta/

# --- 2. SAC (Multi-Patient) ---
log ">>> [4/12] Training SAC on Spine L4 Vertebra (100k timesteps, 3 patients, U-Net GPU)..."
python3 train_sac.py --timesteps 100000 --task navigation --anatomy spine --sim-mode unet --multi-patient --buffer-size 10000 --save-dir ./sac_checkpoints/spine/ --tb-log ./sac_tensorboard/spine/
mkdir -p ./sac_checkpoints && cp ./sac_checkpoints/spine/sac_final_model.zip ./sac_checkpoints/sac_final_model.zip

log ">>> [5/12] Training SAC on Liver Portal Vein (100k timesteps, 3 patients, Physics MB)..."
python3 train_sac.py --timesteps 100000 --task navigation --anatomy liver --sim-mode conv --multi-patient --buffer-size 10000 --save-dir ./sac_checkpoints/liver/ --tb-log ./sac_tensorboard/liver/

log ">>> [6/12] Training SAC on Aorta Bifurcation (100k timesteps, 3 patients, Physics MB)..."
python3 train_sac.py --timesteps 100000 --task navigation --anatomy aorta --sim-mode conv --multi-patient --buffer-size 10000 --save-dir ./sac_checkpoints/aorta/ --tb-log ./sac_tensorboard/aorta/

# --- 3. A2C (Multi-Patient) ---
log ">>> [7/12] Training A2C on Spine L4 Vertebra (100k timesteps, 3 patients, U-Net GPU)..."
python3 train_a2c.py --timesteps 100000 --task navigation --anatomy spine --sim-mode unet --multi-patient --save-dir ./a2c_checkpoints/spine/ --tb-log ./a2c_tensorboard/spine/
mkdir -p ./a2c_checkpoints && cp ./a2c_checkpoints/spine/a2c_final_model.zip ./a2c_checkpoints/a2c_final_model.zip

log ">>> [8/12] Training A2C on Liver Portal Vein (100k timesteps, 3 patients, Physics MB)..."
python3 train_a2c.py --timesteps 100000 --task navigation --anatomy liver --sim-mode conv --multi-patient --save-dir ./a2c_checkpoints/liver/ --tb-log ./a2c_tensorboard/liver/

log ">>> [9/12] Training A2C on Aorta Bifurcation (100k timesteps, 3 patients, Physics MB)..."
python3 train_a2c.py --timesteps 100000 --task navigation --anatomy aorta --sim-mode conv --multi-patient --save-dir ./a2c_checkpoints/aorta/ --tb-log ./a2c_tensorboard/aorta/

# --- 4. TD3 (Multi-Patient) ---
log ">>> [10/12] Training TD3 on Spine L4 Vertebra (100k timesteps, 3 patients, U-Net GPU)..."
python3 train_td3.py --timesteps 100000 --task navigation --anatomy spine --sim-mode unet --multi-patient --buffer-size 10000 --save-dir ./td3_checkpoints/spine/ --tb-log ./td3_tensorboard/spine/
mkdir -p ./td3_checkpoints && cp ./td3_checkpoints/spine/td3_final_model.zip ./td3_checkpoints/td3_final_model.zip

log ">>> [11/12] Training TD3 on Liver Portal Vein (100k timesteps, 3 patients, Physics MB)..."
python3 train_td3.py --timesteps 100000 --task navigation --anatomy liver --sim-mode conv --multi-patient --buffer-size 10000 --save-dir ./td3_checkpoints/liver/ --tb-log ./td3_tensorboard/liver/

log ">>> [12/12] Training TD3 on Aorta Bifurcation (100k timesteps, 3 patients, Physics MB)..."
python3 train_td3.py --timesteps 100000 --task navigation --anatomy aorta --sim-mode conv --multi-patient --buffer-size 10000 --save-dir ./td3_checkpoints/aorta/ --tb-log ./td3_tensorboard/aorta/

# --- 5. Full Benchmark Evaluation ---
log ">>> [13/13] Evaluating all trained models across all organs..."
python3 run_full_benchmark.py --algos random bo ppo sac a2c td3 --episodes 5 --sim-mode conv --output-prefix benchmark_overnight_rl

log "======================================================================"
log "  ALL 12 RL MODELS TRAINED & BENCHMARK EVALUATIONS COMPLETED!"
log "  Results saved to: benchmark_overnight_rl.md and benchmark_overnight_rl.tex"
log "======================================================================"
