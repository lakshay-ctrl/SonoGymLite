"""
smoke_test_task2.py — Run 5k-step smoke tests for all distinct Task 2 configurations.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
PYTHON = sys.executable

SMOKE_TESTS = [
    {
        "name": "PPO + memory (spine, 30mm sphere)",
        "cmd": [PYTHON, "train_reconstruction_mb_occupancy.py", "--anatomy", "spine", "--algo", "ppo", "--timesteps", "5000", "--roi-radius-mm", "30", "--seed", "99"]
    },
    {
        "name": "PPO no memory (spine, 30mm sphere)",
        "cmd": [PYTHON, "train_reconstruction_mb_occupancy.py", "--anatomy", "spine", "--algo", "ppo", "--no-memory", "--timesteps", "5000", "--roi-radius-mm", "30", "--seed", "99"]
    },
    {
        "name": "SAC + memory (spine, 30mm sphere)",
        "cmd": [PYTHON, "train_reconstruction_mb_occupancy.py", "--anatomy", "spine", "--algo", "sac", "--timesteps", "5000", "--roi-radius-mm", "30", "--seed", "99"]
    },
    {
        "name": "PPO + memory whole-organ (spine, inf)",
        "cmd": [PYTHON, "train_reconstruction_mb_occupancy.py", "--anatomy", "spine", "--algo", "ppo", "--roi-radius-mm", "inf", "--timesteps", "5000", "--seed", "99"]
    },
    {
        "name": "Deep-FOV Liver (liver, 1.6mm spacing, 40mm sphere)",
        "cmd": [PYTHON, "train_reconstruction_mb_occupancy.py", "--anatomy", "liver", "--algo", "ppo", "--pixel-spacing", "1.6", "--roi-radius-mm", "40", "--timesteps", "5000", "--seed", "99"]
    }
]

def run_smoke_tests():
    print("=" * 80)
    print("  TASK 2 STEP 0: RUNNING 5K-STEP SMOKE TESTS FOR ALL 5 CONFIGURATIONS")
    print("=" * 80)
    
    results = {}
    for st in SMOKE_TESTS:
        print(f"\n---> Starting: {st['name']} ...", flush=True)
        ret = subprocess.run(st["cmd"], cwd=str(PROJECT_DIR))
        success = (ret.returncode == 0)
        results[st["name"]] = success
        if success:
            print(f"  [PASS] {st['name']}")
        else:
            print(f"  [FAIL] {st['name']} (returncode {ret.returncode})")

    print("\n" + "=" * 80)
    print("  SMOKE TEST SUMMARY")
    print("=" * 80)
    all_pass = True
    for name, res in results.items():
        status = "PASSED" if res else "FAILED"
        print(f"  - {name}: {status}")
        if not res:
            all_pass = False

    return all_pass

if __name__ == "__main__":
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
