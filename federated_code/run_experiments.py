#!/usr/bin/env python3
"""

.\.venv\Scripts\python.exe federated_code/run_experiments.py


run_experiments.py

Automation script to sequentially run all Medical VQA Federated Learning experiments:
1. IID Setting (K=3, rounds=10)
2. Non-IID Setting (alpha=0.5, K=3, rounds=10)
3. Non-IID Setting (alpha=0.1, K=3, rounds=10)

Uses the active virtualenv python interpreter to launch the training runs.
"""

import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n================================================================================")
    print(f"[{ts}] {msg}")
    print(f"================================================================================\n")

def run_experiment(args: list) -> int:
    # Use the same python interpreter running this script
    cmd = [sys.executable, "federated_code/train_federated.py"] + args
    log(f"LAUNCHING EXPERIMENT:\n{' '.join(cmd)}")
    
    start_time = time.time()
    # Execute the command, forwarding stdout/stderr to the console in real-time
    result = subprocess.run(cmd, check=False)
    
    elapsed = time.time() - start_time
    m, s = divmod(int(elapsed), 60)
    
    if result.returncode == 0:
        log(f"SUCCESS: Experiment completed in {m}m {s}s.")
    else:
        log(f"FAILURE: Experiment failed with code {result.returncode} after {m}m {s}s.")
        
    return result.returncode

def main():
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_federated.py"
    
    if not train_script.exists():
        print(f"Error: train_federated.py not found at {train_script}")
        sys.exit(1)
        
    log("Starting Medical VQA Federated Learning Experiment Suite")
    suite_start = time.time()
    
    # Define experiment runs
    configs = [
        {
            "name": "IID Setting (K=3, Rounds=10)",
            "args": ["--setting", "iid", "--num_clients", "3", "--num_rounds", "10"]
        },
        {
            "name": "Non-IID Setting (alpha=0.5, K=3, Rounds=10)",
            "args": ["--setting", "noniid", "--alpha", "0.5", "--num_clients", "3", "--num_rounds", "10"]
        },
        {
            "name": "Non-IID Setting (alpha=0.1, K=3, Rounds=10)",
            "args": ["--setting", "noniid", "--alpha", "0.1", "--num_clients", "3", "--num_rounds", "10"]
        }
    ]
    
    summary = []
    
    for i, config in enumerate(configs, 1):
        print(f"\nExperiment {i}/{len(configs)}: {config['name']}")
        ret = run_experiment(config['args'])
        
        status = "SUCCESS" if ret == 0 else f"FAILED (code {ret})"
        summary.append(f"- Experiment {i}: {config['name']} -> {status}")
        
    total_elapsed = time.time() - suite_start
    tm, ts = divmod(int(total_elapsed), 60)
    
    log("All Federated Learning Experiments Finished!")
    print("Summary of runs:")
    for line in summary:
        print(line)
    print(f"\nTotal elapsed time: {tm}m {ts}s.")

if __name__ == "__main__":
    main()
