#!/usr/bin/env python3
"""
Collect all metrics from final_val_* directories in experiments/VQA-RAD
and save to a CSV file with a structured format for analysis.
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import re

def parse_experiment_path(path: Path) -> Dict[str, str]:
    """
    Parse experiment directory path to extract metadata.
    
    Federated path structure:
    experiments/VQA-RAD/{setting}/{alpha}/k{k}/{model}/final_val_{mode}
    
    Centralized path structure:
    experiments/VQA-RAD/centralized/{model}/{mode}/best_lora_adapter/val
    """
    parts = path.parts
    
    # Default values
    metadata = {
        "dataset": None,
        "experiment_type": "federated",
        "setting": None,
        "alpha": None,
        "k_clients": None,
        "model": None,
        "mode": None
    }
    
    try:
        # Find the experiment root
        exp_idx = parts.index("experiments")
        dataset_idx = exp_idx + 1  # VQA-RAD or other dataset
        
        if dataset_idx >= len(parts):
            return metadata
        
        # Extract dataset name
        metadata["dataset"] = parts[dataset_idx]
        
        # Extract setting (noniid, iid, centralized)
        if dataset_idx + 1 < len(parts):
            metadata["setting"] = parts[dataset_idx + 1]
        
        # Handle centralized setting
        if metadata["setting"] == "centralized":
            metadata["experiment_type"] = "centralized"
            metadata["alpha"] = None
            metadata["k_clients"] = None
            
            # Extract model and mode from centralized path
            # Path: experiments/VQA-RAD/centralized/{model}/{mode}/best_lora_adapter/val
            if dataset_idx + 2 < len(parts):
                metadata["model"] = parts[dataset_idx + 2]
            if dataset_idx + 3 < len(parts):
                metadata["mode"] = parts[dataset_idx + 3]
            return metadata
        
        # Handle federated (noniid/iid) settings
        if metadata["setting"] == "noniid" and dataset_idx + 2 < len(parts):
            alpha_part = parts[dataset_idx + 2]
            if alpha_part.startswith("alpha_"):
                metadata["alpha"] = alpha_part.replace("alpha_", "")
                # Extract k from next part
                if dataset_idx + 3 < len(parts):
                    k_part = parts[dataset_idx + 3]
                    if k_part.startswith("k"):
                        metadata["k_clients"] = k_part[1:]
        
        # Handle iid setting
        if metadata["setting"] == "iid":
            metadata["alpha"] = None
            if dataset_idx + 2 < len(parts):
                k_part = parts[dataset_idx + 2]
                if k_part.startswith("k"):
                    metadata["k_clients"] = k_part[1:]
        
        # Extract model name and mode for federated
        # Model is typically before final_val_*
        for i, part in enumerate(parts):
            if part.startswith("final_val_"):
                if i > 0:
                    metadata["model"] = parts[i - 1]
                # Extract mode from final_val_{mode}
                metadata["mode"] = part.replace("final_val_", "")
                break
    
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse path {path}: {e}")
    
    return metadata

def read_metrics_json(metrics_path: Path) -> Optional[Dict]:
    """Read metrics.json file."""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {metrics_path}: {e}")
        return None

def find_client_1(final_val_path: Path, mode: str) -> Optional[Path]:
    """
    Find client_1 directory matching the mode.
    Expected pattern: clients/client_1_*{mode}
    """
    # Navigate to the parent directory that contains 'clients'
    parent = final_val_path.parent
    clients_dir = parent / "clients"
    
    if not clients_dir.exists():
        return None
    
    # Find client_1 directory with matching mode suffix
    for client_dir in clients_dir.glob("client_1_*"):
        if client_dir.name.endswith(mode):
            return client_dir
    
    return None

def read_hardware_json(client_path: Path) -> Optional[float]:
    """Read gpu_memory_gb from hardware.json."""
    hardware_path = client_path / "hardware.json"
    try:
        with open(hardware_path, 'r') as f:
            data = json.load(f)
            return data.get("avg_allocated_vram_gb")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {hardware_path}: {e}")
        return None

def read_training_time_json(client_path: Path) -> Optional[float]:
    """Read training_time_s from training_time.json."""
    time_path = client_path / "training_time.json"
    try:
        with open(time_path, 'r') as f:
            data = json.load(f)
            return data.get("avg_epoch_time_s")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not read {time_path}: {e}")
        return None

def collect_all_metrics(experiments_root: Path) -> List[Dict]:
    """
    Collect all metrics from:
    - Federated: final_val_*/metrics.json
    - Centralized: centralized/{model}/{mode}/best_lora_adapter/val/metrics.json
    """
    all_results = []
    
    # Find all federated final_val_* directories
    for final_val_dir in experiments_root.rglob("final_val_*"):
        if not final_val_dir.is_dir():
            continue
        
        metrics_path = final_val_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"Warning: No metrics.json found in {final_val_dir}")
            continue
        
        # Parse metadata from path
        metadata = parse_experiment_path(final_val_dir)
        
        # Read metrics
        metrics = read_metrics_json(metrics_path)
        if metrics is None:
            continue
        
        # Extract mode from final_val directory name
        mode = metadata.get("mode", "")
        
        # Clean up mode: remove "04_train_client_continue_" prefix
        if mode.startswith("04_train_client_continue_"):
            mode = mode.replace("04_train_client_continue_", "")
            metadata["mode"] = mode
        
        # Find matching client_1 directory and extract hardware/training time
        gpu_memory_gb = None
        training_time_s = None
        
        if mode:
            client_1_path = find_client_1(final_val_dir, mode)
            if client_1_path:
                gpu_memory_gb = read_hardware_json(client_1_path)
                training_time_s = read_training_time_json(client_1_path)
        
        # Combine metadata and metrics
        result = {
            **metadata,
            "closed_n_samples": metrics.get("closed-ended", {}).get("n_samples", None),
            "closed_accuracy": metrics.get("closed-ended", {}).get("Accuracy", None),
            "open_n_samples": metrics.get("open-ended", {}).get("n_samples", None),
            "open_accuracy": metrics.get("open-ended", {}).get("Accuracy", None),
            "open_exact_match": metrics.get("open-ended", {}).get("Exact Match", None),
            "open_bleu": metrics.get("open-ended", {}).get("BLEU", None),
            "open_rouge_l": metrics.get("open-ended", {}).get("ROUGE-L", None),
            "open_bert_score": metrics.get("open-ended", {}).get("BERTScore", None),
            "hallucination_rate": metrics.get("hallucination", {}).get("Hallucination Rate", None),
            "inference_time_s": metrics.get("performance", {}).get("inference_time_s", None),
            "throughput_samples_per_sec": metrics.get("performance", {}).get("throughput_samples_per_sec", None),
            "training_time_s": training_time_s,
            "gpu_memory_gb": gpu_memory_gb,
        }
        
        all_results.append(result)
    
    # Find all centralized val directories
    centralized_root = experiments_root / "centralized"
    if centralized_root.exists():
        for model_dir in centralized_root.iterdir():
            if not model_dir.is_dir():
                continue
            
            for mode_dir in model_dir.iterdir():
                if not mode_dir.is_dir():
                    continue
                
                # Look for best_lora_adapter/val/metrics.json
                metrics_path = mode_dir / "best_lora_adapter" / "val" / "metrics.json"
                if not metrics_path.exists():
                    continue
                
                # Parse metadata from path
                metadata = parse_experiment_path(metrics_path)
                
                # Read metrics
                metrics = read_metrics_json(metrics_path)
                if metrics is None:
                    continue
                
                # Centralized experiments don't have client training data
                result = {
                    **metadata,
                    "closed_n_samples": metrics.get("closed-ended", {}).get("n_samples", None),
                    "closed_accuracy": metrics.get("closed-ended", {}).get("Accuracy", None),
                    "open_n_samples": metrics.get("open-ended", {}).get("n_samples", None),
                    "open_accuracy": metrics.get("open-ended", {}).get("Accuracy", None),
                    "open_exact_match": metrics.get("open-ended", {}).get("Exact Match", None),
                    "open_bleu": metrics.get("open-ended", {}).get("BLEU", None),
                    "open_rouge_l": metrics.get("open-ended", {}).get("ROUGE-L", None),
                    "open_bert_score": metrics.get("open-ended", {}).get("BERTScore", None),
                    "hallucination_rate": metrics.get("hallucination", {}).get("Hallucination Rate", None),
                    "inference_time_s": metrics.get("performance", {}).get("inference_time_s", None),
                    "throughput_samples_per_sec": metrics.get("performance", {}).get("throughput_samples_per_sec", None),
                    "training_time_s": None,
                    "gpu_memory_gb": None,
                }
                
                all_results.append(result)
    
    return all_results

def save_to_csv(results: List[Dict], output_path: Path):
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return
    
    # Define column order
    columns = [
        "dataset",
        "experiment_type",
        "setting",
        "alpha",
        "k_clients",
        "model",
        "mode",
        "closed_n_samples",
        "closed_accuracy",
        "open_n_samples",
        "open_accuracy",
        "open_exact_match",
        "open_bleu",
        "open_rouge_l",
        "open_bert_score",
        "hallucination_rate",
        "inference_time_s",
        "throughput_samples_per_sec",
        "training_time_s",
        "gpu_memory_gb"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"Saved {len(results)} results to {output_path}")

def main():
    # Get project root
    script_dir = Path(__file__).parent
    experiments_root = script_dir / "experiments"
    
    if not experiments_root.exists():
        print(f"Error: Experiments directory not found: {experiments_root}")
        return
    
    print(f"Scanning experiments directory: {experiments_root}")
    
    # Collect all metrics from all datasets
    all_results = []
    for dataset_dir in experiments_root.iterdir():
        if not dataset_dir.is_dir():
            continue
        print(f"  Scanning dataset: {dataset_dir.name}")
        results = collect_all_metrics(dataset_dir)
        all_results.extend(results)
    
    results = all_results
    
    # Exclude Qwen2.5-VL-3B-Instruct
    results = [r for r in results if r.get("model") != "Qwen2.5-VL-3B-Instruct"]
    
    if not results:
        print("No metrics found.")
        return
    
    # Sort results for better readability
    results.sort(key=lambda x: (
        x["experiment_type"],
        x["setting"] or "",
        x["alpha"] or "",
        int(x["k_clients"]) if x["k_clients"] and x["k_clients"].isdigit() else 0,
        x["model"] or "",
        x["mode"] or ""
    ))
    
    # Save to CSV
    output_path = script_dir / "experiments_metrics.csv"
    save_to_csv(results, output_path)
    
    # Print summary
    print("\nSummary:")
    print(f"Total experiments: {len(results)}")
    
    # Group by setting
    by_setting = {}
    for r in results:
        setting = r["setting"] or "unknown"
        if setting not in by_setting:
            by_setting[setting] = []
        by_setting[setting].append(r)
    
    for setting, exps in by_setting.items():
        print(f"  {setting}: {len(exps)} experiments")

if __name__ == "__main__":
    main()
