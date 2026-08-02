#!/usr/bin/env python3
# =========================================================
# 05_fedavg_final_mrag_kd.py
# FedAvg for LoRA adapters (safetensors format) - mRAG KD version
# Aggregates client adapters from step 04 (continue training)
# Output: global_adapter_{mode}_continue
# uv run federated_code/05_fedavg_final_mrag_kd.py --mode 04_train_client_continue_mrag_kd --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2
# =========================================================

import os
import shutil
import argparse
import torch
from collections import OrderedDict
from pathlib import Path
from safetensors.torch import load_file, save_file

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cpu"


# =========================================================
# LOAD CLIENT LORA
# =========================================================

def load_client_state(client_path):
    path = os.path.join(client_path, "adapter_model.safetensors")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")

    print(f"[LOAD] {path}")
    return load_file(path, device=DEVICE)


# =========================================================
# FEDERATED AVERAGING
# =========================================================

def fedavg(state_dicts):
    avg_state = OrderedDict()
    num_clients = len(state_dicts)

    ref_keys = state_dicts[0].keys()

    for key in ref_keys:
        avg_state[key] = sum(
            sd[key].float() for sd in state_dicts
        ) / num_clients

    return avg_state


# =========================================================
# COPY CONFIG (IMPORTANT FOR PEFT)
# =========================================================

def copy_adapter_config(src_client_path, dst_path):
    src_config = os.path.join(src_client_path, "adapter_config.json")
    dst_config = os.path.join(dst_path, "adapter_config.json")

    if not os.path.exists(src_config):
        raise FileNotFoundError(f"Missing adapter_config.json in {src_client_path}")

    shutil.copy(src_config, dst_config)
    print(f"[SAVE] adapter_config.json to {dst_config}")


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        help="Script name suffix from step 04 (e.g., 04_train_client_continue_mrag_kd)")
    parser.add_argument("--dataset", type=str, default="VQA-RAD")
    parser.add_argument("--setting", type=str, choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--model_short", type=str, default="Qwen2.5-VL-3B-Instruct",
                        help="Short model name for directory naming")
    parser.add_argument("--clients_dir", type=str, default=None,
                        help="Path to clients directory (auto-generated if not provided)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output directory (auto-generated if not provided)")
    args = parser.parse_args()

    # Auto-generate paths if not provided
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / args.model_short
    
    if args.clients_dir is None:
        args.clients_dir = str(base / "clients")
    if args.output_dir is None:
        args.output_dir = str(base / "server" / f"global_adapter_{args.mode}")

    print("\n==============================")
    print(f" FEDAVG FINAL {args.mode.upper()} (LoRA / PEFT) ")
    print("==============================\n")

    # -----------------------------------------------------
    # Find client directories matching the mode suffix
    # -----------------------------------------------------
    clients_dir = Path(args.clients_dir)
    client_names = [
        d for d in os.listdir(clients_dir)
        if d.startswith("client_")
        and d.endswith(f"_{args.mode}")
        and os.path.isdir(clients_dir / d)
    ]

    if len(client_names) == 0:
        raise ValueError(f"No client_*_{args.mode} folders found in {clients_dir}!")

    client_paths = [clients_dir / c for c in client_names]

    print(f"Found {len(client_paths)} clients:")
    for c in client_names:
        print(" -", c)

    # -----------------------------------------------------
    # Load LoRA adapters
    # -----------------------------------------------------
    client_states = []

    for path in client_paths:
        client_states.append(load_client_state(path))

    # -----------------------------------------------------
    # Validate keys
    # -----------------------------------------------------
    ref_keys = set(client_states[0].keys())

    for i, sd in enumerate(client_states):
        if set(sd.keys()) != ref_keys:
            raise ValueError(f"Mismatch in client {client_names[i]}")

    print("\nAll clients compatible OK")

    # -----------------------------------------------------
    # FedAvg aggregation
    # -----------------------------------------------------
    print("Running FedAvg...")

    global_state = fedavg(client_states)

    # -----------------------------------------------------
    # Save output directory
    # -----------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "adapter_model.safetensors"

    save_file(global_state, str(save_path))

    print(f"\nSaved global weights to {save_path}")

    # -----------------------------------------------------
    # Copy adapter config (CRITICAL)
    # -----------------------------------------------------
    copy_adapter_config(str(client_paths[0]), str(output_dir))

    print("\nDONE. Global adapter is PEFT-ready")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()