#!/usr/bin/env python3
# =========================================================
# 05_fedavg_final_icl.py
# FedAvg for LoRA adapters (safetensors format) - ICL version
# ONLY client_*_final_icl folders
# Output: global_adapter_final_icl
# =========================================================

import os
import shutil
import torch
from collections import OrderedDict
from safetensors.torch import load_file, save_file

# =========================================================
# CONFIG
# =========================================================

CLIENTS_DIR = r"D:\nckh_teacher_student\code\teacher_student\experiments\VQA-RAD\noniid\alpha_0.1\k2\Qwen2.5-VL-3B-Instruct\clients"

OUTPUT_DIR = r"D:\nckh_teacher_student\code\teacher_student\experiments\VQA-RAD\noniid\alpha_0.1\k2\Qwen2.5-VL-3B-Instruct\server\global_adapter_final_icl"

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
    print(f"[SAVE] adapter_config.json → {dst_config}")


# =========================================================
# MAIN
# =========================================================

def main():

    print("\n==============================")
    print(" FEDAVG FINAL ICL (LoRA / PEFT) ")
    print("==============================\n")

    # -----------------------------------------------------
    # Only client_*_final_icl
    # -----------------------------------------------------
    client_names = [
        d for d in os.listdir(CLIENTS_DIR)
        if d.startswith("client_")
        and d.endswith("_final_icl")
        and os.path.isdir(os.path.join(CLIENTS_DIR, d))
    ]

    if len(client_names) == 0:
        raise ValueError("No client_*_final_icl folders found!")

    client_paths = [os.path.join(CLIENTS_DIR, c) for c in client_names]

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

    print("\nAll clients compatible ✔")

    # -----------------------------------------------------
    # FedAvg aggregation
    # -----------------------------------------------------
    print("Running FedAvg...")

    global_state = fedavg(client_states)

    # -----------------------------------------------------
    # Save output directory
    # -----------------------------------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    save_path = os.path.join(OUTPUT_DIR, "adapter_model.safetensors")

    save_file(global_state, save_path)

    print(f"\nSaved global weights → {save_path}")

    # -----------------------------------------------------
    # Copy adapter config (CRITICAL)
    # -----------------------------------------------------
    copy_adapter_config(client_paths[0], OUTPUT_DIR)

    print("\nDONE ✔ Global adapter is PEFT-ready")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()