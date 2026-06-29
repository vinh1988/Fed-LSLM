# uv run federated_code/02_fedavg.py --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --mode icl
# 02_fedavg.py

# Dynamically aggregate client adapters (client_{k}_icl or client_{k}_nonicl) to global adapter

# uv run federated_code/02_fedavg.py --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --mode icl
# uv run federated_code/02_fedavg.py --dataset VQA-RAD --setting noniid --alpha 0.1 --k 2 --mode nonicl

from pathlib import Path
import argparse
import shutil

from safetensors.torch import load_file, save_file

MODEL_SHORT = "Qwen2.5-VL-3B-Instruct"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="VQA-RAD")
    parser.add_argument("--setting", type=str, default="noniid")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--mode", type=str, choices=["icl", "nonicl"], required=True,
                        help="Mode: icl or nonicl")
    args = parser.parse_args()

    # Build base path
    base = Path("experiments") / args.dataset / args.setting
    if args.setting == "noniid":
        base = base / f"alpha_{args.alpha}"
    base = base / f"k{args.k}" / MODEL_SHORT

    # Build client paths with suffix
    client_paths = []
    for client_id in range(1, args.k + 1):
        client_path = base / "clients" / f"client_{client_id}_{args.mode}"
        client_paths.append(client_path)

    # Build server path
    server_path = base / "server" / f"global_adapter_{args.mode}"
    server_path.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating {args.k} clients from {args.mode} mode...")
    for i, path in enumerate(client_paths):
        print(f"  Client {i+1}: {path}")
    print(f"Server output: {server_path}")

    # Load all adapters
    adapters = []
    for client_path in client_paths:
        adapter = load_file(client_path / "adapter_model.safetensors")
        adapters.append(adapter)

    # Sanity check: all adapters have same keys
    for i in range(1, len(adapters)):
        assert adapters[i].keys() == adapters[0].keys(), f"Adapter {i+1} keys mismatch"

    # FedAvg: average all adapters
    global_adapter = {}
    for key in adapters[0].keys():
        sum_tensor = adapters[0][key]
        for adapter in adapters[1:]:
            sum_tensor += adapter[key]
        global_adapter[key] = sum_tensor / len(adapters)

    # Save global adapter
    save_file(
        global_adapter,
        str(server_path / "adapter_model.safetensors")
    )

    # Copy adapter config from first client
    shutil.copy(
        client_paths[0] / "adapter_config.json",
        server_path / "adapter_config.json"
    )

    print(f"Global adapter saved: {server_path}")

if __name__ == "__main__":
    main()