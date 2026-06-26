# uv run federated_code/aggregate_results.py
# 02_fedavg.py

# Input

# client1_adapter
# client2_adapter
# ...

# Output

# server/global_adapter
from pathlib import Path
import shutil

from safetensors.torch import load_file, save_file


CLIENT1 = Path(
    r"experiments\VQA-RAD\noniid\alpha_0.1\k2\Qwen2.5-VL-3B-Instruct\clients\client_1"
)

CLIENT2 = Path(
    r"experiments\VQA-RAD\noniid\alpha_0.1\k2\Qwen2.5-VL-3B-Instruct\clients\client_2"
)

SERVER = Path(
    r"experiments\VQA-RAD\noniid\alpha_0.1\k2\Qwen2.5-VL-3B-Instruct\server\global_adapter"
)

SERVER.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load adapters
# --------------------------------------------------

adapter1 = load_file(CLIENT1 / "adapter_model.safetensors")
adapter2 = load_file(CLIENT2 / "adapter_model.safetensors")

# sanity check
assert adapter1.keys() == adapter2.keys()

# --------------------------------------------------
# FedAvg
# --------------------------------------------------

global_adapter = {}

for k in adapter1.keys():
    global_adapter[k] = (
        adapter1[k] + adapter2[k]
    ) / 2.0

# --------------------------------------------------
# Save global adapter
# --------------------------------------------------

save_file(
    global_adapter,
    str(SERVER / "adapter_model.safetensors")
)

shutil.copy(
    CLIENT1 / "adapter_config.json",
    SERVER / "adapter_config.json"
)

print("Global adapter saved:", SERVER)