# Overall Training Pipeline

```text
01_train_client.py
        │
        ▼
client_i/adapter
        │
        ▼
02_fedavg.py
        │
        ▼
server/global_adapter
        │
        ▼
03_train_roco.py      # Continue LoRA training on the ROCO reasoning dataset [Response distillation, Reasoning distillation, Instruction tuning]
        │
        ▼
server/global_adapter_roco
        │
        ▼
04_train_vqarad_continue.py
        │
        ▼
server/global_adapter_final
        │
        ▼
05_evaluate.py
```


# Training Pipeline

Since you've already completed:

* ✅ Client training
* ✅ FedAvg aggregation

the remaining scripts should be developed in the following order.

---

## 03_train_roco.py *(~300–400 lines)*

### Objective

Fine-tune the aggregated global LoRA adapter on the ROCO reasoning dataset.

### Workflow

1. Load the base **Qwen2.5-VL** model.
2. Load the aggregated adapter from:

```text
server/global_adapter
```

using:

```python
PeftModel.from_pretrained(..., is_trainable=True)
```

3. Train on:

```text
roco_reasoning_10.csv
```

(or replace it with the full ROCO reasoning dataset later).

4. Save the updated adapter to:

```text
server/global_adapter_roco
```

---

## 04_continue_vqarad.py *(~500 lines)*

### Objective

Continue fine-tuning on the VQA-RAD dataset using the ROCO-trained adapter.

### Workflow

Reuse almost all of the existing centralized VQA-RAD training code.

The only major change is replacing the fresh LoRA initialization:

```python
get_peft_model(...)
```

with loading the pretrained adapter:

```python
server/global_adapter_roco
```

using:

```python
PeftModel.from_pretrained(..., is_trainable=True)
```

Continue training on **VQA-RAD** and save the resulting adapter to:

```text
server/global_adapter_roco_vqarad
```

---

## 05_eval.py

### Objective

Evaluate any trained adapter on the VQA-RAD test set.

### Workflow

Reuse the existing evaluation pipeline.

The script should:

1. Load the base model.
2. Load the adapter specified by the user.
3. Evaluate on:

```text
processed_data/VQA-RAD/centralized/test.csv
```
