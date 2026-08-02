import torch

print("CUDA available:", torch.cuda.is_available())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)