# tests/test_utils.py

import torch
from src.utils.visualization import denorm

def test_denorm():
    tensor = torch.zeros((1, 3, 64, 64))
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    denorm_tensor = denorm(tensor, mean, std)
    assert torch.allclose(denorm_tensor, torch.tensor(0.5))

