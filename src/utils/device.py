# src/utils/device.py

import torch

def get_device():
    """
    Returns the appropriate device (GPU/CPU) for computation.

    Returns:
        torch.device: 'cuda' if available, else 'cpu'.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

