# src/utils/visualization.py

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

def denorm(tensor, stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    """
    Denormalizes the input tensor.

    Args:
        tensor (torch.Tensor): Normalized tensor.
        stats[1][0] (tuple): Mean used for normalization.
        stats[0][0] (tuple): Std deviation used for normalization.

    Returns:
        torch.Tensor: Denormalized tensor.
    """
    return tensor * stats[1][0] + stats[0][0]

def save_samples(epoch, fixed_noise, generator, sample_dir, stats=((0.5,)*3, (0.5,)*3)):
    """
    Generates and saves sample images.

    Args:
        epoch (int): Current epoch number.
        fixed_noise (torch.Tensor): Fixed noise vector for generating samples.
        generator (nn.Module): Generator model.
        sample_dir (str): Directory to save samples.
        stats (tuple): Tuple of mean and std for denormalization.
    """
    os.makedirs(sample_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
    fake_fname = f'generated-images-{epoch:04d}.png'
    save_image(denorm(fake_images, *stats), os.path.join(sample_dir, fake_fname), nrow=8)
    print(f'Saving {fake_fname}')
    generator.train()

