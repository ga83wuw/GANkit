# src/datasets/dataloader.py

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def get_data_loader(data_dir, image_size, batch_size, num_workers, stats=((0.5,)*3, (0.5,)*3)):
    """
    Returns a DataLoader for the dataset located at data_dir.

    Args:
        data_dir (str): Path to the dataset directory.
        image_size (int): Target size for images.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        stats (tuple): Tuple of mean and std for normalization.

    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return data_loader

