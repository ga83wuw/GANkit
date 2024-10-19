# tests/test_training.py

import torch
from src.models.dcgan import Generator, Discriminator
from src.trainers.trainer_dcgan import DCGANTrainer
from src.datasets.dataloader import get_data_loader
from src.utils.device import get_device

def test_training_step():
    # Minimal configuration for testing
    config = {
        'data': {
            'dataset_path': './data/test_dataset',
            'image_size': 64,
            'batch_size': 4,
            'num_workers': 0,
            'image_channels': 3,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        },
        'training': {
            'num_epochs': 1,
            'learning_rate': 0.0002,
            'batch_size': 4,
            'latent_dim': 100
        },
        'model': {
            'name': 'dcgan',
            'latent_dim': 100,
            'feature_d': 64,
            'feature_g': 64
        },
        'sample_dir': 'samples',
        'model_dir': 'models'
    }

    device = get_device()
    data_loader = get_data_loader(
        data_dir=config['data']['dataset_path'],
        image_size=config['data']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        stats=(config['data']['mean'], config['data']['std'])
    )

    trainer = DCGANTrainer(config, data_loader)
    trainer.train()

