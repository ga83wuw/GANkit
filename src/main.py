# src/main.py

import argparse
import yaml

from src.datasets.dataloader import get_data_loader
from src.trainers.trainer_dcgan import DCGANTrainer

def main(config):
    # Data Loader
    data_loader = get_data_loader(
        data_dir=config['data']['dataset_path'],
        image_size=config['data']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        stats=(config['data']['mean'], config['data']['std'])
    )

    # Trainer
    if config['model']['name'] == 'dcgan':
        trainer = DCGANTrainer(config, data_loader)
        trainer.train()
    else:
        raise NotImplementedError(f"Model {config['model']['name']} not implemented.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAN Training Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to the config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)

