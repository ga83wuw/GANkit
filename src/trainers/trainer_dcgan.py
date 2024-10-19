# src/trainers/trainer_dcgan.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.dcgan import Generator, Discriminator
from src.utils.device import get_device
from src.utils.visualization import save_samples
from src.utils.logger import Logger

class DCGANTrainer:
    """
    Trainer class for DCGAN.
    """
    def __init__(self, config, data_loader):
        self.device = get_device()
        self.data_loader = data_loader
        self.config = config

        self.generator = Generator(
            latent_dim=config['model']['latent_dim'],
            image_channels=config['data']['image_channels'],
            feature_g=config['model']['feature_g']
        ).to(self.device)

        self.discriminator = Discriminator(
            image_channels=config['data']['image_channels'],
            feature_d=config['model']['feature_d']
        ).to(self.device)

        self.criterion = nn.BCELoss()
        self.opt_g = optim.Adam(self.generator.parameters(), lr=config['training']['learning_rate'], betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=config['training']['learning_rate'], betas=(0.5, 0.999))

        self.fixed_noise = torch.randn(64, config['model']['latent_dim'], 1, 1, device=self.device)
        self.logger = Logger(config['training']['num_epochs'], len(self.data_loader))

    def train(self):
        """
        Train the DCGAN model.
        """
        num_epochs = self.config['training']['num_epochs']

        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(tqdm(self.data_loader)):
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)

                # Labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # Train Discriminator
                self.opt_d.zero_grad()

                outputs = self.discriminator(real_images)
                d_loss_real = self.criterion(outputs, real_labels)
                real_score = outputs.mean().item()

                noise = torch.randn(batch_size, self.config['model']['latent_dim'], 1, 1, device=self.device)
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images.detach())
                d_loss_fake = self.criterion(outputs, fake_labels)
                fake_score = outputs.mean().item()

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.opt_d.step()

                # Train Generator
                self.opt_g.zero_grad()
                outputs = self.discriminator(fake_images)
                g_loss = self.criterion(outputs, real_labels)
                g_loss.backward()
                self.opt_g.step()

                # Logging
                self.logger.log(d_loss.item(), g_loss.item(), real_score, fake_score, epoch, i)

            # Save Samples
            save_samples(epoch, self.fixed_noise, self.generator, self.config['sample_dir'])

        # Save Models
        torch.save(self.generator.state_dict(), f"{self.config['model_dir']}/generator.pth")
        torch.save(self.discriminator.state_dict(), f"{self.config['model_dir']}/discriminator.pth")

