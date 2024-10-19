# tests/test_models.py

import torch
from src.models.dcgan import Generator, Discriminator

def test_generator_output_shape():
    latent_dim = 100
    gen = Generator(latent_dim=latent_dim)
    noise = torch.randn(16, latent_dim, 1, 1)
    fake_images = gen(noise)
    assert fake_images.shape == (16, 3, 64, 64)

def test_discriminator_output_shape():
    disc = Discriminator()
    images = torch.randn(16, 3, 64, 64)
    outputs = disc(images)
    assert outputs.shape == (16, 1)

