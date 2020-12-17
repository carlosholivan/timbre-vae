import torch

# Our modules
from vae import configs


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(bce, mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + configs.ParamsConfig.VAE_BETA * kld, kld
