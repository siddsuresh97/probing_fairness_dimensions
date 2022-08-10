"""
    Simple convolutional VAE, suitable for MNIST experiments
"""

import numpy as np
import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, z_dim, c_dim, x_dim, *args): # x_dim : total number of pixels
        super(Encoder, self).__init__()
        self.linear = nn.Linear(x_dim * c_dim, z_dim)
    
    def encode(self, x):
        xflat = x.view(x.shape[0], -1)
        mu = self.linear(xflat)
        return mu, torch.zeros(mu.shape)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, z_dim, c_dim, x_dim, *args): # x_dim : total number of pixels
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.x_dim = x_dim
        self.linear = nn.Linear(z_dim, x_dim * c_dim)

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.linear(z)
        return x