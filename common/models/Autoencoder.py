import torch
import torch.nn as nn
from torch import distributions as dist
from . import encoders, decoders

class Autoencoder(nn.Module):
    ''' Autoencoder class

    Args:
        cfg (dict): config
        device (device): torch device
    '''
    def __init__(self, cfg, device=None, **kwargs) -> None:
        super().__init__()

        encoder = getattr(encoders, cfg['model']['encoder'])(cfg, **kwargs)
        decoder = getattr(decoders, cfg['model']['decoder'])(cfg, **kwargs)

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self._device = device
    
    def forward(self, v, **kwargs):
        '''Forward pass through the network
        
        Args:
            v (tensor): [B, 3V] vertices
        Returns:
            v_hat (tensor): [B, 3V] reconstructed vertices
        Shape:
            B: batch size
            V: number of vertices per mesh
        '''
        z = self.encoder(v, **kwargs)
        v_hat = self.decoder(z, **kwargs)
        return v_hat

    def to(self, device):
        ''' Put the model to the device

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def decode(self, z):
        '''Decode the latent vector
        
        Args:
            z (tensor): [B, L]
        Shape:
            B: batch size
            L: latent dimenstion
        '''
        return self.decoder(z)
