import torch
import torch.nn as nn
from torch import distributions as dist
from . import encoders, decoders

class PCA_Autoencoder(nn.Module):
    ''' Autoencoder wrapped by PCA

    Args:
        cfg (dict): config
        device (device): torch device
    '''
    def __init__(self, cfg, device=None, **kwargs) -> None:
        super().__init__()
        self.dim = kwargs['dim']
        self.pca_dim = cfg['model']['pca_dim']
        kwargs['dim'] = self.pca_dim
        encoder = getattr(encoders, cfg['model']['encoder'])(cfg, **kwargs)
        decoder = getattr(decoders, cfg['model']['decoder'])(cfg, **kwargs)

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self._device = device

        self.center = nn.Parameter(torch.zeros(self.dim, device=device), requires_grad=False)
        self.pc = nn.Parameter(torch.zeros(self.pca_dim, self.dim, device=device), requires_grad=False)

    @torch.no_grad()
    def set_pca(self, center, Vh):
        self.center = nn.Parameter(center, requires_grad=False)
        self.pc = nn.Parameter(Vh[:self.pca_dim,:], requires_grad=False)
    
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
        u = (v - self.center.view(1, -1)) @ self.pc.T
        z = self.encoder(u, **kwargs)
        u_hat = self.decoder(z, **kwargs)
        v_hat = u_hat @ self.pc + self.center.view(1, -1)
        return v_hat

    def to(self, device):
        ''' Put the model to the device

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model
    
    def encode(self, v, **kwargs):
        '''Encode the input vector
        
        Args:
            v (tensor): [B, 3V] vertices
        Returns:
            z (tensor): [B, L]
        Shape:
            B: batch size
            V: number of vertices per mesh
            L: latent dimenstion
        '''
        u = (v - self.center.view(1, -1)) @ self.pc.T
        z = self.encoder(u, **kwargs)
        return z

    def decode(self, z, **kwargs):
        '''Decode the latent vector
        
        Args:
            z (tensor): [B, L]
        Shape:
            B: batch size
            L: latent dimenstion
        '''
        u_hat = self.decoder(z, **kwargs)
        v_hat = u_hat @ self.pc + self.center.view(1, -1)
        return v_hat
