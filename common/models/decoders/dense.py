import torch
from torch import nn

class DenseDecoder(nn.Module):
    '''Dense decoder class
    
    Args:
        cfg (dict): config
        dim (int): input dimension
    '''
    def __init__(self, cfg, dim, **kwargs):
        super().__init__()
        self.dim = dim
        decoder_kwargs = cfg['model']['decoder_kwargs']
        self.latent_dim = decoder_kwargs['latent_dim']
        self.hidden_dim = decoder_kwargs.get('hidden_dim', 128)
        self.n_blocks = decoder_kwargs.get('n_blocks', 3)
        actvn = decoder_kwargs.get('actvn', 'relu')

        if actvn == 'relu':
            self.actvn = nn.ReLU()
        elif actvn == 'sigmoid':
            self.actvn = nn.Sigmoid()
        elif actvn == 'tanh':
            self.actvn = nn.Tanh()
        elif actvn == 'elu':
            self.actvn = nn.ELU()
        elif actvn == 'silu':
            self.actvn = nn.SiLU()
        elif actvn == 'sin':
            self.actvn = torch.sin
        elif actvn == 'softplus':
            self.actvn = nn.Softplus()
        elif actvn == 'mish':
            self.actvn = nn.Mish()
        elif actvn == 'gelu':
            self.actvn = nn.GELU()
        else:
            raise NotImplementedError
        
        self.fc_in = nn.Linear(self.latent_dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_blocks)
        ])
        self.fc_out = nn.Linear(self.hidden_dim, self.dim)

    def forward(self, x):
        x = self.fc_in(x)
        for block in self.blocks:
            x = self.actvn(x)
            x = block(x)
        out = self.actvn(x)
        out = self.fc_out(out)
        return out
    
    def decode(self, z):
        return self.forward(z)
