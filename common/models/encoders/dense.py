import torch
from torch import nn

class DenseEncoder(nn.Module):
    '''Dense encoder class
    
    Args:
        cfg (dict): config
        dim (int): input dimension
    '''
    def __init__(self, cfg, dim, **kwargs):
        super().__init__()
        self.dim = dim
        encoder_kwargs = cfg['model']['encoder_kwargs']
        self.latent_dim = encoder_kwargs['latent_dim']
        self.hidden_dim = encoder_kwargs.get('hidden_dim', 128)
        self.n_blocks = encoder_kwargs.get('n_blocks', 3)
        actvn = encoder_kwargs.get('actvn', 'relu')

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
        
        self.fc_in = nn.Linear(self.dim, self.hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_blocks)
        ])
        self.fc_out = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        x = self.fc_in(x)
        for block in self.blocks:
            x = self.actvn(x)
            x = block(x)
        out = self.actvn(x)
        out = self.fc_out(out)
        return out

