import yaml
from . import models, data
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from .loss import ShellLoss, TetLoss

# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (str): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

# Models
def get_model(cfg, device, **kwargs):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    model = getattr(models, method)(cfg, **kwargs)
    model.to(device)
    return model

# Energy
def get_energy(cfg, **kwargs):
    '''Returns the energy function
       Overwrites material model according to config loss
    
    Args:
        cfg (dict): config
    '''
    # overwrite material model
    loss_name = cfg['training']['loss']
    if 'stvk' in loss_name:
        kwargs['params']['matModel'] = 0
    elif 'nh' in loss_name:
        kwargs['params']['matModel'] = 1
    # get energy func
    shape_model = cfg['training'].get('shape_model', 'shell')
    if shape_model == 'shell':
        return ShellLoss(**kwargs)
    elif shape_model == 'tet':
        return TetLoss(**kwargs)
    else:
        raise ValueError

if __name__ == '__main__':
    print(dir(models))