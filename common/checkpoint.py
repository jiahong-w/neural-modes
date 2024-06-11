import os
import torch
import logging

logger = logging.getLogger(__name__)

class Checkpoint():
    def __init__(self, dir):
        self.dir = dir
        os.makedirs(dir, exist_ok=True)
        self.iter = -1
        self.epoch = -1

    def load(self, filename, device, **kwargs):
        '''
        The function loads from file to kwargs, example kwargs 
            {model=model, 'optimizer'=optimizer}
        '''
        if filename is None:
            # start from scratch
            return
            
        filepath = os.path.join(self.dir, filename)
        assert os.path.isfile(filepath), f'{filepath} does not exist'
        
        state_dict = torch.load(filepath, map_location=device)
        for k, v in kwargs.items():
            if k in state_dict:
                v.load_state_dict(state_dict[k])
            else:
                logger.error(f'state dict for {k} not found in file {filename}')

        self.iter = state_dict.get('iter', -1)
        self.epoch = state_dict.get('epoch', -1)

    def save(self, filename, **kwargs):
        state_dict = {k: v.state_dict() for k, v in kwargs.items()}
        state_dict['iter'] = self.iter
        state_dict['epoch'] = self.epoch

        filepath = os.path.join(self.dir, filename)
        torch.save(state_dict, filepath)
        logger.info(f'save checkpoint at {filepath}')

    def step(self):
        self.iter += 1

    def step_epoch(self):
        self.epoch += 1