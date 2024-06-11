import glob, os
import torch
import numpy as np
import pandas as pd
import trimesh
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class ShellDataset(Dataset):
    '''Shell Dataset
    
    Args:
        cfg (dict): config
    '''
    def __init__(self, cfg, **kwargs):
        self.dtype = torch.get_default_dtype()
        dataset_path = cfg['data']['path']
        self.dataset_path = dataset_path
        obj_mesh = trimesh.load(os.path.join(dataset_path, 'ref.obj'))
        self.obj_mesh = obj_mesh
        self.obj_vertices = torch.as_tensor(obj_mesh.vertices, dtype=self.dtype)
        self.obj_faces = torch.as_tensor(obj_mesh.faces, dtype=int)
        
        # Load ref/rest state
        self.X = torch.as_tensor(pd.read_csv(os.path.join(dataset_path, f'ref.csv'), header=None).values, dtype=self.dtype).view(-1)
        
        # Load material
        params = pd.read_csv(os.path.join(dataset_path, f'params.csv'), header=None).values.flatten()
        self.param_dict = {
            'thickness': params[0],
            'kB': params[1],
            'k1': params[2],
            'k2': params[3],
            'k3': params[4],
            'k4': params[5],
            'kD': params[6],
            'rho': params[7],
            'matModel': int(params[8])
        }
        
    def __len__(self):
        return 0

    def __getitem__(self, index):
        return None
    
    ''' getter functions '''
    def get_X(self):
        return self.X
    def get_obj_mesh(self):
        return self.obj_mesh
    def get_obj_vertices(self):
        return self.obj_vertices
    def get_obj_faces(self):
        return self.obj_faces
    def get_params(self):
        return self.param_dict
    def get_model_input_dim(self):
        return 3 * self.obj_vertices.size(0) # 3 (xyz)
    def get_vis_faces(self):
        '''get faces for visualization of the shell'''
        return self.obj_faces
