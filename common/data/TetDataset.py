import glob, os
import torch
import numpy as np
import pandas as pd
import scipy as sp
import pyvista as pv
import logging
from torch.utils.data import Dataset
import trimesh

logger = logging.getLogger(__name__)

class TetDataset(Dataset):
    '''Tetrahedron Dataset
    
    Args:
        cfg (dict): config
    '''
    def __init__(self, cfg, **kwargs):
        self.dtype = torch.get_default_dtype()
        dataset_path = cfg['data']['path']
        self.dataset_path = dataset_path

        self.tet = pv.UnstructuredGrid(os.path.join(dataset_path, 'ref.vtu'))
        cells = self.tet.cells.reshape(-1, 5)
        assert (cells[:, 0] == 4).all(), 'input is not a valid tetramesh'
        self.cells = torch.as_tensor(cells[:,-4:], dtype=int)
        self.vertices = torch.as_tensor(self.tet.points, dtype=self.dtype).view(-1,3)

        self.surface_mesh = self.tet.extract_surface()
        surface_mesh_faces = self.surface_mesh.faces.reshape(-1, 4)
        assert (surface_mesh_faces[:,0] == 3).all(), 'surface mesh should only contain triangles'
        # Convert vertex indices from surface mesh to tetra mesh
        surface_mesh_faces = self.surface_mesh['vtkOriginalPointIds'][surface_mesh_faces[:,-3:]]
        self.vis_faces = torch.as_tensor(surface_mesh_faces, dtype=int)

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
        return self.vertices.view(-1)
    def get_obj_vertices(self):
        # for compatibility
        return self.vertices
    def get_obj_faces(self):
        # for compatibility
        return self.cells
    def get_vertices(self):
        return self.vertices
    def get_cells(self):
        return self.cells
    def get_params(self):
        return self.param_dict
    def get_model_input_dim(self):
        return 3 * self.tet.n_points # 3 (xyz)
    def get_vis_faces(self):
        '''get faces for visualization'''
        return self.vis_faces