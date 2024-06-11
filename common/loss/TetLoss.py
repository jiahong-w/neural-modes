import torch
import numpy as np
from .SOMaterial import SOMaterial, EMatModel
from .LinearTetElement import LinearTetElementVectorize

class TetLoss:
    '''Energy inspired loss function for tetrahedron mesh
    
    Args:
        params (dict): material parameters
        tetras/obj_faces (tensor): [T, 4] tetras
        vertices/obj_vertices (tensor): [V, 3] vertices
        vectorize (bool): whether to vectorize elements
    '''
    def __init__(self, params, tetras=None, vertices=None, obj_faces=None, obj_vertices=None, vectorize=True, **kwargs):
        self.m_mat = SOMaterial(params['kB'],
                                params['k1'],
                                params['k2'],
                                params['k3'],
                                params['k4'],
                                params['kD'],
                                params['rho'],
                                EMatModel(params['matModel']))
        if tetras is None:
            tetras = obj_faces
        assert tetras is not None, 'tetras and obj_faces cannot both be None'
        if vertices is None:
            vertices = obj_vertices
        assert vertices is not None, 'vertices and obj_vertices cannot both be None'
        self.device = tetras.device
        # initialize elements
        self.vectorize = vectorize
        if self.vectorize:
            self.m_tetElementsVec = self.initLinearTetElementsVectorize(tetras, vertices, **kwargs)
            self.mass = self.m_tetElementsVec.computeMass(vertices)
        else:
            raise NotImplementedError
    
    def initLinearTetElementsVectorize(self, tetras, vertices, **kwargs):
        '''Init linear tetrahedron elements
        
        Args:
            tetras (tensor): [T, 4] tetras
            vertices (tensor): [V, 3] vertices
        Returns:
            m_tetElements (LinearTetElementVectorize): vectorized tetra elements
        '''
        m_tetElements = LinearTetElementVectorize(tetras, self.m_mat, vertices, **kwargs)
        return m_tetElements
    
    def __call__(self, x_hat, x, k1=None, k2=None, restScale=None):
        '''Compute loss over a batch
        
        Args:
            x_hat (tensor): [B, V*C] reconstructed vertices
            x (tensor): [B, V*C] groundtruth vertices
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
            restScale (tensor): [] optional scaling factor of the rest shape
        Returns:
            W (tensor): [B] loss without reduction
        Shapes:
            B: batch size
            V: number of vertices
            C: 3 coordiantes
        '''
        B = x_hat.size(0)
        if self.vectorize:
            W_tet = self.m_tetElementsVec.batchComputeEnergy(x_hat.view(B, -1, 3), x.view(B, -1, 3), k1=k1, k2=k2, restScale=restScale)
            W = W_tet.sum(-1)
        else:
            raise NotImplementedError
        return W

    def getMass(self):
        return self.mass