import torch
from .SOMaterial import SOMaterial
from . import Maple
from dataclasses import dataclass

@dataclass
class Hinge:
    '''Hinge structure
    
    edge/half edge in ascending order
    flaps[0] and tris[0] corresponds to triangle contains the half edge
    flaps[1] and tris[1] corresponds to triangle contains the coexisting half edge
    '''
    edge: tuple[int] = (-1, -1)
    flaps: tuple[int] = (-1, -1)
    tris: tuple[int] = (-1, -1)

class DSHingeElement:

    def __init__(self, inds: torch.Tensor, mat: SOMaterial, **kwargs):
        '''Init function
        
        Args:
            inds (tensor): [4] vertex indices of the element
            mat (SOMaterial): material
        '''
        self.m_inds = inds
        self.m_mat = mat

    def computeEnergy(self, vx, vX):
        '''Computes energy
        
        Args:
            vx (tensor): [V, 3] current position
            vX (tensor): [V, 3] rest position
        Shape:
            V: number of vertices in all elements
        '''
        x = torch.index_select(vx, 0, self.m_inds)
        X = torch.index_select(vX, 0, self.m_inds)

        k_bend = self.m_mat.kB

        W = Maple.DSBending_W(x, X, k_bend)

        return W
    
class DSHingeElementVectorize:
    '''Vectorize DSHingeElement class, can compute all elements at once'''

    def __init__(self, inds: torch.Tensor, mat: SOMaterial, **kwargs):
        '''Init function
        
        Args:
            inds (tensor): [M, 4] vertex indices of the element
            mat (SOMaterial): material
        Shape:
            M: number of elements
        '''
        self.m_inds = inds
        self.m_mat = mat

    def batchComputeEnergy(self, vx, vX, kB=None):
        '''Computes energy in batch
        
        Args:
            vx (tensor): [B, V, 3] current position
            vX (tensor): [B, V, 3] rest position
            kB (tensor): [] optional material parameter
        Shape:
            B: batch size
            V: number of vertices in all elements
        '''
        if kB is None:
            kB = self.m_mat.kB
        B, V, _ = vx.size() # [B, V, C]
        M, _ = self.m_inds.size() # [M, 4]
        x = torch.index_select(vx, 1, self.m_inds.flatten()) # [B, M*4, C]
        x = x.view(B*M, -1).transpose(0, 1) # [4*C, B*M]
        x = x.view(-1, 3, B, M) # [4, C, B, M]
        X = torch.index_select(vX, 1, self.m_inds.flatten()) # [B, M*4, C]
        X = X.view(B*M, -1).transpose(0, 1) # [4*C, B*M]
        X = X.view(-1, 3, B, M) # [4, C, B, M]

        if isinstance(kB, torch.Tensor):
            k_bend = kB.view(B, 1).expand(-1, M)
        else:
            k_bend = kB

        W = Maple.DSBending_W(x, X, k_bend)

        return W.view(B, M)
