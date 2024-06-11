import torch
import torch.linalg as la
from .SOMaterial import SOMaterial, EMatModel
from . import Maple

class LinearTetElementVectorize:
    '''Vectorize LinearTetElement class, can compute all elements at once'''

    def __init__(self, inds: torch.Tensor, mat: SOMaterial, vX: torch.Tensor, **kwargs):
        '''Init function

        Args:
            inds (tensor): [M, 4] vertex indices of the element
            mat (SOMaterial): material
            vX (tensor): [V, 3] rest position
        Shape:
            V: number of vertices in all elements
            M: number of elements
        '''
        self.m_inds = inds
        self.m_mat = mat
    
    @staticmethod
    def convertEandNuToLame(E, nu):
        lame = E*nu / ((1+nu)*(1-2*nu))
        mu = E / (2 * (1 + nu))
        return lame, mu
    
    def computeVolume(self, vx):
        '''Computes volume

        Args:
            vx (tensor): [V, 3] vertex position
        Returns:
            vol (tensor): [M] volume
        Shape:
            V: number of vertices in all elements
            M: number of elements
        '''
        M = self.m_inds.size(0)
        x = torch.index_select(vx, 0, self.m_inds.flatten()) # [M*4, 3]
        x = x.view(M, 4, 3)
        e0 = x[:,1,:] - x[:,0,:]
        e1 = x[:,2,:] - x[:,0,:]
        e2 = x[:,3,:] - x[:,0,:]
        n = la.cross(e0, e1, dim=1) # [M, 3]
        vol = la.vecdot(n, e2, dim=1).abs() / 6
        return vol
    
    def computeMass(self, vx):
        '''Computes lumped mass matrix (diagonal)

        Args:
            vx (tensor): [V, 3] vertex position
        Returns:
            mass (tensor): [V] volume
        Shape:
            V: number of vertices in all elements
        '''
        vol = self.computeVolume(vx)
        mass = torch.zeros(vx.size(0), dtype=vx.dtype, device=vx.device)
        mass = mass.index_add_(0, self.m_inds.flatten(), torch.repeat_interleave(vol, repeats=4, dim=0))
        return mass * self.m_mat.rho * 0.25
        
    def batchComputeEnergy(self, vx, vX, k1=None, k2=None, restScale=None):
        '''Computes energy in batch

        Args:
            vx (tensor): [B, V, 3] current position
            vX (tensor): [B, V, 3] rest position
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
            restScale (tensor): [] optional scaling factor of the rest shape
        Shape:
            B: batch size
            V: number of vertices in all elements
            C: 3 coordiantes
        '''
        if k1 is None:
            k1 = self.m_mat.k1
        if k2 is None:
            k2 = self.m_mat.k2
        if restScale is None:
            restScale = 1
        B, V, _ = vx.size() # [B, V, C]
        M, _ = self.m_inds.size() # [M, 4]
        x = torch.index_select(vx, 1, self.m_inds.flatten()) # [B, M*4, C]
        x = x.view(B*M, -1).transpose(0, 1) # [4*C, B*M]
        x = x.view(-1, 3, B, M) # [4, C, B, M]
        X = torch.index_select(vX, 1, self.m_inds.flatten()) # [B, M*4, C]
        X = X.view(B*M, -1).transpose(0, 1) # [4*C, B*M]
        X = X.view(-1, 3, B, M) # [4, C, B, M]

        lame, mu = self.convertEandNuToLame(k1, k2)

        if self.m_mat.matModel == EMatModel.MM_StVKMod:
            W = Maple.LinearTet_StVK_W(x, X, lame, mu, restScale)
        elif self.m_mat.matModel == EMatModel.MM_NH:
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        return W.view(B, M)