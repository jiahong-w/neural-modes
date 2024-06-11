import torch
import torch.linalg as la
from .SOMaterial import SOMaterial, EMatModel
from . import Maple
import functorch

class CSTElement:
    USE_Y_MAT_DIR = False
    USE_XZ_PLANE = False

    def __init__(self, inds: torch.Tensor, mat: SOMaterial, vX: torch.Tensor, thickness: float, **kwargs):
        '''Init function

        Args:
            inds (tensor): [3] vertex indices of the element
            mat (SOMaterial): material
            vX (tensor): [V, 3] rest position
            thickness (float): element thickness
        Shape:
            V: number of vertices in all elements
        '''
        self.m_inds = inds
        self.m_mat = mat
        self.m_h = thickness
        # precompute area
        self.m_A0 = self.computeArea(vX)
        # precompute jacobian
        if self.USE_Y_MAT_DIR:
            raise NotImplementedError
        elif self.USE_XZ_PLANE:
            raise NotImplementedError
        else:
            x = torch.index_select(vX, 0, self.m_inds)
            e0 = x[1,:] - x[0,:]
            e1 = x[2,:] - x[0,:]
            b2d0 = e0 / la.vector_norm(e0)
            n = la.cross(b2d0, e1)
            b2d1 = la.cross(e0, n)
            b2d1 = b2d1 / la.vector_norm(b2d1)
        e2d00 = (e0 * b2d0).sum(-1)
        e2d10 = (e0 * b2d1).sum(-1)
        e2d01 = (e1 * b2d0).sum(-1)
        e2d11 = (e1 * b2d1).sum(-1)
        e2d = torch.stack([e2d00, e2d01, e2d10, e2d11], dim=-1).view(2, 2)
        self.m_Einv = la.inv(e2d)

    def computeArea(self, vx):
        '''Computes area

        Args:
            vx (tensor): [V, 3] vertex position
        Shape:
            V: number of vertices in all elements'''
        x = torch.index_select(vx, 0, self.m_inds)
        e0 = x[1,:] - x[0,:]
        e1 = x[2,:] - x[0,:]
        n = la.cross(e0, e1)
        return 0.5 * la.vector_norm(n)
    
    @staticmethod
    def convertEandNuToLame(E, nu):
        lame = E*nu / ((1+nu)*(1-2*nu))
        mu = E / (2 * (1 + nu))
        return lame, mu
        
    
    def computeEnergy(self, vx, vX):
        '''Computes energy

        Args:
            vx (tensor): [V, 3] current position
            vX (tensor): [V, 3] rest position
        Shape:
            V: number of vertices in all elements'''
        x = torch.index_select(vx, 0, self.m_inds)

        lame, mu = self.convertEandNuToLame(self.m_mat.k1, self.m_mat.k2)

        if self.m_mat.matModel == EMatModel.MM_StVKMod:
            W = Maple.CST3D_StVK_W(x, lame, mu, self.m_Einv, self.m_A0, self.m_h)
        else:
            raise NotImplementedError
        
        return W

class CSTElementVectorize:
    '''Vectorize CSTElement class, can compute all elements at once'''
    USE_Y_MAT_DIR = False
    USE_XZ_PLANE = False

    def __init__(self, inds: torch.Tensor, mat: SOMaterial, vX: torch.Tensor, thickness: float, fixed_rest_pose: bool, **kwargs):
        '''Init function

        Args:
            inds (tensor): [M, 3] vertex indices of the element
            mat (SOMaterial): material
            vX (tensor): [V, 3] rest position
            thickness (float): element thickness
            fixed_rest_pose (bool): whether the rest pose is fixed at initialization
        Shape:
            V: number of vertices in all elements
            M: number of elements
        '''
        self.m_inds = inds
        self.m_mat = mat
        self.m_h = thickness
        self.fixed_rest_pose = fixed_rest_pose
        if self.fixed_rest_pose:
        # precompute area & jacobian
            self.m_Einv, self.m_A0 = self.compute_Einv_and_area(vX)
        else:
            self.vmap_compute_Einv_and_area = functorch.vmap(self.compute_Einv_and_area, in_dims=0, out_dims=(2, 0))

    def compute_Einv_and_area(self, vX: torch.tensor):
        '''Compute Einv (and area)
        
        Args:
            vX (tensor): [V, 3] rest position
        Returns:
            m_Einv (tensor): [2, 2, M] inverse of jacobian
            m_A0 (tensor):[M] rest area
        Shape:
            V: number of vertices in all elements
            M: number of elements
        '''
        if self.USE_Y_MAT_DIR:
            # area
            m_A0 = self.computeArea(vX)
            raise NotImplementedError
        elif self.USE_XZ_PLANE:
            # area
            m_A0 = self.computeArea(vX)
            raise NotImplementedError
        else:
            # area
            M = self.m_inds.size(0)
            x = torch.index_select(vX, 0, self.m_inds.flatten()) # [M*3, 3]
            x = x.view(M, 3, 3)
            e0 = x[:,1,:] - x[:,0,:] # [M, 3]
            e1 = x[:,2,:] - x[:,0,:] # [M, 3]
            n = la.cross(e0, e1, dim=1) # [M, 3]
            m_A0 = 0.5 * la.vector_norm(n, dim=1)
            # jacobian (reuse computation)
            b2d0 = e0 / la.vector_norm(e0, dim=1, keepdim=True) # [M, 3]
            n = la.cross(b2d0, e1, dim=1) # [M, 3]
            b2d1 = la.cross(e0, n, dim=1) # [M, 3]
            b2d1 = b2d1 / la.vector_norm(b2d1, dim=1, keepdim=True) # [M, 3]
        e2d00 = (e0 * b2d0).sum(-1)
        e2d10 = (e0 * b2d1).sum(-1)
        e2d01 = (e1 * b2d0).sum(-1)
        e2d11 = (e1 * b2d1).sum(-1)
        e2d = torch.stack([e2d00, e2d01, e2d10, e2d11], dim=-1).view(M, 2, 2)
        Einv = la.inv(e2d) # [M, 2, 2]        
        Einv = Einv.reshape(M, 4).transpose(0, 1)
        return Einv.view(2, 2, M), m_A0

    def computeArea(self, vx):
        '''Computes area

        Args:
            vx (tensor): [V, 3] vertex position
        Returns:
            A0 (tensor): [M] rest area
        Shape:
            V: number of vertices in all elements
            M: number of elements
        '''
        M = self.m_inds.size(0)
        x = torch.index_select(vx, 0, self.m_inds.flatten()) # [M*3, 3]
        x = x.view(M, 3, 3)
        e0 = x[:,1,:] - x[:,0,:]
        e1 = x[:,2,:] - x[:,0,:]
        n = la.cross(e0, e1, dim=1) # [M, 3]
        return 0.5 * la.vector_norm(n, dim=1)
    
    @staticmethod
    def convertEandNuToLame(E, nu):
        lame = E*nu / ((1+nu)*(1-2*nu))
        mu = E / (2 * (1 + nu))
        return lame, mu
        
    def batchComputeEnergy(self, vx, vX, thickness=None, k1=None, k2=None):
        '''Computes energy in batch

        Args:
            vx (tensor): [B, V, 3] current position
            vX (tensor): [B, V, 3] rest position
            thickness (tensor): [] optional element thickness
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
        Shape:
            B: batch size
            V: number of vertices in all elements
            C: 3 coordiantes
        '''
        if thickness is None:
            thickness = self.m_h
        if k1 is None:
            k1 = self.m_mat.k1
        if k2 is None:
            k2 = self.m_mat.k2
        B, V, _ = vx.size() # [B, V, C]
        M, _ = self.m_inds.size() # [M, 3]
        x = torch.index_select(vx, 1, self.m_inds.flatten()) # [B, M*3, C]
        x = x.view(B*M, -1).transpose(0, 1) # [3*C, B*M]
        x = x.view(-1, 3, B, M) # [3, C, B, M]

        lame, mu = self.convertEandNuToLame(k1, k2)

        if self.fixed_rest_pose:
            Einv = self.m_Einv.view(2, 2, 1, M).expand(-1, -1, B, -1)
            A0 = self.m_A0.view(1, M).expand(B, -1)
        else:
            Einv, A0 = self.vmap_compute_Einv_and_area(vX)

        if self.m_mat.matModel == EMatModel.MM_StVKMod:
            W = Maple.CST3D_StVK_W(x, lame, mu, Einv, A0, thickness)
        else:
            raise NotImplementedError
        
        return W.view(B, M)
    
    def batchComputeStrain(self, vx, vX, thickness=None, k1=None, k2=None):
        '''Computes strain in batch

        Args:
            vx (tensor): [B, V, 3] current position
            vX (tensor): [B, V, 3] rest position
            thickness (tensor): [] optional element thickness
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
        Shape:
            B: batch size
            V: number of vertices in all elements
            C: 3 coordiantes
        '''
        if thickness is None:
            thickness = self.m_h
        if k1 is None:
            k1 = self.m_mat.k1
        if k2 is None:
            k2 = self.m_mat.k2
        B, V, _ = vx.size() # [B, V, C]
        M, _ = self.m_inds.size() # [M, 3]
        x = torch.index_select(vx, 1, self.m_inds.flatten()) # [B, M*3, C]
        x = x.view(B*M, -1).transpose(0, 1) # [3*C, B*M]
        x = x.view(-1, 3, B, M) # [3, C, B, M]

        if self.fixed_rest_pose:
            Einv = self.m_Einv.view(2, 2, 1, M).expand(-1, -1, B, -1)
            A0 = self.m_A0.view(1, M).expand(B, -1)
        else:
            Einv, A0 = self.vmap_compute_Einv_and_area(vX)

        E = Maple.CST3D_E(x, Einv) # [2, 2, B, M]
        E = torch.permute(E, (2, 3, 0, 1)) # [B, M, 2, 2]
        
        return E
    
    def batchComputeStressNorm(self, vx, vX, thickness=None, k1=None, k2=None, return_per_element=False):
        '''Computes weighted average and max of stress in batch
            Frobenius norm of the second Piola-Kirchhoff stress tensor

        Args:
            vx (tensor): [B, V, 3] current position
            vX (tensor): [B, V, 3] rest position
            thickness (tensor): [] optional element thickness
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
        Shape:
            B: batch size
            V: number of vertices in all elements
            C: 3 coordiantes
        '''
        if thickness is None:
            thickness = self.m_h
        if k1 is None:
            k1 = self.m_mat.k1
        if k2 is None:
            k2 = self.m_mat.k2
        B, V, _ = vx.size() # [B, V, C]
        M, _ = self.m_inds.size() # [M, 3]
        x = torch.index_select(vx, 1, self.m_inds.flatten()) # [B, M*3, C]
        x = x.view(B*M, -1).transpose(0, 1) # [3*C, B*M]
        x = x.view(-1, 3, B, M) # [3, C, B, M]

        lame, mu = self.convertEandNuToLame(k1, k2)

        if self.fixed_rest_pose:
            Einv = self.m_Einv.view(2, 2, 1, M).expand(-1, -1, B, -1)
            A0 = self.m_A0.view(1, M).expand(B, -1)
        else:
            Einv, A0 = self.vmap_compute_Einv_and_area(vX)

        if self.m_mat.matModel == EMatModel.MM_StVKMod:
            S_norm = Maple.CST3D_StVK_S_norm(x, lame, mu, Einv)
            S_norm = S_norm.view(B, M)
            S_norm_avg = torch.sum(S_norm * A0, dim=1) / torch.sum(A0, dim=1)
            S_norm_max = S_norm.max(dim=1)[0]
            S_norm_std = S_norm.std(dim=1)
        else:
            raise NotImplementedError
        
        if return_per_element:
            return S_norm
        else:
            return S_norm_avg, S_norm_max, S_norm_std
