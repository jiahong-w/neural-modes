import torch
import numpy as np
import trimesh
from trimesh.graph import face_adjacency
from .SOMaterial import SOMaterial, EMatModel
from .CSTElement import CSTElement, CSTElementVectorize
from .DSHingeElement import DSHingeElement, DSHingeElementVectorize, Hinge

class ShellLoss:
    '''Energy inspired loss function for shell
    
    Args:
        params (dict): material parameters
        obj_faces (tensor): [F, 3] faces of the original OBJ mesh
        obj_vertices (tensor): [V, 3] vertices of the original OBJ mesh 
        vectorize (bool): whether to vectorize elements
        fixed_rest_pose (bool): whether the rest pose is fixed
    '''
    def __init__(self, params, obj_faces, obj_vertices, vectorize=True, fixed_rest_pose=True, **kwargs):
        self.device = obj_faces.device
        self.thickness = params['thickness']
        self.m_mat = SOMaterial(params['kB'],
                                params['k1'],
                                params['k2'],
                                params['k3'],
                                params['k4'],
                                params['kD'],
                                params['rho'],
                                EMatModel(params['matModel']))
        # initialize elements
        self.vectorize = vectorize
        if self.vectorize:
            self.m_cstElementsVec = self.initCSTElementsVectorize(obj_faces, obj_vertices, self.thickness, fixed_rest_pose, **kwargs)
            self.m_hingeElementsVec = self.initDSHingeElementsVectorize(obj_faces, **kwargs)
        else:
            self.m_cstElements = self.initCSTElements(obj_faces, obj_vertices, self.thickness, **kwargs)
            self.m_hingeElements = self.initDSHingeElements(obj_faces, **kwargs)
 
    @staticmethod
    def buildHingeStructure(faces) -> list[Hinge]:
        '''Build hinge structure to initialize hinge elements
           Assume no duplicate index in face and ordered mesh
           Remove boundary edges already
        
        Args:
            faces (numpy): [F, 3] input faces
        Returns:
            m_hinges (numpy): [H, 6] hinges stored as (edge0, edge1, flaps0, flaps1, tris0, tris1)
        Shape:
            F: number of faces
            H: number of hinges
        '''
        face_adj, shared_edges = face_adjacency(faces=faces, return_edges=True)
        m_hinges = -1 * np.ones((len(face_adj), 6), dtype=int)
        for i in range(len(face_adj)):
            f0_idx, f1_idx = face_adj[i, 0], face_adj[i, 1]
            f0, f1 = faces[f0_idx], faces[f1_idx]
            v0, v1 = shared_edges[i, 0], shared_edges[i, 1]
            # find vertex index in face
            # assumption: no duplicate index in face
            v0_idx_in_f0, v1_idx_in_f0 = None, None
            # meanwhile find the other triangle vertex
            v2_in_f0, v2_in_f1 = None, None
            for idx in range(len(f0)):
                if f0[idx] == v0:
                    v0_idx_in_f0 = idx
                elif f0[idx] == v1:
                    v1_idx_in_f0 = idx
                else:
                    v2_in_f0 = f0[idx]
                if f1[idx] != v0 and f1[idx] != v1:
                    v2_in_f1 = f1[idx]
            # assumption: ordered mesh
            if v0_idx_in_f0 < v1_idx_in_f0:
                m_hinges[i] = (min(v0,v1), max(v0,v1), v2_in_f0, v2_in_f1, f0_idx, f1_idx)
            else:
                m_hinges[i] = (min(v0,v1), max(v0,v1), v2_in_f1, v2_in_f0, f1_idx, f0_idx)
        return m_hinges

    def initCSTElements(self, obj_faces, obj_vertices, thickness, **kwargs):
        '''Init constant strain elements
        
        Args:
            obj_faces (tensor): [F, 3] faces of the original OBJ mesh
            obj_vertices (tensor): [V, 3] vertices of the original OBJ mesh
            thickness (float): element thickness
        Returns:
            m_cstElements (list): list of cst elements
        '''
        m_cstElements = []
        num_elems = obj_faces.shape[0]
        for i in range(num_elems):
            inds = torch.zeros(3, dtype=int, device=self.device)
            inds[:3] = obj_faces[i,:]
            elem = CSTElement(inds, self.m_mat, obj_vertices, thickness, **kwargs)
            m_cstElements.append(elem)
        return m_cstElements
    
    def initCSTElementsVectorize(self, obj_faces, obj_vertices, thickness, fixed_rest_pose, **kwargs):
        '''Init constant strain elements
        
        Args:
            obj_faces (tensor): [F, 3] faces of the original OBJ mesh
            obj_vertices (tensor): [V, 3] vertices of the original OBJ mesh
            thickness (float): element thickness
            fixed_rest_pose (bool): whether the rest pose is fixed at initialization
        Returns:
            m_cstElements (CSTElementVectorize): vectorized cst elements
        '''
        m_cstElements = CSTElementVectorize(obj_faces, self.m_mat, obj_vertices, thickness, fixed_rest_pose, **kwargs)
        return m_cstElements
    
    def initDSHingeElements(self, obj_faces, **kwargs):
        '''Init discrete hinge elements
        
        Args:
            obj_faces (tensor): [F, 3] faces of the original OBJ mesh
        Returns:
            m_hingeElements (list): list of hinge elements
        '''
        m_hingeElements = []
        m_hinges = self.buildHingeStructure(obj_faces.cpu().numpy())
        for h in m_hinges:
            inds = torch.as_tensor(h[:4], dtype=int, device=self.device)
            elem = DSHingeElement(inds, self.m_mat, **kwargs)
            m_hingeElements.append(elem)
        return m_hingeElements
    
    def initDSHingeElementsVectorize(self, obj_faces, **kwargs):
        '''Init discrete hinge elements
        
        Args:
            obj_faces (tensor): [F, 3] faces of the original OBJ mesh
        Returns:
            m_hingeElements (DSHingeElementVectorize): vectorized hinge elements
        '''
        m_hingeElements = []
        m_hinges = self.buildHingeStructure(obj_faces.cpu().numpy())
        inds = torch.as_tensor(m_hinges[:, :4], dtype=int, device=self.device)
        m_hingeElements = DSHingeElementVectorize(inds, self.m_mat, **kwargs)
        return m_hingeElements
    
    def __call__(self, x_hat, x, thickness=None, kB=None, k1=None, k2=None, return_per_element=False):
        '''Compute loss over a batch
        
        Args:
            x_hat (tensor): [B, V*C] reconstructed vertices
            x (tensor): [B, V*C] groundtruth vertices
            thickness (float): optional element thickness
            kB (tensor): [] optional material parameter
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
        Returns:
            W (tensor): [B] loss without reduction
        Shapes:
            B: batch size
            V: number of vertices
            C: 3 coordiantes
        '''
        B = x_hat.size(0)
        if self.vectorize:
            W_cst = self.m_cstElementsVec.batchComputeEnergy(x_hat.view(B, -1, 3), x.view(B, -1, 3), thickness=thickness, k1=k1, k2=k2)
            W_hinge = self.m_hingeElementsVec.batchComputeEnergy(x_hat.view(B, -1, 3), x.view(B, -1, 3), kB=kB)
            if return_per_element:
                return W_cst, W_hinge
            else:
                W = W_cst.sum(-1) + W_hinge.sum(-1)
        else:
            W = torch.zeros(B, dtype=x.dtype, device=x.device)
            for elem in self.m_cstElements:
                for i in range(B):
                    W[i] += elem.computeEnergy(x_hat[i,:].view(-1,3), x[i,:].view(-1,3))
            for elem in self.m_hingeElements:
                for i in range(B):
                    W[i] += elem.computeEnergy(x_hat[i,:].view(-1,3), x[i,:].view(-1,3))
        return W

    def stressNorm(self, x_hat, x, thickness=None, kB=None, k1=None, k2=None, return_per_element=False):
        '''Compute norm of stress tensors over a batch
        
        Args:
            x_hat (tensor): [B, V*C] reconstructed vertices
            x (tensor): [B, V*C] groundtruth vertices
            thickness (float): optional element thickness
            kB (tensor): [] optional material parameter
            k1 (tensor): [] optional material parameter
            k2 (tensor): [] optional material parameter
        Returns:
            S_norm_avg (tensor): [B] weighted average F-norm of stress tensors
            S_norm_max (tensor): [B] max of F-norm of stress tensors
            S_norm_std (tensor): [B] std of F-norm of stress tensors
        Shapes:
            B: batch size
            V: number of vertices
            C: 3 coordiantes
        '''
        B = x_hat.size(0)
        return self.m_cstElementsVec.batchComputeStressNorm(x_hat.view(B, -1, 3), x.view(B, -1, 3), thickness=thickness, k1=k1, k2=k2, return_per_element=return_per_element)
    
    def strain(self, x_hat, x):
        '''Compute norm of stress tensors over a batch
        
        Args:
            x_hat (tensor): [B, V*C] reconstructed vertices
            x (tensor): [B, V*C] groundtruth vertices
        Returns:
            S_norm_avg (tensor): [B, M, 2, 2] per element strain
        Shapes:
            B: batch size
            V: number of vertices
            C: 3 coordiantes
        '''
        B = x_hat.size(0)
        return self.m_cstElementsVec.batchComputeStrain(x_hat.view(B, -1, 3), x.view(B, -1, 3))