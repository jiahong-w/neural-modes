import argparse
import os, sys, logging
import numpy as np
import trimesh
from common import Checkpoint, config, data
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

import polyscope as ps
import polyscope.imgui as psim

logger = logging.getLogger('sim')

def setup_logger(name, log_dir=None, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(asctime)s] %(name)s %(levelname)s: %(message)s', datefmt='%m/%d %H:%M:%S'
    )

    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'log_rank{distributed_rank}.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

class Routine:
    def __init__(self, cfg):
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')
        dataset_class = cfg['data']['dataset_class']
        dataset = getattr(data, dataset_class)(cfg, split=None, preload=False)
        
        # Load checkpoint
        self.dim = dataset.get_model_input_dim()
        self.model = config.get_model(cfg, device, dim=self.dim) 
        cp_dir = cfg['testing'].get('checkpoint_dir', None)
        cp_dir = cfg['training']['checkpoint_dir'] if cp_dir is None else cp_dir
        self.cp_dir = cp_dir
        cp = Checkpoint(cp_dir)
        best_checkpoint = cfg['testing'].get('best_checkpoint', None)
        assert best_checkpoint is not None, 'does not specify the checkpoint file'
        cp.load(best_checkpoint, device, model=self.model)
        for p in self.model.parameters():
            p.requires_grad = False
        
        # Load mesh
        self.faces = dataset.get_vis_faces()
        self.X = dataset.get_X()
        self.device = device
        self.X = self.X.to(device)
        
        # Initialize latent variable
        self.latent_dim = 5
        assert self.latent_dim == cfg['model']['decoder_kwargs']['latent_dim']
        self.z = torch.zeros((self.latent_dim), device=self.device)
        self.z_init = torch.zeros((self.latent_dim), device=self.device)
        self.z_rigid = torch.zeros(6, device=self.device)
        self.z_rigid_init = torch.zeros(6, device=self.device)
        self.more_basis_dim = 15
        self.z_more_basis_init = torch.zeros((self.more_basis_dim), device=self.device)
        self.z_more_basis = torch.zeros((self.more_basis_dim), device=self.device)
        
        # GUI control
        self.run_sim =False
        self.use_linear_mode = True
        self.use_more_basis = True
        self.smooth_shade = True

        # Create energy function
        params = dataset.get_params()
        obj_faces = dataset.get_obj_faces().to(device)
        obj_vertices = dataset.get_obj_vertices().to(device)
        self.stvk_func = config.get_energy(cfg, params=params, obj_faces=obj_faces, obj_vertices=obj_vertices, fixed_rest_pose=True, dtype=torch.get_default_dtype(), log=False)

        self.M = self.stvk_func.getMass().repeat_interleave(3)
        self.f_gravity = torch.zeros(self.dim, device=self.device)
        self.f_gravity[2::3] = -9.8 * self.stvk_func.getMass()

        # Load linear eigenmodes
        linear_modes_file = os.path.join(cfg['data']['path'],'linear_modes.pt')
        if os.path.isfile(linear_modes_file):
            self.linear_modes = torch.load(linear_modes_file, map_location=self.device)
        else:
            logger.error(f'Linear modes file {linear_modes_file} does not exist')
            raise FileExistsError

        self.reset()
    
    @torch.no_grad()    
    def reset(self):
        self.t = 0.0
        self.z = self.z_init.detach().clone()
        self.z_more_basis = self.z_more_basis_init.detach().clone()
        linear_modes = self.linear_modes[:,6:6+self.latent_dim]
        linear_modes_more = self.linear_modes[:,6:6+self.more_basis_dim]
        if self.use_linear_mode:
            if self.use_more_basis:
                l = self.X.view(1,-1) + (linear_modes_more @ self.z_more_basis).view(1, -1)
                x = l
                self.x = x.flatten().detach().clone()
                self.prev_x = x.flatten().detach().clone()
                self.prev_prev_x = x.flatten().detach().clone()
            else:   
                l = self.X.view(1,-1) + (linear_modes @ self.z).view(1, -1)
                x = l
                self.x = x.flatten().detach().clone()
                self.prev_x = x.flatten().detach().clone()
                self.prev_prev_x = x.flatten().detach().clone()
        else:
            y = self.model.decode(self.z.view(1,-1))
            l = self.X.view(1,-1) + (linear_modes @ self.z).view(1, -1)
            x = y + l
            self.x = x.flatten().detach().clone()
            self.prev_x = x.flatten().detach().clone()
            self.prev_prev_x = x.flatten().detach().clone()

        self.z_rigid = self.z_rigid_init.detach().clone()
        # add strings
        self.string_vdix = [2370, 1545, 1319, 1774, 541, 307] # shell, left arm, right arm, head, left foot, right foot
        string_vertices = self.x.view(-1, 3)[self.string_vdix, :]
        self.string_endpoints = string_vertices.detach().clone()
        self.string_endpoints[:, 2] = 1.45
        self.string_init_lengths = 1.45 - string_vertices[:, 2]
        self.string_lengths = self.string_init_lengths
    
    def step(self):
        # physical simulation
        self.model.eval()
        linear_modes = self.linear_modes[:,6:6+self.latent_dim]
        rigid_linear_modes = self.linear_modes[:,:6]
        linear_modes_more = self.linear_modes[:,6:6+self.more_basis_dim]
        dt = 1e-1
        self.t += dt
        self.string_lengths[1] = self.string_init_lengths[1] + 0.01 * np.sin(2*self.t)
        self.string_lengths[2] = self.string_init_lengths[2] + 0.01 * np.sin(self.t)
        if self.use_linear_mode:
            if self.use_more_basis:
                self.z_more_basis.requires_grad = True
                self.z_rigid.requires_grad = True
                optimizer = torch.optim.LBFGS([self.z_more_basis, self.z_rigid], lr=1, max_iter=10000, tolerance_change=1.0e-2)
                def closure():
                    optimizer.zero_grad()
                    l = self.X + (linear_modes_more @ self.z_more_basis) + (rigid_linear_modes @ self.z_rigid)
                    x = l
                    stvk = self.stvk_func(x.view(1,-1), self.X.view(1,-1)).sum()
                    a = (x - 2 * self.prev_x + self.prev_prev_x)
                    inertia = 0.5 / (dt * dt) * (a * self.M * a).sum()
                    gravity = - (self.f_gravity * x).sum()
                    string_v = x.view(-1,3)[self.string_vdix, :]
                    string_len = (string_v - self.string_endpoints).square().sum(1).sqrt()
                    penalty_string = 1.0e11 * (string_len - self.string_lengths).square().sum()
                    loss = stvk + inertia + gravity + penalty_string
                    loss_scalar = loss
                    loss_scalar.backward()
                    return loss_scalar
                optimizer.step(closure)
                with torch.no_grad():
                    l = self.X.view(1,-1) + (linear_modes_more @ self.z_more_basis).view(1, -1) + (rigid_linear_modes @ self.z_rigid).view(1,-1)
                    x = l
                    self.prev_prev_x = self.prev_x.detach().clone()
                    self.prev_x = x.flatten().detach().clone()
            else:
                self.z.requires_grad = True
                self.z_rigid.requires_grad = True
                optimizer = torch.optim.LBFGS([self.z, self.z_rigid], lr=1, max_iter=10000, tolerance_change=1.0e-2)
                def closure():
                    optimizer.zero_grad()
                    l = self.X + (linear_modes @ self.z) + (rigid_linear_modes @ self.z_rigid)
                    x = l
                    stvk = self.stvk_func(x.view(1,-1), self.X.view(1,-1)).sum()
                    a = (x - 2 * self.prev_x + self.prev_prev_x)
                    inertia = 0.5 / (dt * dt) * (a * self.M * a).sum()
                    gravity = - (self.f_gravity * x).sum()
                    string_v = x.view(-1,3)[self.string_vdix, :]
                    string_len = (string_v - self.string_endpoints).square().sum(1).sqrt()
                    penalty_string = 1.0e12 * (string_len - self.string_lengths).square().sum()
                    loss = stvk + inertia + gravity + penalty_string
                    loss_scalar = loss
                    loss_scalar.backward()
                    return loss_scalar
                optimizer.step(closure)
                with torch.no_grad():
                    l = self.X.view(1,-1) + (linear_modes @ self.z).view(1, -1) + (rigid_linear_modes @ self.z_rigid).view(1,-1)
                    x = l
                    self.prev_prev_x = self.prev_x.detach().clone()
                    self.prev_x = x.flatten().detach().clone()
        else:
            self.z.requires_grad = True
            self.z_rigid.requires_grad = True
            optimizer = torch.optim.LBFGS([self.z, self.z_rigid], lr=1, max_iter=10000, tolerance_change=1.0e-2)
            def closure():
                optimizer.zero_grad()
                y = self.model(self.z.view(1,-1)).flatten()
                l = self.X + (linear_modes @ self.z) + (rigid_linear_modes @ self.z_rigid)
                x = y + l
                stvk = self.stvk_func(x.view(1,-1), self.X.view(1,-1)).sum()
                a = (x - 2 * self.prev_x + self.prev_prev_x)
                inertia = 0.5 / (dt * dt) * (a * self.M * a).sum()
                gravity = - (self.f_gravity * x).sum()
                string_v = x.view(-1,3)[self.string_vdix, :]
                string_len = (string_v - self.string_endpoints).square().sum(1).sqrt()
                penalty_string = 1.0e11 * (string_len - self.string_lengths).square().sum()
                loss = stvk + inertia + gravity + penalty_string
                loss_scalar = loss
                loss_scalar.backward()
                return loss_scalar
            optimizer.step(closure)
            with torch.no_grad():
                y = self.model(self.z.view(1,-1))
                l = self.X.view(1,-1) + (linear_modes @ self.z).view(1, -1) + (rigid_linear_modes @ self.z_rigid).view(1,-1)
                x = y + l
                self.prev_prev_x = self.prev_x.detach().clone()
                self.prev_x = x.flatten().detach().clone()
        
    @torch.no_grad()
    def get_string_vertices(self):
        string_vertices = self.prev_x.view(-1, 3)[self.string_vdix, :].detach()
        string_vertices = torch.cat([self.string_endpoints, string_vertices], dim=0)
        string_vertices = string_vertices.cpu().numpy()
        return string_vertices
    
    @torch.no_grad()
    def get_string_lines(self):
        string_lines = np.zeros((len(self.string_vdix), 2))
        string_lines[:, 0] = np.arange(len(self.string_vdix))
        string_lines[:, 1] = np.arange(len(self.string_vdix)) + len(self.string_vdix)
        return string_lines
    
    def update(self):
        ps_elems = ps.register_surface_mesh('mesh', self.prev_x.detach().cpu().numpy().reshape(-1,3), self.faces.numpy(), smooth_shade=self.smooth_shade, color=(0.11, 0.388, 0.890))
        ps_curve = ps.register_curve_network('string', self.get_string_vertices(), self.get_string_lines(), color=(0.0, 0.3, 0), radius=0.002)
        return

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='config file path')
    args = parser.parse_args()
    cfg = config.load_config(args.config)
    setup_logger(None, log_dir=os.path.join(cfg['training']['checkpoint_dir'], cfg['training']['log_dir']))

    engine = Routine(cfg)

    # Initialize polyscope
    ps.init()
    ps.set_ground_plane_mode('none')
    ps.set_automatically_compute_scene_extents(False)

    engine.update()
    
    # Draw spheres
    for i in range(6):
        sphere_mesh = trimesh.load(os.path.join(cfg['data']['path'], f'sphere_{i}.obj'))
        ps.register_surface_mesh(f'sphere_{i}', sphere_mesh.vertices, sphere_mesh.faces, color=(0.11, 0.388, 0.890))

    def GUI_loop():
        nonlocal engine
        update = False

        if engine.use_linear_mode:
            if engine.use_more_basis:
                psim.TextUnformatted('Current: Linear modes -- extra basis')
            else:
                psim.TextUnformatted('Current: Linear modes')
        else:
            psim.TextUnformatted('Current: Neural Modes')
        psim.Separator()

        if engine.run_sim:
            engine.step()
            update = True

        changed, engine.run_sim = psim.Checkbox("Run simulation", engine.run_sim)

        changed, engine.use_linear_mode = psim.Checkbox("Use linear modes", engine.use_linear_mode)
        if changed:
            engine.reset()
        update |= changed

        if engine.use_linear_mode:
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
            if(psim.TreeNode('Options')):
                changed, engine.use_more_basis = psim.Checkbox("Extra linear basis", engine.use_more_basis)
                if changed:
                    engine.reset()
                update |= changed
                psim.TreePop()

        changed, engine.smooth_shade = psim.Checkbox("Use smooth shading", engine.smooth_shade)
        update |= changed

        if update:
            engine.update()

    ps.set_user_callback(GUI_loop)
    ps.set_up_dir("z_up")
    ps.set_front_dir("neg_y_front")
    ps.set_length_scale(2.8)
    ps.set_ground_plane_mode("shadow_only")
    ps.show()

if __name__ == '__main__':
    main()