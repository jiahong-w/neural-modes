import argparse
import os, sys, logging
import numpy as np
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
        self.latent_dim = 3
        assert self.latent_dim == cfg['model']['decoder_kwargs']['latent_dim']
        self.z = torch.zeros((self.latent_dim), device=self.device)
        self.z_init = torch.tensor([-13, 49.5, -45], device=self.device)
        self.z_rigid = torch.zeros(6, device=self.device)
        self.z_rigid_init = torch.zeros(6, device=self.device)
        
        # GUI control
        self.run_sim =False
        self.smooth_shade = True

        # Fix boundary vertices
        tolerance = 0.02
        min_z = self.X.view(-1,3).min(0).values[2]
        self.fixed_vertex_mask = self.X.view(-1,3)[:,2] <= min_z + tolerance
        self.boundary_stiffness = 1e10

        # Create energy function
        params = dataset.get_params()
        obj_faces = dataset.get_obj_faces().to(device)
        obj_vertices = dataset.get_obj_vertices().to(device)
        self.stvk_func = config.get_energy(cfg, params=params, obj_faces=obj_faces, obj_vertices=obj_vertices, fixed_rest_pose=True, dtype=torch.get_default_dtype(), log=False)

        self.M = self.stvk_func.getMass().repeat_interleave(3)

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
        linear_modes = self.linear_modes[:,:self.latent_dim]
        y = self.model.decode(self.z.view(1,-1))
        l = self.X.view(1,-1) + (linear_modes @ self.z).view(1, -1)
        x = y + l
        self.x = x.flatten().detach().clone()
        self.prev_x = x.flatten().detach().clone()
        self.prev_prev_x = x.flatten().detach().clone()
        self.z_rigid = self.z_rigid_init.detach().clone()
    
    def step(self):
        # run simulation
        self.model.eval()
        linear_modes = self.linear_modes[:,:self.latent_dim]
        dt = 5e-3
        self.t += dt
        self.z.requires_grad = True
        self.z_rigid.requires_grad = True
        optimizer = torch.optim.LBFGS([self.z], lr=1, max_iter=10000, tolerance_change=1.0e-9)
        def closure():
            optimizer.zero_grad()
            y = self.model(self.z.view(1,-1)).flatten()
            l = self.X + (linear_modes @ self.z)
            x = y + l
            stvk = self.stvk_func(x.view(1,-1), self.X.view(1,-1)).sum()
            a = (x - 2 * self.prev_x + self.prev_prev_x)
            inertia = 0.5 / (dt * dt) * (a * self.M * a).sum()
            loss = stvk + inertia
            loss_scalar = loss
            loss_scalar.backward()
            return loss_scalar
        optimizer.step(closure)
        with torch.no_grad():
            y = self.model(self.z.view(1,-1))
            l = self.X.view(1,-1) + (linear_modes @ self.z).view(1, -1)
            x = y + l
            self.prev_prev_x = self.prev_x.detach().clone()
            self.prev_x = x.flatten().detach().clone()
    
    def update(self):
        ps_elems = ps.register_surface_mesh('mesh', self.prev_x.detach().cpu().numpy().reshape(-1,3), self.faces.numpy(), smooth_shade=self.smooth_shade)
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

    def GUI_loop():
        nonlocal engine
        update = False

        psim.TextUnformatted('Current: Neural Modes')
        psim.Separator()

        if psim.Button('Reset'):
            engine.reset()
            update |= True

        if engine.run_sim:
            engine.step()
            update = True

        changed, engine.run_sim = psim.Checkbox("Run simulation", engine.run_sim)

        changed, engine.smooth_shade = psim.Checkbox("Use smooth shading", engine.smooth_shade)
        update |= changed

        if update:
            engine.update()

    ps.set_user_callback(GUI_loop)
    ps.set_up_dir("z_up")
    ps.look_at((11,-11,1),(0,0,1))
    ps.show()

if __name__ == '__main__':
    main()