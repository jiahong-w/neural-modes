import argparse
import os, sys, logging
import numpy as np
from common import Checkpoint, config, data
import torch
torch.manual_seed(0)
import torch.autograd.functional as AF
torch.set_default_dtype(torch.float64)
from torch.utils.tensorboard import SummaryWriter

import polyscope as ps
import polyscope.imgui as psim

logger = logging.getLogger('train')

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
        assert cfg['model']['decoder_kwargs']['latent_dim'] == 3
        self.cfg = cfg
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda') if use_cuda else torch.device('cpu')

        # Load mesh and create model
        dataset_class = cfg['data']['dataset_class']
        dataset = getattr(data, dataset_class)(cfg, split=None, preload=False)
        self.model = config.get_model(cfg, device, dim=dataset.get_model_input_dim())
        self.faces = dataset.get_vis_faces()
        self.X = dataset.get_X()
        self.device = device
        
        # Default state
        self.state = {
            'z_0': 0,
            'z_1': 0,
            'z_2': 0,
        }

        # Create energy function
        params = dataset.get_params()
        obj_faces = dataset.get_obj_faces().to(device)
        obj_vertices = dataset.get_obj_vertices().to(device)
        self.stvk_func = config.get_energy(cfg, params=params, obj_faces=obj_faces, obj_vertices=obj_vertices, dtype=torch.get_default_dtype(), log=False)
        self.X = self.X.to(self.device)
        
        # Fix boundary vertices
        tolerance = 0.02
        min_z = self.X.view(-1,3).min(0).values[2]
        self.fixed_vertex_mask = self.X.view(-1,3)[:,2] <= min_z + tolerance
        self.boundary_stiffness = 1e10
        
        # Compute linear eigenmodes
        linear_modes_file = os.path.join(cfg['data']['path'],'linear_modes.pt')
        hess_file = os.path.join(cfg['data']['path'],'hess.pt')
        if os.path.isfile(linear_modes_file):
            self.linear_modes = torch.load(linear_modes_file, map_location=self.device)
        else:
            if os.path.isfile(hess_file):
                hess = torch.load(hess_file, map_location=self.device)
            else:
                stvk_func_single = lambda x: self.stvk_func(x.view(1, -1), self.X.view(1, -1)).sum() + 0.5*self.boundary_stiffness*(x.view(-1, 3)[self.fixed_vertex_mask,:]-self.X.view(-1, 3)[self.fixed_vertex_mask,:]).square().sum()
                hess = AF.hessian(stvk_func_single, self.X.detach().clone(), strict=True)
                torch.save(hess, hess_file)
                logger.info(f'save hessian to {hess_file}')
            L, Q = torch.linalg.eigh(hess)
            self.linear_modes = Q

            # Save linear modes
            torch.save(self.linear_modes, linear_modes_file)
            logger.info(f'save linear modes to {linear_modes_file}')

        # GUI control
        self.use_linear_mode = True
        self.smooth_shade = True

        # Training
        self.optimizer = torch.optim.LBFGS(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1, max_iter=20, line_search_fn='strong_wolfe')

        # Checkpoint
        cp_dir = cfg['training']['checkpoint_dir']
        self.cp_dir = cp_dir
        self.cp = Checkpoint(cp_dir)
        saved_checkpoint = cfg['training'].get('saved_checkpoint', None)
        if saved_checkpoint is not None:
            self.cp.load(saved_checkpoint, device, model=self.model, optimizer=self.optimizer)

        # Tensorboard
        self.writer = SummaryWriter(os.path.join(self.cp_dir, cfg['training']['tensorboard_dir']), purge_step=self.cp.iter+1)

        self.cp.step()
        
    def train(self):
        print_every = self.cfg['training']['print_every']
        checkpoint_every = self.cfg['training']['checkpoint_every']
        batch_size = self.cfg['training']['batch_size']
        rest_idx = 0

        L = 3
        X = self.X.view(1, -1).expand(batch_size, -1)
        linear_modes = self.linear_modes[:,:L]
        
        # Iteratively minimize energy    
        while True:
            with torch.no_grad():
                z = torch.rand(batch_size, L, device=self.device) * 200 - 100
                z[rest_idx,:] = 0
                l = X + (z @ linear_modes.T) 
                constraint_dir = (z @ linear_modes.T)
                constraint_dir = constraint_dir / torch.norm(constraint_dir, p=2, dim=-1, keepdim=True) 
                constraint_dir[rest_idx] = 0
            
            self.model.train()
            def closure():
                self.optimizer.zero_grad()
                y = self.model(z)
                origin = y[rest_idx,:].square().sum()
                stvk = self.stvk_func(y + l, X).mean()
                boundary = 0.5*self.boundary_stiffness*(y.view(batch_size, -1, 3)[:,self.fixed_vertex_mask,:]).square().sum(-1).sum(-1).mean()
                ortho = (y * constraint_dir).sum(1).square().mean()
                loss = stvk + boundary + 1.0e8 * ortho + 1.0e7 * origin
                loss_scalar = loss
                loss_scalar.backward()
                return loss_scalar
            self.optimizer.step(closure)

            # Record metrics
            with torch.no_grad():
                y = self.model(z)
                origin = y[rest_idx,:].square().sum()
                stvk = self.stvk_func(y + l, X).mean()
                boundary = 0.5*self.boundary_stiffness*(y.view(batch_size, -1, 3)[:,self.fixed_vertex_mask,:]).square().sum(-1).sum(-1).mean()
                ortho = (y * constraint_dir).sum(1).square().mean()
                loss = stvk + boundary + 1.0e8 * ortho + 1.0e7 * origin
            
            self.writer.add_scalar('train/loss', loss, self.cp.iter)
            self.writer.add_scalar('train/stvk', stvk, self.cp.iter)

            if (self.cp.iter+1) % print_every == 0:
                logger.info(f'[iter {self.cp.iter}] loss={loss.item()} stvk={stvk.item()} bnd={boundary.item()} ortho={ortho.item()} origin={origin.item()}')         
            
            if (self.cp.iter+1) % checkpoint_every == 0:
                self.cp.save(f'model_it{self.cp.iter}.pt', model=self.model, optimizer=self.optimizer)
            
            self.cp.step()
    
    def update(self):
        L = 3
        z = torch.zeros(L, device=self.device)
        z[0] = self.state['z_0']
        z[1] = self.state['z_1']
        z[2] = self.state['z_2']
        X = self.X.view(1, -1)
        linear_modes = self.linear_modes[:, :L]
        l = X + (linear_modes @ z).view(1, -1)
        if self.use_linear_mode:
            x = l.detach().cpu().numpy()
        else:
            self.model.eval()
            with torch.no_grad():
                y = self.model(z.view(1,-1))
                x = (y + l).detach().cpu().numpy()
        
        ps_elems = ps.register_surface_mesh('mesh', x.reshape(-1,3), self.faces.numpy(), smooth_shade=self.smooth_shade)
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

        if engine.use_linear_mode:
            psim.TextUnformatted('Current: Linear modes')
        else:
            psim.TextUnformatted('Current: Neural Modes')
        psim.Separator()

        update = False

        psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)
        if(psim.TreeNode('Modal coordinate control')):
            for i in range(3):
                changed, engine.state[f'z_{i}'] = psim.SliderFloat(f'Coord {i} (z{i})', engine.state[f'z_{i}'], v_min=-100, v_max=100)
                update |= changed
            psim.TreePop()


        changed, engine.use_linear_mode = psim.Checkbox("Use linear modes", engine.use_linear_mode)
        update |= changed

        changed, engine.smooth_shade = psim.Checkbox("Use smooth shading", engine.smooth_shade)
        update |= changed

        if psim.Button('Start training'):
            engine.train()

        if update:
            engine.update()

    ps.set_user_callback(GUI_loop)
    ps.set_up_dir("z_up")
    ps.look_at((11,-11,1),(0,0,1))
    ps.show()

if __name__ == '__main__':
    main()