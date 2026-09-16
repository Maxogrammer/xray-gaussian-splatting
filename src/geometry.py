import torch
import torch.nn.functional as F
import numpy as np
from src.config import Config

class PhantomGenerator:
    @staticmethod
    def generate_3d_phantom(res: int = Config.VOXEL_RES) -> torch.Tensor:

        grid_range = torch.linspace(-1.0, 1.0, res)
        z, y, x = torch.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
        
        coords = torch.stack([x, y, z], dim=-1).to(Config.DEVICE)
        r_sq = torch.sum(coords**2, dim=-1)
        
        phantom = torch.zeros((res, res, res), device=Config.DEVICE, dtype=Config.DTYPE)
        
        phantom[r_sq <= 0.85**2] = 0.8
        phantom[r_sq <= 0.80**2] = 0.3
        
        ellipsoid_mask = (coords[..., 0]/0.25)**2 + (coords[..., 1]/0.25)**2 + (coords[..., 2]/0.45)**2 <= 1.0
        phantom[ellipsoid_mask] = 1.0
        
        air_cavity_1 = torch.sum((coords - torch.tensor([0.4, 0.4, 0.0], device=Config.DEVICE))**2, dim=-1) <= 0.15**2
        air_cavity_2 = torch.sum((coords - torch.tensor([-0.4, -0.4, 0.2], device=Config.DEVICE))**2, dim=-1) <= 0.12**2
        phantom[air_cavity_1] = 0.0
        phantom[air_cavity_2] = 0.0
        
        return phantom.unsqueeze(0).unsqueeze(0)


class RayGenerator:
    @staticmethod
    def generate_all_rays() -> tuple[torch.Tensor, torch.Tensor]:
        M = Config.NUM_VIEWS
        N_det = Config.DETECTOR_RES
        
        angles = torch.linspace(0.0, np.pi, M, device=Config.DEVICE, dtype=Config.DTYPE)
        
        det_coords = torch.linspace(-Config.DETECTOR_SIZE/2, Config.DETECTOR_SIZE/2, N_det, device=Config.DEVICE, dtype=Config.DTYPE)
        u_grid, v_grid = torch.meshgrid(det_coords, det_coords, indexing='ij')
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        
        all_rays_o = []
        all_rays_d = []
        
        for theta in angles:
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            
            d_theta = torch.tensor([cos_t, sin_t, 0.0], device=Config.DEVICE, dtype=Config.DTYPE)
            u_theta = torch.tensor([-sin_t, cos_t, 0.0], device=Config.DEVICE, dtype=Config.DTYPE)
            v_theta = torch.tensor([0.0, 0.0, 1.0], device=Config.DEVICE, dtype=Config.DTYPE)
            
            offsets = u_flat.unsqueeze(1) * u_theta + v_flat.unsqueeze(1) * v_theta
            origins = offsets - Config.R_ORBIT * d_theta
            
            directions = d_theta.unsqueeze(0).expand(origins.shape[0], -1)
            
            all_rays_o.append(origins)
            all_rays_d.append(directions)
            
        return torch.cat(all_rays_o, dim=0), torch.cat(all_rays_d, dim=0)


class CTProjector:
    @staticmethod
    def project_volume(volume: torch.Tensor, rays_o: torch.Tensor, rays_d: torch.Tensor) -> torch.Tensor:
        B = rays_o.shape[0]
        S = Config.NUM_SAMPLES_RAY
        
        t_samples = torch.linspace(0.5, 3.5, S, device=Config.DEVICE, dtype=Config.DTYPE)
        
        points = rays_o.unsqueeze(1) + t_samples.unsqueeze(0).unsqueeze(-1) * rays_d.unsqueeze(1)
        
        grid_coords = points.unsqueeze(0).unsqueeze(3)
        
        sampled_densities = F.grid_sample(
            volume, 
            grid_coords, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        sampled_densities = sampled_densities.squeeze(0).squeeze(0).squeeze(-1)
        
        projections = torch.sum(sampled_densities, dim=-1, keepdim=True) * Config.STEP_SIZE
        
        return projections
