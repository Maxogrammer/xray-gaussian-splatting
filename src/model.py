import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.config import Config

class RadiativeGaussianModel(nn.Module):
    def __init__(self, num_gaussians: int = Config.NUM_GAUSSIANS_START):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        u1 = torch.rand(num_gaussians, dtype=Config.DTYPE)
        u2 = torch.rand(num_gaussians, dtype=Config.DTYPE)
        u3 = torch.rand(num_gaussians, dtype=Config.DTYPE)
        
        theta = 2.0 * np.pi * u1
        phi = torch.acos(2.0 * u2 - 1.0)
        r = 0.7 * torch.pow(u3, 1.0/3.0)
        
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)
        
        self.means = nn.Parameter(torch.stack([x, y, z], dim=-1))
        
        self.scales_log = nn.Parameter(
            torch.ones((num_gaussians, 3), dtype=Config.DTYPE) * -3.0
        )
        
        quats = torch.zeros((num_gaussians, 4), dtype=Config.DTYPE)
        quats[:, 0] = 1.0
        quats += torch.randn_like(quats) * 0.05
        self.quaternions_raw = nn.Parameter(quats)
        
        self.opacities_raw = nn.Parameter(
            torch.ones((num_gaussians, 1), dtype=Config.DTYPE) * -1.0
        )

    def get_opacities(self) -> torch.Tensor:
        return F.softplus(self.opacities_raw)

    def get_rotation_matrices(self) -> torch.Tensor:
        q = F.normalize(self.quaternions_raw, p=2, dim=-1)
        r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        N = self.num_gaussians
        R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)
        
        R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
        R[:, 0, 1] = 2.0 * (x * y - r * z)
        R[:, 0, 2] = 2.0 * (x * z + r * y)
        R[:, 1, 0] = 2.0 * (x * y + r * z)
        R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
        R[:, 1, 2] = 2.0 * (y * z - r * x)
        R[:, 2, 0] = 2.0 * (x * z - r * y)
        R[:, 2, 1] = 2.0 * (y * z + r * x)
        R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
        return R

    def get_inverse_covariances(self) -> torch.Tensor:
        R = self.get_rotation_matrices()
        inv_scales_sq = torch.exp(-2.0 * self.scales_log)
        R_scaled = R * inv_scales_sq.unsqueeze(1)
        inv_cov = torch.bmm(R_scaled, R.transpose(1, 2))
        return inv_cov
