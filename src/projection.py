import torch
import numpy as np
from src.config import Config
from src.model import RadiativeGaussianModel

class AnalyticalProjector:
    @staticmethod
    def project_gaussians_chunk(
        model: RadiativeGaussianModel, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(model, "module"):
            model = model.module
                
        means = model.means
        inv_cov = model.get_inverse_covariances()
        opacities = model.get_opacities()
        
        v = rays_o.unsqueeze(1) - means.unsqueeze(0)
        
        a = torch.einsum('bj,nij,bj->bn', rays_d, inv_cov, rays_d)
        b = torch.einsum('bj,nij,bnj->bn', rays_d, inv_cov, v)
        c = torch.einsum('bni,nij,bnj->bn', v, inv_cov, v)
        
        eps = 1e-7
        sqrt_term = torch.sqrt(2.0 * np.pi / (a + eps))
        exponent = -0.5 * (c - (b**2) / (a + eps))
        exponent = torch.clamp(exponent, max=0.0)
        
        individual_projections = opacities.T * sqrt_term * torch.exp(exponent)
        pred_y = torch.sum(individual_projections, dim=-1, keepdim=True)
        
        return pred_y

    @classmethod
    def project_gaussians(
        cls, 
        model: RadiativeGaussianModel, 
        rays_o: torch.Tensor, 
        rays_d: torch.Tensor, 
        chunk_size: int = 4096
    ) -> torch.Tensor:
        all_pred_y = []
        B_total = rays_o.shape[0]
        
        for i in range(0, B_total, chunk_size):
            chunk_o = rays_o[i:i + chunk_size]
            chunk_d = rays_d[i:i + chunk_size]
            chunk_pred = cls.project_gaussians_chunk(model, chunk_o, chunk_d)
            all_pred_y.append(chunk_pred)
            
        return torch.cat(all_pred_y, dim=0)
