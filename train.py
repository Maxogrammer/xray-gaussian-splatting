import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src.config import Config
from src.geometry import PhantomGenerator, RayGenerator, CTProjector
from src.model import RadiativeGaussianModel
from src.projection import AnalyticalProjector

class RayDataset(Dataset):
    def __init__(self, rays_o: torch.Tensor, rays_d: torch.Tensor, target_y: torch.Tensor):
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.target_y = target_y

    def __len__(self):
        return self.rays_o.shape[0]

    def __getitem__(self, idx):
        return self.rays_o[idx], self.rays_d[idx], self.target_y[idx]


def perform_densification_and_pruning(model, optimizer, grad_norms_accum, step_count):
    with torch.no_grad():
        avg_grads = grad_norms_accum / max(1, step_count)
        
        tau_grad = Config.TAU_GRAD
        tau_opacity = Config.TAU_OPACITY
        tau_scale = Config.TAU_SCALE
        
        scales = torch.exp(model.scales_log)
        opacities = F.softplus(model.opacities_raw)
        
        prune_mask = (opacities.squeeze(-1) < tau_opacity)
        high_grad_mask = (avg_grads > tau_grad)
        
        clone_mask = high_grad_mask & (torch.max(scales, dim=-1)[0] < tau_scale) & (~prune_mask)
        split_mask = high_grad_mask & (torch.max(scales, dim=-1)[0] >= tau_scale) & (~prune_mask)
        
        new_means_clone = model.means[clone_mask]
        new_scales_clone = model.scales_log[clone_mask]
        new_quats_clone = model.quaternions_raw[clone_mask]
        new_opacities_clone = model.opacities_raw[clone_mask]
        
        means_to_split = model.means[split_mask]
        scales_to_split = model.scales_log[split_mask]
        quats_to_split = model.quaternions_raw[split_mask]
        opacities_to_split = model.opacities_raw[split_mask]
        
        split_scales_new = scales_to_split - np.log(1.6)
        R = model.get_rotation_matrices()[split_mask]
        shift_dir = R[:, :, 0] * torch.exp(scales_to_split[:, 0:1])
        
        means_split_1 = means_to_split + shift_dir * 0.5
        means_split_2 = means_to_split - shift_dir * 0.5
        
        added_means = torch.cat([new_means_clone, means_split_1, means_split_2], dim=0)
        added_scales = torch.cat([new_scales_clone, split_scales_new, split_scales_new], dim=0)
        added_quats = torch.cat([new_quats_clone, quats_to_split, quats_to_split], dim=0)
        added_opacities = torch.cat([new_opacities_clone, opacities_to_split, opacities_to_split], dim=0)
        
        keep_mask = ~prune_mask
        
        final_means = torch.cat([model.means[keep_mask], added_means], dim=0)
        final_scales = torch.cat([model.scales_log[keep_mask], added_scales], dim=0)
        final_quats = torch.cat([model.quaternions_raw[keep_mask], added_quats], dim=0)
        final_opacities = torch.cat([model.opacities_raw[keep_mask], added_opacities], dim=0)
        
        if final_means.shape[0] > Config.MAX_GAUSSIANS:
            final_means = final_means[:Config.MAX_GAUSSIANS]
            final_scales = final_scales[:Config.MAX_GAUSSIANS]
            final_quats = final_quats[:Config.MAX_GAUSSIANS]
            final_opacities = final_opacities[:Config.MAX_GAUSSIANS]
            
        model.num_gaussians = final_means.shape[0]
        model.means = nn.Parameter(final_means)
        model.scales_log = nn.Parameter(final_scales)
        model.quaternions_raw = nn.Parameter(final_quats)
        model.opacities_raw = nn.Parameter(final_opacities)
        
        new_optimizer = torch.optim.Adam([
            {'params': [model.means], 'lr': Config.LR_MEANS},
            {'params': [model.scales_log, model.quaternions_raw, model.opacities_raw], 'lr': Config.LR_REST}
        ])
        
        return new_optimizer, model.num_gaussians


def train():
    device = Config.DEVICE
    print(f"Device: {device}")
    
    model = RadiativeGaussianModel().to(device)
    optimizer = torch.optim.Adam([
        {'params': [model.means], 'lr': Config.LR_MEANS},
        {'params': [model.scales_log, model.quaternions_raw, model.opacities_raw], 'lr': Config.LR_REST}
    ])
    
    phantom = PhantomGenerator.generate_3d_phantom()
    rays_o, rays_d = RayGenerator.generate_all_rays()
    
    with torch.no_grad():
        target_y = CTProjector.project_volume(phantom, rays_o, rays_d)
        
    dataset = RayDataset(rays_o, rays_d, target_y)
    dataloader = DataLoader(dataset, batch_size=8192, shuffle=True)
    
    epochs = 6
    print(f"Initial Gaussians: {model.num_gaussians}")
    
    grad_norms_accum = torch.zeros(model.num_gaussians, device=device)
    densify_step_counter = 0
    global_step_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, batch in enumerate(dataloader):
            batch_o, batch_d, batch_y = batch
            batch_o, batch_d, batch_y = batch_o.to(device), batch_d.to(device), batch_y.to(device)
            
            pred_y = AnalyticalProjector.project_gaussians_chunk(model, batch_o, batch_d)
            
            scales = torch.exp(model.scales_log)
            opacities = model.get_opacities()
            
            reg_l1 = Config.LAMBDA_L1 * torch.sum(opacities)
            volumes = torch.prod(scales, dim=-1)
            reg_vol = Config.LAMBDA_VOL * torch.sum(volumes)
            
            max_scale = torch.max(scales, dim=-1)[0]
            min_scale = torch.min(scales, dim=-1)[0]
            aspect_ratio = max_scale / (min_scale + 1e-7)
            reg_aspect = Config.LAMBDA_ASPECT * torch.sum(torch.log(aspect_ratio) ** 2)
            
            loss = F.mse_loss(pred_y, batch_y) + reg_l1 + reg_vol + reg_aspect
            
            optimizer.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                if model.means.grad is not None:
                    grad_norms = torch.norm(model.means.grad, dim=-1)
                    if grad_norms_accum.shape[0] != model.num_gaussians:
                        grad_norms_accum = torch.zeros(model.num_gaussians, device=device)
                        densify_step_counter = 0
                    grad_norms_accum += grad_norms
                    densify_step_counter += 1
            
            optimizer.step()
            epoch_loss += loss.item()
            global_step_counter += 1
            
            if global_step_counter % Config.DENSIFY_INTERVAL == 0 and epoch < epochs - 1:
                optimizer, new_N = perform_densification_and_pruning(
                    model, optimizer, grad_norms_accum, densify_step_counter
                )
                grad_norms_accum = torch.zeros(new_N, device=device)
                densify_step_counter = 0
                
            if step % 20 == 0:
                print(f"Epoch {epoch+1:02d} | Step {step:02d}/{len(dataloader)} | Loss: {loss.item():.6f} | N: {model.num_gaussians}")
                
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(dataloader)
        print(f"-> Epoch {epoch+1:02d} finished | Avg Loss: {avg_loss:.6f} | N: {model.num_gaussians} | Time: {elapsed:.1f}s")
        
    torch.save(model.state_dict(), "gaussians_final.pt")
    print("Optimization completed. Weights saved to 'gaussians_final.pt'")

if __name__ == "__main__":
    train()
