import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2

def _sort(
            rays_o: torch.Tensor,
            rays_d: torch.Tensor,
            rotations: torch.Tensor, #(N, 3, 3),
            scales: torch.Tensor, #(N, 3),
            positions: torch.Tensor, #(N, 3),
    ):
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        N = scales.shape[0]
        M = rays_o.shape[0]


        scales_inv = 1.0 / scales
        S_inv = torch.diag_embed(scales_inv)
        R_inv = rotations.transpose(-2, -1)
        transform = torch.bmm(S_inv, R_inv)

        rays_o_transformed = torch.bmm(transform, (rays_o.unsqueeze(0) - positions.unsqueeze(1)).transpose(1, 2)).transpose(1, 2)  # (N, M, 3)
        rays_d_transformed = torch.einsum('nij,mj->nmi', transform, rays_d) # (N, M, 3)

        a = torch.sum(rays_d_transformed**2, dim=-1) # (N, M)
        b = 2 * torch.sum(rays_o_transformed * rays_d_transformed, dim=-1) # (N, M)
        c = torch.sum(rays_o_transformed**2, dim=-1) - 1 # (N, M)

        discriminant = b**2 - 4 * a * c # (N, M)
        valid = discriminant >= 0
        t0 = (-b - torch.sqrt(discriminant.clamp(min=0))) / (2 * a)
        
        t1 = (-b + torch.sqrt(discriminant.clamp(min=0))) / (2 * a)

        ind = torch.arange(N, device=rays_o.device).unsqueeze(-1).expand(-1, M)
        
 
        t0 = torch.where((t0 > 0) & valid, t0, -1)
        t1 = torch.where((t1 > 0) & valid, t1, -1)

        ray_in = (t0 > 0).to(torch.int)
        ray_out = - (t1 > 0).to(torch.int)
        # ind_in = torch.where(t0 > 0, ind, -1)
        # ind_out = torch.where(t1 > 0, ind, -1)

        # return torch.cat([t0, t1], dim=0), torch.cat([ray_in, ray_out], dim=0), torch.cat([ind_in, ind_out], dim=0)

        t = torch.cat([t0, t1], dim=0) # (2N, M)
        ray_in_or_out = torch.cat([ray_in, ray_out], dim=0) # (2N, M)
        ind_in_or_out = torch.cat([ind, ind], dim=0) # (2N, M)

        indices = torch.argsort(t, dim=0)
        intersections = torch.gather(t, 0, indices)
        ray_in_or_out_sorted = torch.gather(ray_in_or_out, 0, indices)
        ind_in_or_out_sorted = torch.gather(ind_in_or_out, 0, indices)
        # intersections, indices = torch.sort(t_min, dim=0)

        # (2N, M), (2N, M), (2N, M)
        return indices, ray_in_or_out_sorted, ind_in_or_out_sorted, intersections


class EllipsoidRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3] # (N, 2)
        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        ### FILL:
        ### J_proj = ...
        # proj = torch.zeros((1, 2, 3), device=means3D.device)
        J_proj[..., 0, 0] = K[0, 0] / cam_points[..., 2]
        J_proj[..., 0, 2] = - K[0, 0] / (cam_points[..., 2] ** 2) * cam_points[..., 0]
        J_proj[..., 1, 1] = K[1, 1] / cam_points[..., 2]
        J_proj[..., 1, 2] = - K[1, 1] / (cam_points[..., 2] ** 2) * cam_points[..., 1]
        # Transform covariance to camera space
        ### FILL: Aplly world to camera rotation to the 3d covariance matrix
        ### covs_cam = ...  # (N, 3, 3)
        covs_cam = R @ covs3d @ R.T
        # Project to 2D
        covs2D = J_proj @ covs_cam @ J_proj.permute(0, 2, 1)  # (N, 2, 2)
        
        return means2D, covs2D, depths

    def get_ray_directions(self, H, W, K) -> torch.Tensor:
        i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='ij')
        i = i.t()
        j = j.t()
        directions = torch.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], torch.ones_like(i)], dim=-1)
        return torch.nn.functional.normalize(directions, dim=-1)

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)
        
        # Compute determinant for normalization
        ### FILL: compute the gaussian values
        ### gaussian = ... ## (N, H, W)
        dx_T = dx.reshape(N, H, W, 1, 2)
        dx = dx.reshape(N, H, W, 2, 1)
        covs2D_inv = torch.inverse(covs2D.reshape(N, 1, 1, 2, 2))
        # P = torch.matmul(dx, torch.matmul(covs2D_inv, dx))
        P = dx_T @ covs2D_inv @ dx
        
        gaussian = torch.exp(-0.5 * P.squeeze(-1).squeeze(-1)) / (2 * np.pi * torch.det(covs2D).sqrt()[..., None, None])

        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            # covs3d: torch.Tensor,           # (N, 3, 3)
            rays_o: torch.Tensor,           # (M, 3)
            rays_d: torch.Tensor,           # (M, 3)
            rotations: torch.Tensor,        # (N, 3, 3)
            scales: torch.Tensor,           # (N, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        
        ind, ray_in_or_out, ind_in_or_out, tt =  _sort(rays_o, rays_d, rotations, scales, means3D)
        delta_t = torch.diff(tt, dim=0) # (2N-1, M)
        ray_in_or_out = ray_in_or_out[:-1]
        delta_density = opacities[ind_in_or_out, :][:-1] #(2N-1, M, 1)
        delta_density = delta_density * ray_in_or_out.float().unsqueeze(-1)

        delta_color = colors[ind_in_or_out, :][:-1] # (2N-1, M, 3)
        delta_color = delta_color * ray_in_or_out.float().unsqueeze(-1)

        density = torch.cumsum(delta_density, dim=0) # (2N - 1, M, 1)
        color = torch.cumsum(delta_color * delta_density, dim=0) # (2N - 1, M, 3)
        color = color / density.clamp(min=1e-6) # (2N - 1, M, 3)

        tmp = torch.exp(-delta_t.unsqueeze(-1) * density) # (2N - 1, M, 1)
        prod = torch.cumprod(tmp, dim=0) # (2N - 1, M, 1)
        prod = torch.cat([torch.ones(1, prod.shape[1], 1).cuda(), prod[1:]], dim=0) # (2N - 1, M, 1)
        C = torch.sum(prod * color * tmp, dim=0) # (M, 3)

        rendered = C
        # rendered = C

        # N = means3D.shape[0]
        
        # # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        # means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # # 2. Depth mask
        # valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # # 3. Sort by depth
        # indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        # means2D = means2D[indices]      # (N, 2)
        # covs2D = covs2D[indices]       # (N, 2, 2)
        # colors = colors[ indices]       # (N, 3)
        # opacities = opacities[indices] # (N, 1)
        # valid_mask = valid_mask[indices] # (N,)
        
        # # 4. Compute gaussian values
        # gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # # 5. Apply valid mask
        # gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # # 6. Alpha composition setup
        # alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        # colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        # colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
    
        # 7. Compute weights
        ### FILL:
        ### weights = ... # (N, H, W)
        # weights = alphas * torch.cumprod(1 - alphas, dim=0)
        
        # # 8. Final rendering
        # rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
