import struct
import torch
import numpy as np

def export_to_standard_ply(checkpoint_path="gaussians_final.pt", ply_path="gaussians.ply"):
    try:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    means = state_dict["means"].numpy()
    scales_log = state_dict["scales_log"].numpy()
    quats = state_dict["quaternions_raw"].numpy()
    opacities_raw = state_dict["opacities_raw"].numpy()
    
    num_pts = means.shape[0]
    quats_norm = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    opacities = 1.0 / (1.0 + np.exp(-opacities_raw))
    
    print(f"Exporting {num_pts} Gaussians to binary PLY...")
    
    with open(ply_path, "wb") as f:
        header = f"""ply
format binary_little_endian 1.0
element vertex {num_pts}
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        f.write(header.encode("utf-8"))
        
        for i in range(num_pts):
            x, y, z = means[i]
            intensity = min(1.0, opacities[i][0] * 3.0)
            sh = (intensity - 0.5) / 0.28209
            
            opac = opacities_raw[i][0]
            s0, s1, s2 = scales_log[i]
            r0, r1, r2, r3 = quats_norm[i]
            
            data = struct.pack(
                "<17f", 
                x, y, z, 
                0.0, 0.0, 0.0,
                sh, sh, sh, 
                opac, 
                s0, s1, s2, 
                r0, r1, r2, r3
            )
            f.write(data)
            
    print(f"Saved: {ply_path}")

if __name__ == "__main__":
    export_to_standard_ply()