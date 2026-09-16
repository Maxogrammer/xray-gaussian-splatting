import json
import torch
import torch.nn.functional as F

def export_to_html_viewer(checkpoint_path: str = "gaussians_final.pt", output_html: str = "viewer.html"):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    means = state_dict["means"].numpy()
    scales_log = state_dict["scales_log"].numpy()
    quaternions_raw = state_dict["quaternions_raw"].numpy()
    opacities_raw = state_dict["opacities_raw"].numpy()
    
    scales = torch.exp(torch.tensor(scales_log)).numpy()
    opacities = F.softplus(torch.tensor(opacities_raw)).numpy()
    quats_tensor = F.normalize(torch.tensor(quaternions_raw), p=2, dim=-1).numpy()
    quats_js = quats_tensor[:, [1, 2, 3, 0]]
    
    gaussian_data = []
    for i in range(len(means)):
        gaussian_data.append({
            "pos": means[i].tolist(),
            "scale": scales[i].tolist(),
            "quat": quats_js[i].tolist(),
            "opacity": float(opacities[i][0])
        })
        
    json_data = json.dumps(gaussian_data)
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3DGS CT Reconstruction Viewer</title>
    <style>
        body {{ margin: 0; background: #0e1117; overflow: hidden; font-family: monospace; }}
        #hud {{ position: absolute; top: 15px; left: 15px; color: #00ffcc; background: rgba(14,17,23,0.9); padding: 12px; border: 1px solid #262730; border-radius: 4px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="hud">
        <b>3DGS Tomography Viewer</b><br>
        Active Gaussians: {len(means)}<br>
        LMB: Rotate | Wheel: Zoom
    </div>
    <script>
        const gaussians = {json_data};
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0e1117);
        const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(2.5, 2.5, 3.5);
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshBasicMaterial({{ transparent: true, opacity: 0.08, depthWrite: false, blending: THREE.AdditiveBlending }});
        const mesh = new THREE.InstancedMesh(geometry, material, gaussians.length);
        
        const matrix = new THREE.Matrix4();
        const pos = new THREE.Vector3();
        const rot = new THREE.Quaternion();
        const scale = new THREE.Vector3();
        const color = new THREE.Color();
        
        for (let i = 0; i < gaussians.length; i++) {{
            const g = gaussians[i];
            pos.set(g.pos[0], g.pos[1], g.pos[2]);
            rot.set(g.quat[0], g.quat[1], g.quat[2], g.quat[3]);
            scale.set(g.scale[0], g.scale[1], g.scale[2]);
            matrix.compose(pos, rot, scale);
            mesh.setMatrixAt(i, matrix);
            const intensity = Math.min(1.0, g.opacity * 2.0);
            color.setRGB(intensity, intensity, intensity);
            mesh.setColorAt(i, color);
        }}
        mesh.instanceMatrix.needsUpdate = true;
        scene.add(mesh);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Viewer saved to {output_html}")

if __name__ == "__main__":
    export_to_html_viewer()
