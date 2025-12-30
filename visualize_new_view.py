import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

# Add parent directory to path to import models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure SpaTrackerV2 is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'submodules', 'SpaTrackerV2'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from spacelayer.models.pipeline import CameraMotionGenerator

def generate_pose_from_params(position, quaternion, device):
    """
    Generates a single c2w pose matrix from a THREE.js position and quaternion.
    """
    # Convert to torch tensors
    pos_tensor = torch.tensor([position['x'], position['y'], position['z']], device=device)
    quat_tensor = torch.tensor([quaternion['x'], quaternion['y'], quaternion['z'], quaternion['w']], device=device)

    # Create a 4x4 transformation matrix from position and quaternion
    c2w = torch.eye(4, device=device)
    
    # Rotation matrix from quaternion
    x, y, z, w = quat_tensor
    c2w[0, 0] = 1 - 2*y*y - 2*z*z
    c2w[0, 1] = 2*x*y - 2*z*w
    c2w[0, 2] = 2*x*z + 2*y*w
    c2w[1, 0] = 2*x*y + 2*z*w
    c2w[1, 1] = 1 - 2*x*x - 2*z*z
    c2w[1, 2] = 2*y*z - 2*x*w
    c2w[2, 0] = 2*x*z - 2*y*w
    c2w[2, 1] = 2*y*z + 2*x*w
    c2w[2, 2] = 1 - 2*x*x - 2*y*y
    
    # Position
    c2w[:3, 3] = pos_tensor

    # Invert the camera's y and z axes to match the convention used in the pipeline
    # (THREE.js camera looks down -Z, graphics pipelines often assume +Z)
    # This aligns the camera's orientation with the world coordinate system.
    yz_flip = torch.eye(4, device=device)
    yz_flip[1, 1] = -1
    yz_flip[2, 2] = -1
    c2w = c2w @ yz_flip

    return c2w

def main():
    """
    Main function to project 3D points to a new custom camera view.
    """
    # --- 1. Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "results/new_view_projection"
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Load Data ---
    num_frames_to_process = 37 # Should match the data
    data_path = "/mnt/data-2/bob/spacelayer/results/parkour_background_points/parkour_background_tensors.pt"
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    
    track3d_pred_cam = data['track3d_pred'][:num_frames_to_process,:,:3].to(device)
    intrinsics = data['intrs'][:num_frames_to_process].to(device)
    c2w_traj = data['c2w_traj'][:num_frames_to_process].to(device)

    # --- 3. Convert 3D points from CAMERA to WORLD coordinates ---
    print("Converting 3D points from camera to world coordinates...")
    T, N, _ = track3d_pred_cam.shape
    points_hom_cam = torch.cat([track3d_pred_cam, torch.ones(T, N, 1, device=device)], dim=-1)
    world_points_hom = torch.einsum("tij,tnj->tni", c2w_traj, points_hom_cam)
    world_points = world_points_hom[:, :, :3]
    print(f"World points shape: {world_points.shape}")

    # --- 4. Define a new camera pose from interactive viewer parameters ---
    # TODO: Paste the parameters from the browser's console here.
    # Press 'p' in the visualizer to print these values.
    
    # Example parameters (replace with your own)
    new_cam_position = {'x': 0.02026883019087483, 'y': 0.12288383331566874, 'z': 0.17266361058722757}
    new_cam_quaternion = {'x':-0.09440774263046758, 'y':-0.07318722271684928, 'z':-0.006959441442382149, 'w':0.9928153779717674}

    print("Generating a new camera pose from specified parameters...")
    # Generate a single pose and repeat it for all frames to create a static shot
    single_c2w = generate_pose_from_params(new_cam_position, new_cam_quaternion, device)
    new_c2w_matrices = torch.inverse(single_c2w.unsqueeze(0)) # w2c is inverse of c2w
    print("Generated new static camera pose.")

    # --- 5. Project world points to get new 2D tracks ---
    print("Projecting world points to the new camera view...")
    # We need a CameraMotionGenerator instance to use its projection method.
    H_proj = int(intrinsics[0, 1, 2].item() * 2)
    W_proj = int(intrinsics[0, 0, 2].item() * 2)
    new_cam_gen = CameraMotionGenerator(
        motion_type='static', # Motion type is irrelevant
        frame_num=1, # Only need one frame for static projection
        H=H_proj,
        W=W_proj,
        device=device
    )
    new_cam_gen.set_intr(intrinsics)

    with torch.no_grad():
        # Project all 3D points from all frames onto the new static camera
        # We use the 3D points from the first frame as representative
        projected_uvz_new = new_cam_gen.w2s_vggt(
            world_points=world_points.float(),
            extrinsics=new_c2w_matrices.float(), # Use our new static extrinsics
            intrinsics=intrinsics.float(),
            poses=None,
            override_extrinsics=True # Ensure we use the provided extrinsics
        )
        
        # Create a visibility mask for the new projections
        visibility_new = (projected_uvz_new[..., 2] > 1e-3).float().unsqueeze(-1)

    # --- 6. Visualize the new projected points on a single image ---
    print("Visualizing the projected points for the new view...")
    
    # Use the first frame's projection for visualization
    first_frame_projections = projected_uvz_new[0].cpu().numpy()
    first_frame_visibility = visibility_new[0].cpu().numpy().squeeze() > 0
    
    visible_points = first_frame_projections[first_frame_visibility]

    plt.figure(figsize=(W_proj/100, H_proj/100), dpi=100)
    plt.title("Projected Points in New Camera View")
    plt.imshow(np.zeros((H_proj, W_proj, 3))) # Black background
    
    if visible_points.shape[0] > 0:
        plt.scatter(visible_points[:, 0], visible_points[:, 1], c='r', s=2) # Draw red points
    
    plt.xlim(0, W_proj)
    plt.ylim(H_proj, 0) # Invert Y-axis for image coordinates
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.gca().set_aspect('equal', adjustable='box')
    
    projection_img_path = os.path.join(output_dir, "projected_points_new_view.png")
    plt.savefig(projection_img_path)
    plt.close()
    print(f"Saved new view projection visualization to {projection_img_path}")
    print("\nScript finished. You can find the output image in the 'results/new_view_projection' directory.")

if __name__ == '__main__':
    main()
