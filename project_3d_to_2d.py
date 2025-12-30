import torch
import numpy as np
import os
from models.pipeline import CameraMotionGenerator
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def visualize_trajectory_video(projected_uvz_np, composite_mask_np, output_path, image_size, fps=10):
    """
    Visualizes the 2D trajectories over all frames and saves it as an MP4 video using OpenCV.
    """
    print("Visualizing projected trajectories as a video...")
    
    T, N, _ = projected_uvz_np.shape
    H, W = image_size
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Note: The figure size needs to be converted from inches to pixels for the video writer.
    # We will render the plot to a numpy array directly, so we can control the exact pixel dimensions.
    fig = plt.figure(figsize=(W/100, H/100), dpi=100)
    
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Determine overall plot limits from all valid points across all frames
    valid_points_all_frames = projected_uvz_np[composite_mask_np]
    min_x = valid_points_all_frames[:, 0].min()
    max_x = valid_points_all_frames[:, 0].max()
    min_y = valid_points_all_frames[:, 1].min()
    max_y = valid_points_all_frames[:, 1].max()
    
    padding_x = (max_x - min_x) * 0.1
    padding_y = (max_y - min_y) * 0.1
    
    plot_min_x = min_x - padding_x
    plot_max_x = max_x + padding_x
    plot_min_y = min_y - padding_y
    plot_max_y = max_y + padding_y

    # Generate a unique color for each track
    colors = plt.cm.jet(np.linspace(0, 1, N))

    for t in tqdm(range(T), desc="Generating video frames"):
        plt.clf()  # Clear the current figure
        
        frame_points = projected_uvz_np[t]
        frame_mask = composite_mask_np[t]
        
        valid_frame_points = frame_points[frame_mask]
        valid_colors = colors[frame_mask]

        plt.title(f"Projected 2D Trajectories (Frame {t+1}/{T})")
        plt.xlim(plot_min_x, plot_max_x)
        plt.ylim(plot_max_y, plot_min_y)  # Invert Y-axis
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        
        if len(valid_frame_points) > 0:
            plt.scatter(valid_frame_points[:, 0], valid_frame_points[:, 1], c=valid_colors, s=4)

        # Render the plot to a numpy array
        fig.canvas.draw()
        # Use tostring_argb() which returns an ARGB buffer
        argb_buf = fig.canvas.tostring_argb()
        img = np.frombuffer(argb_buf, dtype=np.uint8)
        # Reshape to (height, width, 4)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        # The buffer is ARGB. We need to convert it to BGR for OpenCV.
        # First, let's get it to RGBA by rolling the channels.
        img = np.roll(img, shift=-1, axis=2) # ARGB -> RGBA
        
        # Convert RGBA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        video_writer.write(img)

    video_writer.release()
    plt.close(fig)
    print(f"Saved trajectory visualization video to {output_path}")


def project_3d_to_2d():
    """
    This script demonstrates how to project 3D points to a 2D plane
    using the corrected w2s_vggt function.
    """
    # 1. Load data from sptrack
    print("Loading data...")
    data_path = "/mnt/data-2/bob/spacelayer/results/bmx-bumps_background_points/bmx-bumps_background_tensors.pt"
    data = torch.load(data_path, map_location='cpu')

    # Process up to 49 frames, but no more than what's available in the data.
    num_frames_to_process = min(len(data['track3d_pred']), 49)
    track3d_pred = data['track3d_pred'][:num_frames_to_process]  # (T, N, 6) -> Assumed to be in CAMERA coordinates
    c2w_traj = data['c2w_traj'][:num_frames_to_process]      # (T, 4, 4) -> camera-to-world matrices
    intrs = data['intrs'][:num_frames_to_process]            # (T, 3, 3) -> intrinsic matrices

    T, N, _ = track3d_pred.shape
    print(f"Data loaded: {T} frames, {N} points per frame.")

    # 2. Convert 3D points from CAMERA coordinates to WORLD coordinates
    print("Converting 3D points from camera to world coordinates...")
    
    # Extract xyz and create homogeneous coordinates
    track3d_xyz = track3d_pred[..., :3]
    points_hom_cam = torch.cat([track3d_xyz, torch.ones(T, N, 1)], dim=-1)
    
    # Transform to world coordinates using c2w matrices
    # Resulting shape: (T, N, 4)
    world_points_hom = torch.einsum("tij,tnj->tni", c2w_traj, points_hom_cam)
    world_points = world_points_hom[:, :, :3]
    print(f"World points shape: {world_points.shape}")

    # 3. Prepare matrices for projection
    print("Preparing matrices for projection...")
    
    # Initialize the CameraMotionGenerator
    # The device can be 'cpu' or 'cuda' depending on your setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cam_gen = CameraMotionGenerator(motion_type='static', frame_num=T, device=device)
    cam_gen.set_intr(intrs)

    # The w2s_vggt function requires world-to-camera (w2c) matrices.
    # We can get them by inverting the camera-to-world (c2w) matrices.
    w2c_matrices = torch.inverse(c2w_traj)
    print(f"w2c matrices shape: {w2c_matrices.shape}")

    # 4. Call w2s_vggt to project WORLD points to SCREEN coordinates
    print("Projecting world points to 2D screen coordinates...")
    
    projected_uvz = cam_gen.w2s_vggt(
        world_points=world_points.float(),
        extrinsics=w2c_matrices,  # Pass the w2c matrices
        intrinsics=intrs,
        poses=None,              # No new poses needed for this operation
        override_extrinsics=False
    )
    
    # The output `projected_uvz` contains the 2D coordinates (u, v) and depth (z)
    projected_2d_points = projected_uvz[..., :2]
    
    print("\nProjection complete!")
    print(f"Shape of the projected 2D points (u, v): {projected_2d_points.shape}")
    print(f"Shape of the full output (u, v, z): {projected_uvz.shape}")

    # You can now use `projected_2d_points` for your visualization or further processing.
    # For example, let's check the coordinates of the first point in the first frame:
    if T > 0 and N > 0:
        print(f"\nExample: Projected 2D coordinates of the first point in the first frame: {projected_2d_points[0, 0].cpu().numpy()}")

    # 5. Filter out invalid points and visualize the projected trajectories
    print("Filtering out points with small depth or outside the image frame...")
    
    # Get image size from intrinsics to define the valid screen area
    H = int(intrs[0, 1, 2].item() * 2)
    W = int(intrs[0, 0, 2].item() * 2)
    image_size = (H, W)

    # Create a composite mask for points that are valid
    valid_depth_mask = projected_uvz[..., 2] > 1e-3  # Valid depth
    on_screen_mask = (projected_uvz[..., 0] >= 0) & (projected_uvz[..., 0] < W) & \
                     (projected_uvz[..., 1] >= 0) & (projected_uvz[..., 1] < H)
    
    composite_mask = valid_depth_mask & on_screen_mask
    
    # Visualize the filtered projected trajectories as a video
    output_dir = "spacelayer/results/pipeline_demo"
    os.makedirs(output_dir, exist_ok=True)
    video_output_path = os.path.join(output_dir, "projected_trajectories_video.mp4")
    
    visualize_trajectory_video(
        projected_uvz.cpu().numpy(), 
        composite_mask.cpu().numpy(), 
        video_output_path, 
        image_size
    )

    # --- 6. Generate and Project from a New Camera View ---
    print("\n--- Generating and Projecting from a New Camera View ---")

    # 6a. Generate a new spiral camera motion
    print("Generating a new spiral camera motion...")
    new_cam_gen = CameraMotionGenerator(
        motion_type='spiral 0.01',  # A spiral motion
        frame_num=world_points.shape[0],
        H=image_size[0],
        W=image_size[1],
        device=device
    )
    new_cam_gen.set_intr(intrs)  # Use the same intrinsics
    
    # Set original extrinsics to establish a reference frame for the new motion
    new_cam_gen.set_extr(torch.inverse(c2w_traj)) 
    
    # Get new camera poses (c2w) by directly calling the spiral generation function
    # The first parameter is the radius of the spiral.
    new_c2w_matrices = new_cam_gen.spiral_poses(radius=0.1) 

    # The projection function needs world-to-camera (w2c) matrices
    new_w2c_matrices = torch.inverse(new_c2w_matrices)

    print(f"Generated {new_w2c_matrices.shape[0]} new camera poses.")

    # 6b. Project world points using the new camera poses
    print("Projecting world points from the new camera views...")
    projected_uvz_new = new_cam_gen.w2s_vggt(
        world_points=world_points.float(),
        extrinsics=new_w2c_matrices.float(),  # Use the new w2c matrices
        intrinsics=intrs.float(),
        poses=None,
        override_extrinsics=False # This is counter-intuitive but how the function is designed
    )

    # 6c. Filter and visualize the new projected trajectories
    print("Filtering and visualizing new projected trajectories...")
    valid_depth_mask_new = projected_uvz_new[..., 2] > 1e-3
    on_screen_mask_new = (projected_uvz_new[..., 0] >= 0) & (projected_uvz_new[..., 0] < W) & \
                         (projected_uvz_new[..., 1] >= 0) & (projected_uvz_new[..., 1] < H)
    composite_mask_new = valid_depth_mask_new & on_screen_mask_new

    # Visualize the new trajectories
    new_video_output_path = os.path.join(output_dir, "projected_trajectories_new_view.mp4")
    visualize_trajectory_video(
        projected_uvz_new.cpu().numpy(),
        composite_mask_new.cpu().numpy(),
        new_video_output_path,
        image_size
    )
    print(f"Projection from new view complete. Video saved to {new_video_output_path}")


if __name__ == '__main__':
    project_3d_to_2d()
