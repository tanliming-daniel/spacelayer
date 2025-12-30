import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
import imageio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import glob
import cv2

# Add parent directory to path to import models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Ensure SpaTrackerV2 preprocess is importable
sys.path.append(os.path.join(os.path.dirname(__file__), 'submodules', 'SpaTrackerV2'))
import os
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import importlib.util
from pathlib import Path
# Dynamically load preprocess_image from the SpaTrackerV2 submodule to avoid import path issues
load_fn_path = Path(__file__).parent / 'submodules' / 'SpaTrackerV2' / 'models' / 'SpaTrackV2' / 'models' / 'vggt4track' / 'utils' / 'load_fn.py'
spec = importlib.util.spec_from_file_location('load_fn', str(load_fn_path))
load_fn_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_fn_mod)
preprocess_image = load_fn_mod.preprocess_image

from spacelayer.models.pipeline import TriangleMeshPipeline, CameraMotionGenerator, WorldSpaceTrianglePipeline
from spacelayer.models.tools import sample_points
from models.util_2d import compute_loss

def visualize_mesh_triangles(video_frames, track2d_pred, output_dir, global_coords):
    """
    Visualizes the Delaunay triangulation of 2D tracks on top of the video frames.
    """
    print("Visualizing mesh triangles as a video...")
    
    # Create a directory for visualization frames
    viz_frames_dir = os.path.join(output_dir, "mesh_viz_frames")
    os.makedirs(viz_frames_dir, exist_ok=True)

    # Prepare data for plotting
    tracks_np = track2d_pred.cpu().numpy()
    num_frames_viz, num_tracks, _ = tracks_np.shape
    
    # Get global coordinate transformations
    min_x_global = global_coords['min_x']
    min_y_global = global_coords['min_y']

    # Generate a frame for each time step
    for f in tqdm(range(num_frames_viz), desc="Generating Mesh Visualization Frames"):
        # Get the original video frame
        frame_image = video_frames[f]
        h, w, _ = frame_image.shape

        # Filter visible vertices for this frame
        visible_mask = tracks_np[f, :, 2] > 0
        visible_vertices = tracks_np[f, visible_mask, :2]

        if len(visible_vertices) < 4:
            # Not enough points to form a triangle, just show the frame
            plt.figure(figsize=(w/100, h/100), dpi=100)
            plt.imshow(frame_image)
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        else:
            # Perform Delaunay triangulation on visible vertices
            from scipy.spatial import Delaunay
            try:
                # Plot the original frame and the mesh on top
                plt.figure(figsize=(w/100, h/100), dpi=100)
                plt.imshow(frame_image)
                
                delaunay = Delaunay(visible_vertices)
                triangles = delaunay.simplices
                
                # The tracks are in global coordinates, so we need to shift them back
                # to the original image coordinates for plotting.
                plt.triplot(visible_vertices[:, 0] + min_x_global, visible_vertices[:, 1] + min_y_global, triangles, 'w-', lw=0.5)
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            except Exception as e:
                print(f"Could not generate mesh for frame {f}: {e}")
                plt.figure(figsize=(w/100, h/100), dpi=100)
                plt.imshow(frame_image)
                plt.axis('off')
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Save the figure
        frame_path = os.path.join(viz_frames_dir, f"frame_{f:04d}.png")
        plt.savefig(frame_path)
        plt.close()

    # Compile frames into a video
    mesh_video_path = os.path.join(output_dir, "mesh_visualization.mp4")
    with imageio.get_writer(mesh_video_path, fps=10, quality=8) as writer:
        for f in tqdm(range(num_frames_viz), desc="Compiling Mesh Video"):
            frame_path = os.path.join(viz_frames_dir, f"frame_{f:04d}.png")
            writer.append_data(imageio.imread(frame_path))
    
    # Clean up frames
    import shutil
    shutil.rmtree(viz_frames_dir)

    print(f"Saved mesh visualization video to {mesh_video_path}")

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
    if len(valid_points_all_frames) == 0:
        print("Warning: No valid points to visualize.")
        video_writer.release()
        plt.close(fig)
        return

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

def main():
    """
    Main function to run the training and rendering demo.
    """
    # --- 1. Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "results/pipeline_demo"
    os.makedirs(output_dir, exist_ok=True)

    # Config for the renderer model
    renderer_config = {
        'feat_dim': 128,
        'D': 8,
        'W': 256,
        'feat_embed': True,
        'feat_freq': 9,
        'pos_embed': True,
        'pos_freq': 9,
        'grid_size': (32, 32, 32), # D, H, W
        'world_bbox': (-1.5, -1.5, -1.5, 1.5, 1.5, 1.5) # A reasonable default bounding box
    }

    # Training parameters
    learning_rate = 0.001
    num_epochs = 100 # Increased epochs for better training with sampling
    points_per_epoch = 655360 # Number of points to sample per epoch

    # --- 2. Load Data ---
    num_frames_to_process = 37
    data_path = "/mnt/data-2/bob/spacelayer/results/bmx-bumps_background_points/bmx-bumps_background_tensors.pt"
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, map_location='cpu')
    
    # Extract data based on the keys saved by background_points.py
    track2d_coords = data['track2d_pred'][:num_frames_to_process,:,:2].to(device) # Shape (F, V, 2)
    vis_pred = data['vis_pred'][:num_frames_to_process].to(device)           # Shape (F, V, 1)
    track3d_pred_cam = data['track3d_pred'][:num_frames_to_process,:,:3].to(device)
    intrinsics = data['intrs'][:num_frames_to_process].to(device)
    extrinsics = data['extrs'][:num_frames_to_process].to(device)
    c2w_traj = data['c2w_traj'][:num_frames_to_process].to(device)

    # Combine 2D tracks and visibility into one tensor
    track2d_pred = torch.cat([track2d_coords, vis_pred], dim=-1)

    # Load video frames from the specified directory
    video_dir = "/mnt/data-2/bob/spacelayer/data/DAVIS/JPEGImages/480p/bmx-bumps"
    video_files = sorted(glob.glob(os.path.join(video_dir, "*.jpg")))[:num_frames_to_process]
    video_frames = []
    import PIL.Image
    for video_file in video_files:
        frame = PIL.Image.open(video_file).convert("RGB")
        frame = np.array(frame)
        video_frames.append(frame)
    video_frames = np.stack(video_frames, axis=0)  # Shape (F, H, W, 3)

    # Preprocess video frames so width/height match model preprocessing (consistent with track coords)
    # Convert to tensor (T, C, H, W), values in [0,1]
    video_tensor = torch.from_numpy(video_frames).permute(0, 3, 1, 2).float() / 255.0
    processed_video_tensor = preprocess_image(video_tensor)
    # Convert preprocessed video back to numpy in HWC uint8
    processed_video_np = (processed_video_tensor.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    video_frames = processed_video_np

    # Get image size from processed video frames
    num_frames, image_H, image_W, _ = video_frames.shape
    image_size = (image_H, image_W)

    # Load foreground masks to define the background area
    mask_dir = "/mnt/data-2/bob/spacelayer/data/DAVIS/Annotations/480p/car-turn"
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))[:num_frames_to_process]

    if len(mask_files) >= num_frames:
        print(f"Loading {num_frames} foreground masks from {mask_dir}...")
        loaded_masks = []
        # Ensure we only load as many masks as we have frames
        for f in mask_files[:num_frames]:
            mask_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            # Resize mask to match frame dimensions
            resized_mask = cv2.resize(mask_img, (image_W, image_H), interpolation=cv2.INTER_NEAREST)
            # Invert the foreground mask to get the background mask for rendering
            background_mask = resized_mask == 0
            loaded_masks.append(background_mask)
        masks_pred = torch.from_numpy(np.stack(loaded_masks)).to(device)
    else:
        print(f"Warning: Found {len(mask_files)} masks, but expected {num_frames}. Falling back to using the whole image as background.")
        masks_pred = torch.ones(num_frames, image_H, image_W, dtype=torch.bool).to(device)
    

    # --- 2a. Global Coordinate System Transformation (similar to bmxtree1080.py) ---
    print("Transforming coordinates to a global system...")

    # Calculate global min/max for x and y coordinates from track2d_pred
    min_x_global = track2d_pred[:, :, 0].min().item()
    min_y_global = track2d_pred[:, :, 1].min().item()
    max_x_global = track2d_pred[:, :, 0].max().item()
    max_y_global = track2d_pred[:, :, 1].max().item()

    # Calculate global width and height
    global_w = int(np.ceil(max_x_global - min_x_global))
    global_h = int(np.ceil(max_y_global - min_y_global))
    
    # Translate 2D tracks to the new global coordinate system so they start from (0,0)
    track2d_pred[:, :, 0] -= min_x_global
    track2d_pred[:, :, 1] -= min_y_global
    print(f"Translated 2D tracks by (-{min_x_global:.2f}, -{min_y_global:.2f})")
    print(f"Global coordinate system dimensions: {global_w}x{global_h}")

    global_coords = {
        'min_x': min_x_global,
        'min_y': min_y_global,
        'global_w': global_w,
        'global_h': global_h
    }

    # --- 2b. Convert 3D points from CAMERA to WORLD coordinates ---
    print("Converting 3D points from camera to world coordinates...")
    T, N, _ = track3d_pred_cam.shape
    points_hom_cam = torch.cat([track3d_pred_cam, torch.ones(T, N, 1, device=device)], dim=-1)
    world_points_hom = torch.einsum("tij,tnj->tni", c2w_traj, points_hom_cam)
    world_points = world_points_hom[:, :, :3]
    print(f"World points shape: {world_points.shape}")

    # --- 2c. Normalize 3D WORLD Tracks ---
    print("Normalizing 3D world tracks...")
    track3d_pred_unnormalized = world_points.clone() # This is now unnormalized world points
    
    bbox_min = world_points.reshape(-1, 3).min(dim=0)[0]
    bbox_max = world_points.reshape(-1, 3).max(dim=0)[0]
    center = (bbox_min + bbox_max) / 2.0
    scale = (bbox_max - bbox_min).max() / 2.0
    
    track3d_pred = (world_points - center) / scale # This is now NORMALIZED world points
    print(f"Normalized 3D tracks to be within a [-1, 1] bounding box.")

    # Create a mock cams_all for the camera generator, as it's needed for intrinsics
    cams_all = {'intrinsics': intrinsics, 'extrinsics': extrinsics}

    print("Data loaded and adapted successfully.")
    print(f"Tracks 2D shape (coords + vis): {track2d_pred.shape}")
    print(f"Tracks 3D shape (normalized world): {track3d_pred.shape}")
    print(f"Masks shape (generated): {masks_pred.shape}")
    print(f"Image size: {image_size}")
    print(f"Video frames shape: {video_frames.shape}")

    # --- 3. Initialize Models and Optimizer ---
    renderer_model = WorldSpaceTrianglePipeline(renderer_config).to(device)
    mesh_pipeline = TriangleMeshPipeline()
    
    optimizer = optim.Adam(renderer_model.parameters(), lr=learning_rate)
    
    print(f"Models initialized. Renderer has {sum(p.numel() for p in renderer_model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 4. Pre-process all frames for training ---
    print("Pre-processing all frames for sampling...")
    all_frame_data = []
    for i in tqdm(range(track2d_pred.shape[0]), desc="Processing Frames"):
        frame_vertices_2d = track2d_pred[i]
        frame_mask = masks_pred[i]
        mesh_data = mesh_pipeline._mesh_process(
            frame_vertices_2d, frame_mask, image_size[0], image_size[1],
            gt=video_frames[i], frame_idx=i, global_coords=global_coords
        )
        all_frame_data.append(mesh_data)
    print(f"Processed {len(all_frame_data)} valid frames.")

    # --- 5. Training Loop ---
    print(f"Starting training for {num_epochs} epochs...")
    losses = []
    # Use the static NORMALIZED 3D world vertices for the renderer
    frame_vertices_3d = track3d_pred[0].to(device) # Shape: (V, 3)
    import time
    for epoch in tqdm(range(num_epochs), desc="Training"):
        optimizer.zero_grad()
        t0 = time.time()
        # Sample points from all frames
        sampled_data = sample_points(all_frame_data, points_per_epoch, device)
        t1 = time.time()
        if sampled_data is None:
            print("No points available for sampling. Skipping epoch.")
            continue
        
        # Forward pass
        t2 = time.time()
        # Pass the static 3D track data to the model
        rendered_rgb = renderer_model(sampled_data, frame_vertices_3d, image_size, rgb_frame=None)
        t3 = time.time()
        gt_rgb_sample = sampled_data['gt_rgb']
        total_loss = compute_loss(rendered_rgb, gt_rgb_sample)
        t4 = time.time()

        if total_loss > 0:
            # Backward pass and optimization
            total_loss.backward()
            t5 = time.time()
            torch.nn.utils.clip_grad_norm_(renderer_model.parameters(), max_norm=1.0)
            optimizer.step()
            t6 = time.time()
        else:
            t5 = t6 = time.time()
        losses.append(total_loss.item())
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f} | "
                       f"采样: {t1-t0:.2f}s, forward: {t3-t2:.2f}s, loss: {t4-t3:.2f}s, backward+step: {(t6-t5):.2f}s")

    print("Training finished.")

    # Plot and save the loss curve
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    loss_curve_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(loss_curve_path)
    print(f"Saved training loss curve to {loss_curve_path}")

    # --- 5. Reconstruction of Original Video ---
    print("Starting reconstruction of original video...")
    renderer_model.eval()
    reconstructed_frames = []
    with torch.no_grad():
        for i in tqdm(range(track2d_pred.shape[0]), desc="Reconstructing Original Video"):
            frame_vertices_2d = track2d_pred[i]
            # No need to select frame_vertices_3d here, pass the whole tensor
            frame_mask = masks_pred[i]
            mesh_data = mesh_pipeline._mesh_process(
                frame_vertices_2d, frame_mask, image_size[0], image_size[1],
                gt=video_frames[i], frame_idx=i, global_coords=global_coords
            )
            if mesh_data is None:
                reconstructed_frames.append(np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8))
                continue
            
            # Pass the static 3D track data to the model
            rendered_rgb = renderer_model(mesh_data, frame_vertices_3d, image_size, rgb_frame=None)

            frame_np = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            mask = frame_mask.cpu().numpy() if isinstance(frame_mask, torch.Tensor) else frame_mask
            mask_flat = mask.flatten()
            rgb_np = (rendered_rgb.cpu().numpy() * 255).astype(np.uint8)
            frame_np.reshape(-1, 3)[mask_flat == 1] = rgb_np
            reconstructed_frames.append(frame_np)

    recon_video_path = os.path.join(output_dir, "reconstruction.mp4")
    imageio.mimsave(recon_video_path, reconstructed_frames, fps=10, quality=8)
    print(f"Reconstruction finished. Saved video to {recon_video_path}")

    # --- 6. Inference with New Camera Motion and Re-rendering ---
    print("\n--- Inference with New Camera Motion and Re-rendering ---")
    renderer_model.eval()

    # 6a. Generate a new zoom-in camera motion
    print("Generating a new zoom-in camera motion...")
    T = track3d_pred_unnormalized.shape[0]
    new_cam_gen = CameraMotionGenerator(
        motion_type='zoom_in',
        frame_num=T,
        H=image_size[0],
        W=image_size[1],
        device=device
    )
    new_cam_gen.set_intr(intrinsics)
    new_cam_gen.set_extr(torch.inverse(c2w_traj))
    new_c2w_matrices = new_cam_gen.get_default_motion()
    new_w2c_matrices = torch.inverse(new_c2w_matrices)
    print(f"Generated {new_w2c_matrices.shape[0]} new camera poses.")

    # 6b. Project world points to get new 2D tracks
    print("Projecting world points to get new 2D tracks...")
    with torch.no_grad():
        projected_uvz_new = new_cam_gen.w2s_vggt(
            world_points=track3d_pred_unnormalized.float(),
            extrinsics=new_w2c_matrices.float(),
            intrinsics=intrinsics.float(),
            poses=None,
            override_extrinsics=False
        )
        
        # Create a visibility mask for the new projections
        visibility_new = (projected_uvz_new[..., 2] > 1e-3).float().unsqueeze(-1)
        
        # Combine new 2D coordinates with visibility
        new_track2d_pred = torch.cat([projected_uvz_new[..., :2], visibility_new], dim=-1)

    # 6c. Create a new global coordinate system for the new 2D tracks
    print("Creating a new global coordinate system for re-rendering...")
    visible_mask_new = new_track2d_pred[..., 2] > 0
    visible_points_x = new_track2d_pred[..., 0][visible_mask_new]
    visible_points_y = new_track2d_pred[..., 1][visible_mask_new]

    if visible_points_x.numel() > 0 and visible_points_y.numel() > 0:
        new_min_x_global = visible_points_x.min().item()
        new_min_y_global = visible_points_y.min().item()
        new_max_x_global = visible_points_x.max().item()
        new_max_y_global = visible_points_y.max().item()
    else:
        new_min_x_global, new_min_y_global, new_max_x_global, new_max_y_global = 0, 0, image_W, image_H

    new_global_w = int(np.ceil(new_max_x_global - new_min_x_global))
    new_global_h = int(np.ceil(new_max_y_global - new_min_y_global))
    
    new_global_coords = {
        'min_x': new_min_x_global,
        'min_y': new_min_y_global,
        'global_w': new_global_w,
        'global_h': new_global_h
    }
    print(f"New global coordinate system for rendering: {new_global_w}x{new_global_h}")

    # 6d. Render frames from the new camera view
    print("Rendering frames from the new camera view...")
    rendered_frames_new_view = []
    with torch.no_grad():
        for i in tqdm(range(new_track2d_pred.shape[0]), desc="Rendering New View"):
            frame_vertices_2d_new = new_track2d_pred[i].clone()
            
            # Normalize the new 2D vertices using the NEW global coordinate system
            frame_vertices_2d_new[:, 0] -= new_min_x_global
            frame_vertices_2d_new[:, 1] -= new_min_y_global

            # Create a mask that covers the entire image for full rendering
            frame_mask = torch.ones(image_H, image_W, dtype=torch.bool).to(device)

            mesh_data = mesh_pipeline._mesh_process(
                frame_vertices_2d_new, frame_mask, image_size[0], image_size[1],
                gt=video_frames[i], frame_idx=i, global_coords=new_global_coords
            )

            if mesh_data is None:
                rendered_frames_new_view.append(np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8))
                continue

            # Render using the original NORMALIZED 3D world points
            rendered_rgb = renderer_model(mesh_data, frame_vertices_3d, image_size, rgb_frame=None)

            # Convert rendered colors back into an image
            rgb_np = (rendered_rgb.cpu().numpy() * 255.0).astype(np.uint8)
            frame_np = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            mask_flat = frame_mask.cpu().numpy().flatten()
            frame_np.reshape(-1, 3)[mask_flat == 1] = rgb_np
            rendered_frames_new_view.append(frame_np)

    # Save the rendered frames as a video
    video_path_new_view = os.path.join(output_dir, "rendered_video_new_view.mp4")
    imageio.mimsave(video_path_new_view, rendered_frames_new_view, fps=10, quality=8)
    print(f"Inference with new view finished. Saved rendered video to {video_path_new_view}")

    # 6e. (Optional) Visualize the new projected trajectories
    print("Visualizing new projected trajectories...")
    H_proj = int(intrinsics[0, 1, 2].item() * 2)
    W_proj = int(intrinsics[0, 0, 2].item() * 2)
    image_size_proj = (H_proj, W_proj)
    
    valid_depth_mask_new = projected_uvz_new[..., 2] > 1e-3
    on_screen_mask_new = (projected_uvz_new[..., 0] >= 0) & (projected_uvz_new[..., 0] < W_proj) & \
                         (projected_uvz_new[..., 1] >= 0) & (projected_uvz_new[..., 1] < H_proj)
    composite_mask_new = valid_depth_mask_new & on_screen_mask_new
    
    video_output_path = os.path.join(output_dir, "projected_trajectories_video.mp4")
    visualize_trajectory_video(
        projected_uvz_new.cpu().numpy(),
        composite_mask_new.cpu().numpy(),
        video_output_path,
        image_size_proj
    )

    # --- Visualize Mesh Triangles ---
    visualize_mesh_triangles(video_frames, track2d_pred.clone(), output_dir, global_coords)
    # --- End Visualize Mesh Triangles ---

if __name__ == '__main__':
    main()
