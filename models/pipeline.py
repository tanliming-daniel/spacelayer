import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.util_2d import compute_bary_triangles, compute_bary_vertices, Embedding
from scipy.spatial import Delaunay

class CameraMotionGenerator:
    def __init__(self, motion_type, frame_num=49, H=480, W=720, fx=None, fy=None, fov=55, device='cuda'):
        self.motion_type = motion_type
        self.frame_num = frame_num
        self.fov = fov
        self.device = device
        self.W = W
        self.H = H
        self.intr = torch.tensor([
            [0, 0, W / 2],
            [0, 0, H / 2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        # if fx, fy not provided
        if not fx or not fy:
            fov_rad = math.radians(fov)
            fx = fy = (W / 2) / math.tan(fov_rad / 2)
 
        self.intr[0, 0] = fx
        self.intr[1, 1] = fy   

        self.extr = torch.eye(4, device=device)

    def s2w_vggt(self, points, extrinsics, intrinsics):
        """
        Transform points from pixel coordinates to world coordinates
        
        Args:
            points: Point cloud data of shape [T, N, 3] in uvz format
            extrinsics: Camera extrinsic matrices [B, T, 4, 4] or [T, 4, 4] (c2w)
            intrinsics: Camera intrinsic matrices [B, T, 3, 3] or [T, 3, 3]
            
        Returns:
            world_points: Point cloud in world coordinates [T, N, 3]
        """
        # Ensure tensors are on the correct device
        points = points.to(self.device)
        extrinsics = extrinsics.to(self.device)
        intrinsics = intrinsics.to(self.device)

        if extrinsics.dim() == 4:
            extrinsics = extrinsics[0]
        if intrinsics.dim() == 4:
            intrinsics = intrinsics[0]

        T, N, _ = points.shape
        
        uv = points[..., :2]
        z = points[..., 2:3]
        
        # Create homogeneous coordinates [u, v, 1]
        uv_hom = torch.cat([uv, torch.ones(T, N, 1, device=self.device)], dim=-1)
        
        # Invert intrinsics for back-projection
        K_inv = torch.inverse(intrinsics) # Shape (T, 3, 3)
        
        # Back-project to camera coordinates
        # (T, N, 3) = (T, N, 3) @ (T, 3, 3).T
        cam_coords_hom = torch.einsum('tnj,tij->tni', uv_hom, K_inv)
        cam_coords = cam_coords_hom * z # Scale by depth
        
        # Add homogeneous coordinate for transformation
        cam_coords_hom4 = torch.cat([cam_coords, torch.ones(T, N, 1, device=self.device)], dim=-1)
        
        # Transform from camera to world coordinates using c2w extrinsics
        # (T, N, 4) = (T, N, 4) @ (T, 4, 4).T
        world_points_hom = torch.einsum('tnj,tij->tni', cam_coords_hom4, extrinsics)
        
        return world_points_hom[..., :3]

    def w2s_vggt(self, world_points, extrinsics, intrinsics, poses=None, override_extrinsics=False):
        """
        Project points from world coordinates to camera view using PyTorch.
        
        Args:
            world_points (torch.Tensor): Point cloud in world coordinates [T, N, 3].
            extrinsics (torch.Tensor): Original camera extrinsic matrices (w2c) [T, 4, 4].
            intrinsics (torch.Tensor): Camera intrinsic matrices [T, 3, 3].
            poses (torch.Tensor, optional): New camera pose matrices (c2w) [T, 4, 4].
            override_extrinsics (bool): If True, `poses` replace `extrinsics`. If False, `poses` are applied on top of `extrinsics`.
            
        Returns:
            torch.Tensor: Point cloud in camera coordinates [T, N, 3] in uvz format.
        """
        # Ensure all inputs are torch tensors on the correct device
        world_points = world_points.to(self.device)
        extrinsics = extrinsics.to(self.device)
        intrinsics = intrinsics.to(self.device)
        if poses is not None:
            poses = poses.to(self.device)

        T, N, _ = world_points.shape

        # Determine the final world-to-camera (w2c) transformation matrix
        if poses is not None:
            new_w2c = torch.inverse(poses)  # New w2c from new c2w poses
            if override_extrinsics:
                w2c_final = new_w2c
            else:
                # Apply new pose on top of original camera: new_w2c @ original_w2c
                w2c_final = torch.einsum('tij,tjk->tik', new_w2c, extrinsics)
        else:
            w2c_final = extrinsics

        # Add homogeneous coordinate to world points
        world_points_hom = torch.cat([world_points, torch.ones(T, N, 1, device=self.device)], dim=-1)
        
        # Transform points from world to camera space
        # (T, N, 4) = (T, N, 4) @ (T, 4, 4).T
        cam_points_hom = torch.einsum('tnj,tij->tni', world_points_hom, w2c_final)
        
        # Get points in camera frame (non-homogeneous)
        cam_points = cam_points_hom[..., :3]
        
        # Extract depth (z-coordinate in camera space)
        z = cam_points[..., 2:3]
        
        # Create a mask for points in front of the camera
        valid_mask = (z > 1e-5).squeeze(-1)
        
        # Project to pixel coordinates
        # Normalize by depth
        cam_points_normalized = cam_points / (z + 1e-8)
        
        # Apply intrinsic matrix
        # (T, N, 3) = (T, N, 3) @ (T, 3, 3).T
        pixel_coords_hom = torch.einsum('tnj,tij->tni', cam_points_normalized, intrinsics)
        
        # Final uv coordinates
        uv = pixel_coords_hom[..., :2]
        
        # Combine uv and z to get final result
        result = torch.cat([uv, z], dim=-1)
        
        # Set invalid points to zero
        result[~valid_mask] = 0
        
        return result

    
    def set_intr(self, K):
        if isinstance(K, np.ndarray):
            K = torch.from_numpy(K)
        self.intr = K.to(self.device)

    def set_extr(self, extr):
        if isinstance(extr, np.ndarray):    
            extr = torch.from_numpy(extr)
        self.extr = extr.to(self.device)

    def rot_poses(self, angle, axis='y'):
        """Generate a single rotation matrix
        
        Args:
            angle (float): Rotation angle in degrees
            axis (str): Rotation axis ('x', 'y', or 'z')
            
        Returns:
            torch.Tensor: Single rotation matrix [4, 4]
        """
        angle_rad = math.radians(angle)
        cos_theta = torch.cos(torch.tensor(angle_rad))
        sin_theta = torch.sin(torch.tensor(angle_rad))
        
        if axis == 'x':
            rot_mat = torch.tensor([
                [1, 0, 0, 0],
                [0, cos_theta, -sin_theta, 0],
                [0, sin_theta, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        elif axis == 'y':
            rot_mat = torch.tensor([
                [cos_theta, 0, sin_theta, 0],
                [0, 1, 0, 0],
                [-sin_theta, 0, cos_theta, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        elif axis == 'z':
            rot_mat = torch.tensor([
                [cos_theta, -sin_theta, 0, 0],
                [sin_theta, cos_theta, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=torch.float32)
        else:
            raise ValueError("Invalid axis value. Choose 'x', 'y', or 'z'.")
            
        return rot_mat.to(self.device)

    def trans_poses(self, dx, dy, dz):
        """
        params:
        - dx: float, displacement along x axis。
        - dy: float, displacement along y axis。
        - dz: float, displacement along z axis。

        ret:
        - matrices: torch.Tensor
        """
        trans_mats = torch.eye(4).unsqueeze(0).repeat(self.frame_num, 1, 1)  # (n, 4, 4)

        delta_x = dx / (self.frame_num - 1)
        delta_y = dy / (self.frame_num - 1)
        delta_z = dz / (self.frame_num - 1)

        for i in range(self.frame_num):
            trans_mats[i, 0, 3] = i * delta_x
            trans_mats[i, 1, 3] = i * delta_y
            trans_mats[i, 2, 3] = i * delta_z

        return trans_mats.to(self.device)
    

    def _look_at(self, camera_position, target_position):
        # look at direction
        direction = target_position - camera_position
        direction /= np.linalg.norm(direction)
        # calculate rotation matrix
        up = np.array([0, 1, 0])
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        rotation_matrix = np.vstack([right, up, direction])
        rotation_matrix = np.linalg.inv(rotation_matrix)
        return rotation_matrix

    def spiral_poses(self, radius, forward_ratio = 0.5, backward_ratio = 0.5, rotation_times = 0.1, look_at_times = 0.5):
        """Generate spiral camera poses
        
        Args:
            radius (float): Base radius of the spiral
            forward_ratio (float): Scale factor for forward motion
            backward_ratio (float): Scale factor for backward motion
            rotation_times (float): Number of rotations to complete
            look_at_times (float): Scale factor for look-at point distance
            
        Returns:
            torch.Tensor: Camera poses of shape [num_frames, 4, 4]
        """
        # Generate spiral trajectory
        t = np.linspace(0, 1, self.frame_num)
        r = np.sin(np.pi * t) * radius * rotation_times
        theta = 2 * np.pi * t
        
        # Calculate camera positions
        # Limit y motion for better floor/sky view
        y = r * np.cos(theta) * 0.3  
        x = r * np.sin(theta)
        z = -r
        z[z < 0] *= forward_ratio
        z[z > 0] *= backward_ratio
        
        # Set look-at target
        target_pos = np.array([0, 0, radius * look_at_times])
        cam_pos = np.vstack([x, y, z]).T
        cam_poses = []
        
        for pos in cam_pos:
            rot_mat = self._look_at(pos, target_pos)
            trans_mat = np.eye(4)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = pos
            cam_poses.append(trans_mat[None])
            
        camera_poses = np.concatenate(cam_poses, axis=0)
        return torch.from_numpy(camera_poses).to(self.device)

    def get_default_motion(self):
        """Parse motion parameters and generate corresponding motion matrices
        
        Supported formats:
        - trans <dx> <dy> <dz> [start_frame] [end_frame]: Translation motion
        - rot <axis> <angle> [start_frame] [end_frame]: Rotation motion
        - spiral <radius> [start_frame] [end_frame]: Spiral motion
        
        Multiple transformations can be combined using semicolon (;) as separator:
        e.g., "trans 0 0 0.5 0 30; rot x 25 0 30; trans 0.1 0 0 30 48"
        
        Note:
            - start_frame and end_frame are optional
            - frame range: 0-49 (will be clamped to this range)
            - if not specified, defaults to 0-49
            - frames after end_frame will maintain the final transformation
            - for combined transformations, they are applied in sequence
            - moving left, up and zoom out is positive in video
        
        Returns:
            torch.Tensor: Motion matrices [num_frames, 4, 4]
        """
        if not isinstance(self.motion_type, str):
            raise ValueError(f'camera_motion must be a string, but got {type(self.motion_type)}')
        
        # Split combined transformations
        transform_sequences = [s.strip() for s in self.motion_type.split(';')]
        
        # Initialize the final motion matrices
        final_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
        
        # Process each transformation in sequence
        for transform in transform_sequences:
            params = transform.lower().split()
            if not params:
                continue
                
            motion_type = params[0]
            
            # Default frame range
            start_frame = 0
            end_frame = 48  # 49 frames in total (0-48)
            
            if motion_type == 'trans':
                # Parse translation parameters
                if len(params) not in [4, 6]:
                    raise ValueError(f"trans motion requires 3 or 5 parameters: 'trans <dx> <dy> <dz>' or 'trans <dx> <dy> <dz> <start_frame> <end_frame>', got: {transform}")
                
                dx, dy, dz = map(float, params[1:4])
                
                if len(params) == 6:
                    start_frame = max(0, min(48, int(params[4])))
                    end_frame = max(0, min(48, int(params[5])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                # Generate current transformation
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
                for frame_idx in range(49):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        current_motion[frame_idx, :3, 3] = torch.tensor([dx, dy, dz], device=self.device) * t
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'rot':
                # Parse rotation parameters
                if len(params) not in [3, 5]:
                    raise ValueError(f"rot motion requires 2 or 4 parameters: 'rot <axis> <angle>' or 'rot <axis> <angle> <start_frame> <end_frame>', got: {transform}")
                
                axis = params[1]
                if axis not in ['x', 'y', 'z']:
                    raise ValueError(f"Invalid rotation axis '{axis}', must be 'x', 'y' or 'z'")
                angle = float(params[2])
                
                if len(params) == 5:
                    start_frame = max(0, min(48, int(params[3])))
                    end_frame = max(0, min(48, int(params[4])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
                for frame_idx in range(49):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        current_angle = angle * t
                        current_motion[frame_idx] = self.rot_poses(current_angle, axis)
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            elif motion_type == 'spiral':
                # Parse spiral motion parameters
                if len(params) not in [2, 4]:
                    raise ValueError(f"spiral motion requires 1 or 3 parameters: 'spiral <radius>' or 'spiral <radius> <start_frame> <end_frame>', got: {transform}")
                
                radius = float(params[1])
                
                if len(params) == 4:
                    start_frame = max(0, min(48, int(params[2])))
                    end_frame = max(0, min(48, int(params[3])))
                    if start_frame > end_frame:
                        start_frame, end_frame = end_frame, start_frame
                
                current_motion = torch.eye(4, device=self.device).unsqueeze(0).repeat(49, 1, 1)
                spiral_motion = self.spiral_poses(radius)
                for frame_idx in range(49):
                    if frame_idx < start_frame:
                        continue
                    elif frame_idx <= end_frame:
                        t = (frame_idx - start_frame) / (end_frame - start_frame)
                        idx = int(t * (len(spiral_motion) - 1))
                        current_motion[frame_idx] = spiral_motion[idx]
                    else:
                        current_motion[frame_idx] = current_motion[end_frame]
                
                # Combine with previous transformations
                final_motion = torch.matmul(final_motion, current_motion)
                
            else:
                raise ValueError(f'camera_motion type must be in [trans, spiral, rot], but got {motion_type}')
        
        return final_motion

class TriangleMeshPipeline(nn.Module):
    def __init__(self, renderer_model=None):
        """
        Initializes the pipeline.
        Args:
            renderer_model (nn.Module, optional): A pre-initialized neural network model for rendering.
                                                  If None, the pipeline will only perform the mesh processing step.
        """
        super().__init__()
        self.renderer = renderer_model

    @staticmethod
    def to_pixel_samples(img):
        """ Convert the image to coord-RGB pairs.
            img: Tensor, (3, H, W)
        """
        coord = TriangleMeshPipeline.make_coord(img.shape[-2:], [[0,1],[0,1]])
        rgb = img.view(3, -1).permute(1, 0)
        return coord, rgb

    @staticmethod
    def make_coord(shape, ranges=None, flatten=True):
        """ Make coordinates at grid centers.
        """
        # print(shape)
        coord_seqs = []
        for i, n in enumerate(shape):
            # print(i,n)
            if ranges is None:
                v0, v1 = -1, 1
            else:
                v0, v1 = ranges[i]
            r = (v1 - v0) / (2 * n)
            seq = v0 + r + (2 * r) * torch.arange(n).float()
            coord_seqs.append(seq)
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
        if flatten:
            ret = ret.view(-1, ret.shape[-1])
        return ret

    def _mesh_process(self, vertices, mask, h, w, gt, frame_idx=0, global_coords=None):
        """
        Processes a single layer for a single frame to generate mesh-based rendering data.
        
        Args:
            vertices (torch.Tensor): Vertex data for the layer, shape (num_vertices, 3) with (x, y, visibility).
            mask (torch.Tensor): Boolean mask for the layer, shape (H, W).
            h (int): Image height.
            w (int): Image width.
            frame_idx (int): The index of the current frame, used for temporal features.
            global_coords (dict, optional): A dictionary with global coordinate system parameters.
            gt_rgb_frame (torch.Tensor, optional): The ground truth RGB frame of shape (H, W, 3).
                                                  If provided, the function will also return the GT colors
                                                  for the pixels inside the final mesh.
            
        Returns:
            dict: A dictionary containing all the data required for the rendering model.
                  Returns None if no visible vertices are found.
        """

        min_x_global = global_coords['min_x']
        min_y_global = global_coords['min_y']
        global_w = global_coords['global_w']
        global_h = global_coords['global_h']
        visible_mask = vertices[:, 2] > -100
        visible_indices = torch.nonzero(visible_mask).squeeze(1)  
        vertices_filtered = vertices[visible_indices]

        delaunay = Delaunay(vertices_filtered.cpu().numpy()[:, :2])
        triangles = delaunay.simplices
        in_mask_triangles_all = torch.tensor([[visible_indices[i] for i in tri] for tri in triangles], device='cpu')

        
        in_mask_triangles_all_vertices = vertices[in_mask_triangles_all]
        in_mask_triangles_all_vertices[:,:,0] /= global_w
        in_mask_triangles_all_vertices[:,:,1] /= global_h
        # Move triangle vertex tensor to CUDA so subsequent operations (coord_inmask on CUDA)
        # operate on the same device and avoid device mismatch errors.
        if not in_mask_triangles_all_vertices.is_cuda:
            in_mask_triangles_all_vertices = in_mask_triangles_all_vertices.cuda()

        gt = (gt/255).astype(np.float32)
        gt = torch.FloatTensor(gt).permute(2, 0, 1)[None]
        coord_c, rgb = self.to_pixel_samples(gt.contiguous())
        coord_c = coord_c.flip(-1)
        coord_c[:, 1] = coord_c[:, 1] * (h / global_h) - (min_y_global / global_h)
        coord_c[:, 0] = coord_c[:, 0] * (w / global_w) - (min_x_global / global_w)
        
        # Move mask to CPU for indexing CPU tensor coord_c
        cpu_mask = mask.flatten().cpu()
        coord_inmask = coord_c[cpu_mask==1]    
        rgb_inmask = rgb[cpu_mask==1]
        
        coord_inmask = torch.as_tensor(coord_inmask).cuda()

        lambda1, lambda2, lambda3 = compute_bary_triangles(in_mask_triangles_all_vertices, coord_inmask)
        batch_size = 10000
        num_points = lambda1.shape[0]
        
        pixel_triangle_indices = torch.full((num_points,), -1, dtype=torch.long, device='cuda')
        
        # Process in batches
        for i in range(0, num_points, batch_size):
            end_idx = min(i + batch_size, num_points)
            batch_lambda1 = lambda1[i:end_idx]
            batch_lambda2 = lambda2[i:end_idx]
            batch_lambda3 = lambda3[i:end_idx]
            
            # Compute valid triangles for this batch
            batch_point_triangle_matrix = (batch_lambda1 >= 0) & (batch_lambda2 >= 0) & (batch_lambda3 >= 0)
            
            # Find first valid triangle for each point (if any)
            batch_valid_points = batch_point_triangle_matrix.any(dim=1)
            valid_indices = torch.nonzero(batch_valid_points).squeeze(1)
            
            if valid_indices.numel() > 0:
                # Only process points that have valid triangles
                batch_indices = batch_point_triangle_matrix[valid_indices].float().argmax(dim=1)
                pixel_triangle_indices[i + valid_indices] = batch_indices
            
            # Free memory
            del batch_lambda1, batch_lambda2, batch_lambda3, batch_point_triangle_matrix
            torch.cuda.empty_cache()
        invalid_pixel_indices = torch.nonzero(pixel_triangle_indices == -1).squeeze(1)  # Ensure 1D
        
        # Handle empty or single invalid pixel cases
        if invalid_pixel_indices.numel() == 0:
            coord_inmask_invalid = torch.zeros((0, 2), device='cuda')
        else:
            coord_inmask_invalid = coord_inmask[invalid_pixel_indices]
            if coord_inmask_invalid.dim() == 1:  # Single point case
                coord_inmask_invalid = coord_inmask_invalid.unsqueeze(0)  # Make it (1, 2)
        
        in_mask_indices = torch.unique(torch.tensor(in_mask_triangles_all).cuda())
        in_mask_vertices = vertices[in_mask_indices]
        
        # Normalize vertices
        in_mask_vertices[:, 0] /= global_w
        in_mask_vertices[:, 1] /= global_h

        triangles_tensor = in_mask_triangles_all_vertices
        in_mask_triangles_centers = torch.mean(triangles_tensor, dim=1)

        # Process invalid pixels only if we have any
        if coord_inmask_invalid.numel() > 0:
            # Calculate distances using broadcasting
            distances = torch.norm(coord_inmask_invalid[:, None, :] - in_mask_triangles_centers[None, :, :2], dim=2)
            nearest_vertex_indices = torch.argmin(distances, dim=1)
        else:
            nearest_vertex_indices = torch.tensor([], device='cuda', dtype=torch.long)

        # Process barycentric coordinates
        barycentric_coords = torch.zeros((pixel_triangle_indices.shape[0], 3), dtype=torch.float32, device='cuda')
        valid_indices = pixel_triangle_indices != -1
        
        
        
        if torch.any(valid_indices):
            barycentric_coords[valid_indices, 0] = lambda1[torch.arange(len(pixel_triangle_indices), device='cuda')[valid_indices], pixel_triangle_indices[valid_indices]]
            barycentric_coords[valid_indices, 1] = lambda2[torch.arange(len(pixel_triangle_indices), device='cuda')[valid_indices], pixel_triangle_indices[valid_indices]]
            barycentric_coords[valid_indices, 2] = lambda3[torch.arange(len(pixel_triangle_indices), device='cuda')[valid_indices], pixel_triangle_indices[valid_indices]]

        if invalid_pixel_indices.numel() > 0:
            selected_triangles = triangles_tensor[nearest_vertex_indices]
            lambda1_invalid, lambda2_invalid, lambda3_invalid = compute_bary_vertices(selected_triangles, coord_inmask_invalid)
            
            
            barycentric_coords[invalid_pixel_indices, 0] = lambda1_invalid
            barycentric_coords[invalid_pixel_indices, 1] = lambda2_invalid
            barycentric_coords[invalid_pixel_indices, 2] = lambda3_invalid

        triangles_all_tensor = torch.tensor(in_mask_triangles_all, dtype=torch.long).cuda()
        pixel_vertices = torch.zeros((pixel_triangle_indices.shape[0], 3), dtype=torch.long, device='cuda')
        in_mask_triangles_all=None
        
        if torch.any(valid_indices):
            valid_idx_list = torch.nonzero(valid_indices).squeeze(1)
            for i in range(0, valid_idx_list.size(0), batch_size):
                end_idx = min(i + batch_size, valid_idx_list.size(0))
                batch_indices = valid_idx_list[i:end_idx]
                # Process this batch
                pixel_vertices[batch_indices] = triangles_all_tensor[pixel_triangle_indices[batch_indices].long()]
                # Free memory
                del batch_indices
                    
        if invalid_pixel_indices.numel() > 0 and nearest_vertex_indices.numel() > 0:
            pixel_vertices[invalid_pixel_indices] = triangles_all_tensor[nearest_vertex_indices]
        
        batch_size_coords = min(batch_size, 5000) 
        total_points = pixel_vertices.shape[0]
        

        pixel_vertice_coords_list = []
        
        for i in range(0, total_points, batch_size_coords):
            end_idx = min(i + batch_size_coords, total_points)
            batch_size_current = end_idx - i
            

            batch_coords_tensor = torch.zeros((batch_size_current, 3, 3), dtype=torch.float32, device='cuda')
            
            batch_vertices = pixel_vertices[i:end_idx] 
            
            batch_coords = vertices[batch_vertices]
            batch_coords_tensor[:, :, :2] = batch_coords[:, :, :2]  # XY coords
            
            batch_coords_tensor[:, :, 0] /= global_w
            batch_coords_tensor[:, :, 1] /= global_h
            batch_coords_tensor[:, :, 2] = frame_idx  # Set the time dimension
            
            pixel_vertice_coords_list.append(batch_coords_tensor)
            

        try:
            pixel_vertice_coords = torch.cat(pixel_vertice_coords_list, dim=0)
        except RuntimeError:
            del pixel_vertice_coords_list
            torch.cuda.empty_cache()
            
            pixel_vertice_coords = torch.zeros((total_points, 3, 3), dtype=torch.float32, device='cpu')  
            for i in range(0, total_points, batch_size_coords):
                end_idx = min(i + batch_size_coords, total_points)
                batch_vertices = pixel_vertices[i:end_idx] 
                batch_coords = vertices[batch_vertices].cpu()  # Move to CPU
                
                pixel_vertice_coords[i:end_idx, :, :2] = batch_coords[:, :, :2]
                pixel_vertice_coords[i:end_idx, :, 0] /= global_w
                pixel_vertice_coords[i:end_idx, :, 1] /= global_h
                pixel_vertice_coords[i:end_idx, :, 2] = frame_idx

            
            pixel_vertice_coords = pixel_vertice_coords.cuda()

        result= {
                'pixel_vertices_coords': pixel_vertice_coords,
                'pixel_vertices_indices': pixel_vertices,
                'barycentric_coords': barycentric_coords,
                'coord_inmask': coord_inmask,
                'gt_rgb': rgb_inmask
            }
        
        return result

   

class WorldSpaceTrianglePipeline(nn.Module):
    """
    A structured pipeline that renders triangles based on 3D world-space features.
    1. Samples features for 3D vertices from a world-space feature grid.
    2. Projects vertices to 2D to form a mesh.
    3. Interpolates vertex features to pixels using barycentric coordinates.
    4. Renders pixel colors using an MLP.
    """
    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary containing all configuration parameters.
                - feat_dim (int): Dimension of the feature vectors in the grid.
                - D (int): Number of linear layers in the MLP.
                - W (int): Width of the linear layers in the MLP.
                - pos_embed (bool): Whether to use positional embedding for world coordinates.
                - pos_freq (int): Number of frequencies for world coordinate embedding.
                - feat_embed (bool): Whether to use positional embedding for sampled features.
                - feat_freq (int): Number of frequencies for feature embedding.
                - grid_size (tuple): Resolution of the grid [D, H, W].
                - world_bbox (tuple): Bounding box of the grid in world space [min_x, min_y, min_z, max_x, max_y, max_z].
        """
        super().__init__()
        self.config = config
        feat_dim = config['feat_dim']
        D = config['D']
        W = config['W']
        self.D = D
        self.W = W
        self.skips = [1,D // 2,D // 4]
        
        # Define the 3D feature grid
        self.world_grid = nn.Parameter(torch.randn(1, feat_dim, *config['grid_size']))
        self.register_buffer('world_bbox', torch.tensor(config['world_bbox']))

        # --- Correctly calculate MLP input dimension ---
        mlp_input_dim = 0
        
        # Feature embedding part
        self.feat_embed = config.get('feat_embed', False)
        if self.feat_embed:
            self.feat_freq = config.get('feat_freq', 9)
            self.embedding_feat = Embedding(self.feat_freq)
            mlp_input_dim += feat_dim * (2 * self.feat_freq + 1)
        else:
            mlp_input_dim += feat_dim

        # Positional embedding part
        self.pos_embed = config.get('pos_embed', False)
        if self.pos_embed:
            self.pos_freq = config.get('pos_freq', 9)
            self.embedding_pos = Embedding(self.pos_freq)
            mlp_input_dim += 3 * (2 * self.pos_freq + 1)
            
        self.mlp_input_dim = mlp_input_dim

        # --- Correctly define MLP layers ---
        layers = []
        for i in range(D):
            if i == 0:
                layer = nn.Linear(self.mlp_input_dim, W)
            elif i in self.skips:
                # Skip connection concatenates original MLP input with intermediate features
                layer = nn.Linear(self.mlp_input_dim + W, W)
            else:
                layer = nn.Linear(W, W)
            layers.append(nn.Sequential(layer, nn.ReLU(True)))
        self.xyz_encoding_layers = nn.ModuleList(layers)
        
        self.rgb_layer = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

    def _normalize_coords(self, world_coords):
        """Normalize world coordinates to the range [-1, 1] vfxbounding box."""
        bbox_min = self.world_bbox[:3].to(world_coords.device)
        bbox_max = self.world_bbox[3:].to(world_coords.device)
        return 2.0 * (world_coords[..., :3] - bbox_min) / (bbox_max - bbox_min) - 1.0

    def _sample_vertex_features(self, norm_coords):
        """
        Samples features for a set of 3D world-space vertices from the feature grid.
        
        Args:
            norm_coords (torch.Tensor): Normalized world coordinates of vertices [N, 3] in range [-1, 1].
        
        Returns:
            torch.Tensor: Sampled features for each vertex [N, feat_dim].
        """
        # grid_sample expects coords in (N, 1, 1, 1, 3) format with (z, y, x) order
        grid_sampler_coords = norm_coords.flip(-1).view(-1, 1, 1, 1, 3)
        
        sampled_features = F.grid_sample(
            self.world_grid.expand(grid_sampler_coords.shape[0], -1, -1, -1, -1),
            grid_sampler_coords,
            align_corners=True
        ).squeeze()
        
        return sampled_features

    def forward(self, data_dict, frame_vertices_3d, image_size, rgb_frame=None):
        """
        Processes pre-computed mesh data for a single frame to generate a rendered output.
        
        Args:
            data_dict (dict): A dictionary containing pre-processed geometric data.
            frame_vertices_3d (torch.Tensor): World-space tracks for the current frame (V, 3).
            image_size (tuple): (H, W).
            rgb_frame (torch.Tensor, optional): Ground truth RGB frame for training (H, W, 3).
        
        Returns:
            torch.Tensor: The rendered RGB values for all pixels in the mask (N_pixels, 3).
        """
        h, w = image_size
        coord_inmask = data_dict["coord_inmask"]
        pixel_vertex_indices = data_dict["pixel_vertices_indices"]
        barycentric_coords = data_dict["barycentric_coords"]

        if coord_inmask.shape[0] == 0:
            return torch.zeros((h, w, 3), device=frame_vertices_3d.device)

        # 1. Sample features for ALL vertices in the current frame from the 3D grid
        all_vertex_features = self._sample_vertex_features(frame_vertices_3d) # (V, feat_dim)

        # Directly gather features for each pixel's vertices using advanced indexing.
        # This avoids slow Python loops and dictionary mapping.
        vertex_features_for_pixels = all_vertex_features[pixel_vertex_indices] # (N_pixels, 3, feat_dim)

        # 2. Interpolate vertex features to get pixel features
        pixel_features = torch.bmm(barycentric_coords.unsqueeze(1), vertex_features_for_pixels).squeeze(1)

        # 3. Interpolate world coordinates and apply embeddings
        vertex_world_coords_for_pixels = frame_vertices_3d[pixel_vertex_indices]
        pixel_world_coords = torch.bmm(barycentric_coords.unsqueeze(1), vertex_world_coords_for_pixels).squeeze(1)
        
        # --- Correctly build MLP input with embeddings ---
        mlp_input_parts = []
        if self.feat_embed:
            mlp_input_parts.append(self.embedding_feat(pixel_features))
        else:
            mlp_input_parts.append(pixel_features)
        
        if self.pos_embed:
            mlp_input_parts.append(self.embedding_pos(pixel_world_coords))
        
        mlp_input = torch.cat(mlp_input_parts, dim=-1)

        # 4. Render RGB using MLP
        output = mlp_input
        for j, layer in enumerate(self.xyz_encoding_layers):
            if j in self.skips:
                output = torch.cat([mlp_input, output], -1)
            output = layer(output)
        rendered_rgb = self.rgb_layer(output)

        return rendered_rgb
