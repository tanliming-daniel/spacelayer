import torch
import torch.nn as nn
import torch.nn.functional as F
from .util_2d import Embedding, map_to_initcolorX


import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay
import numpy as np

from .util_2d import Embedding, compute_bary_triangles, compute_bary_vertices

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
        self.skips = [D // 2]
        # 支持 feat_embed
        self.feat_embed = config.get('feat_embed', False)
        self.feat_freq = config.get('feat_freq', 10)
        
        # Define the 3D feature grid
        self.world_grid = nn.Parameter(torch.randn(1, feat_dim, *config['grid_size']))
        self.register_buffer('world_bbox', torch.tensor(config['world_bbox']))

        mlp_input_dim = feat_dim
        if self.feat_embed:
            self.embedding_feat = Embedding(self.feat_freq)
            mlp_input_dim = feat_dim * (2 * self.feat_freq + 1)

        # Setup positional embedding for world coordinates (xyz) of vertices
        if config.get('pos_embed', False):
            self.embedding_pos = Embedding(config['pos_freq'])
            mlp_input_dim += 3 * (2 * config['pos_freq'] + 1)

        # Setup MLP layers for rendering
        layers = []
        for i in range(D):
            if i == 0:
                layer = nn.Linear(mlp_input_dim, W)
            elif i in self.skips:
                layer = nn.Linear(mlp_input_dim + W, W)
            else:
                layer = nn.Linear(W, W)
            layers.append(nn.Sequential(layer, nn.ReLU(True)))
        self.xyz_encoding_layers = nn.ModuleList(layers)
        self.rgb_layer = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

    def _normalize_coords(self, world_coords):
        """Normalize world coordinates to the range [-1, 1] based on the bounding box."""
        bbox_min = self.world_bbox[:3]
        bbox_max = self.world_bbox[3:]
        return 2.0 * (world_coords - bbox_min) / (bbox_max - bbox_min) - 1.0

    def _sample_vertex_features(self, vertices_world_coords):
        """
        Samples features for a set of 3D world-space vertices from the feature grid.
        
        Args:
            vertices_world_coords (torch.Tensor): World coordinates of vertices [N, 3].
        
        Returns:
            torch.Tensor: Sampled features for each vertex [N, feat_dim].
        """
        norm_coords = self._normalize_coords(vertices_world_coords)
        
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
            data_dict (dict): A dictionary containing pre-processed geometric data:
                - "coord_inmask" (torch.Tensor): Screen-space coordinates of pixels to render (N_pixels, 2).
                - "pixel_vertex_indices" (torch.Tensor): Indices of the three vertices for each pixel's triangle (N_pixels, 3).
                - "barycentric_coords" (torch.Tensor): Barycentric coordinates for each pixel (N_pixels, 3).
            frame_vertices_3d (torch.Tensor): World-space tracks for the current frame (V, 3).
            image_size (tuple): (H, W).
            rgb_frame (torch.Tensor, optional): Ground truth RGB frame for training (H, W, 3).
        
        Returns:
            dict: A dictionary containing the rendered frame and loss if in training mode.
        """
        h, w = image_size
        coord_inmask = data_dict["coord_inmask"]
        pixel_vertex_indices = data_dict["pixel_vertex_indices"]
        barycentric_coords = data_dict["barycentric_coords"]

        if coord_inmask.shape[0] == 0:
            return {
                "rendered_frame": torch.zeros((h, w, 3), device=frame_vertices_3d.device),
                "loss": 0.0
            }

        # 1. Sample features for vertices from the 3D grid
        unique_vertex_indices = torch.unique(pixel_vertex_indices)
        unique_vertex_world_coords = frame_vertices_3d[unique_vertex_indices]
        unique_vertex_features = self._sample_vertex_features(unique_vertex_world_coords)
        # 支持 feat_embed
        if self.feat_embed:
            unique_vertex_features = self.embedding_feat(unique_vertex_features)
        # Create a mapping from original index to feature index
        mapper = {orig_idx.item(): i for i, orig_idx in enumerate(unique_vertex_indices)}
        mapped_indices = torch.tensor([mapper[idx.item()] for idx in pixel_vertex_indices.view(-1)], device=frame_vertices_3d.device).view(-1, 3)
        vertex_features_for_pixels = unique_vertex_features[mapped_indices] # (N_pixels, 3, feat_dim)

        # 2. Interpolate vertex features to get pixel features
        pixel_features = torch.bmm(barycentric_coords.unsqueeze(1), vertex_features_for_pixels).squeeze(1)

        # 3. Get positional embedding for interpolated world coordinates
        vertex_world_coords_for_pixels = frame_vertices_3d[pixel_vertex_indices]
        pixel_world_coords = torch.bmm(barycentric_coords.unsqueeze(1), vertex_world_coords_for_pixels).squeeze(1)
        
        mlp_input = pixel_features
        if self.config.get('pos_embed', False):
            pos_embed = self.embedding_pos(pixel_world_coords)
            mlp_input = torch.cat([mlp_input, pos_embed], dim=-1)

        # 4. Render RGB using MLP
        output = mlp_input
        for j, layer in enumerate(self.xyz_encoding_layers):
            if j in self.skips:
                output = torch.cat([mlp_input, output], -1)
            output = layer(output)
        
        rendered_rgb = self.rgb_layer(output)
        
        # 5. Create final image and calculate loss
        frame_render = torch.zeros((h, w, 3), device=frame_vertices_3d.device)
        coord_inmask_int = coord_inmask.long()
        frame_render[coord_inmask_int[:, 1], coord_inmask_int[:, 0]] = rendered_rgb
        
        loss = 0.0
        if rgb_frame is not None:
            gt_rgb = rgb_frame[coord_inmask_int[:, 1], coord_inmask_int[:, 0]]
            loss = F.mse_loss(rendered_rgb, gt_rgb)

        return {
            "rendered_frame": frame_render,
            "loss": loss
        }



class ShapeLayerX(nn.Module):
    def __init__(self, feat_dim, D, W, feat_embed, feat_freq, pos_embed, pos_freq, time_embed, time_freq, mapper):
        super(ShapeLayerX, self).__init__()

        self.feat_dim = feat_dim
        self.D = D
        self.W = W
        self.mapper = mapper
        self.skips = [1, D//2+1, D//4]
        self.pos_embed = pos_embed
        self.feat_embed = feat_embed
        self.time_embed = time_embed

        if self.feat_embed:
            input_dim = feat_dim * (2 * feat_freq + 1)
            self.embedding_feat = Embedding(feat_freq)
        else:
            input_dim = feat_dim

        if self.pos_embed:
            input_dim += 9 * (2 * pos_freq + 1)
            self.embedding_xyz = Embedding(pos_freq)

        if self.time_embed:
            input_dim += 2 * time_freq + 1
            self.embedding_time = Embedding(time_freq)

        if self.D == 1:
            layer = nn.Sequential(nn.Linear(input_dim, W), nn.ReLU(True))
            setattr(self, f"latent_encoding_1", layer)
        else:
            for i in range(D):
                if i == 0:
                    layer = nn.Linear(input_dim, W)
                elif i in self.skips:
                    layer = nn.Linear(input_dim + W, W)
                else:
                    layer = nn.Linear(W, W)
                layer = nn.Sequential(layer, nn.ReLU(True))
                setattr(self, f"latent_encoding_{i+1}", layer)

        self.rgb_layer = nn.Sequential(
            nn.Linear(W, 3),
            nn.Sigmoid()
        )

        self.grid_size = 64  # Size of the grid feature map
        self.grid_features = nn.Parameter(torch.randn(1, feat_dim, self.grid_size, self.grid_size) * 0.1)


        self.mu = 0
        self.sigma = 0.0001

    def sample_grid_features(self, norm_coords):
        """
        Sample features from grid using normalized coordinates
        Args:
            norm_coords: (N, 2) normalized coordinates in [-1, 1] range
        Returns:
            sampled_features: (N, feat_dim)
        """
        grid = norm_coords.view(-1, 1, 1, 2)

        # Sample features from grid
        sampled_features = F.grid_sample(
            self.grid_features,
            grid,
            align_corners=True
        )

        return sampled_features.squeeze(-1).squeeze(-1)  # (N, feat_dim)

    def forward(self, pixel_vertices, pixel_vertice_coords, coord_inmask, barycentric_coords, frame_idx=None, edit=False, segs_sum=None, scale_factor=1, src_ind=0, edit_ind=1, edit_tex=False, tex_alpha=1.0, alpha_1=0.3, alpha_2=0.7):
        
        # Sample features for all unique vertices from the grid
        unique_vertices, inverse_indices = torch.unique(pixel_vertices, return_inverse=True)
        
        # Assuming norm_coords are the normalized coordinates for the unique_vertices
        # This part needs to be connected to the data pipeline
        # For now, let's assume a function `get_norm_coords(unique_vertices)` exists
        # norm_coords = get_norm_coords(unique_vertices) 
        # sampled_feats = self.sample_grid_features(norm_coords)
        
        # Placeholder for `feats` - this should be the sampled features
        # In a real scenario, you'd pass the normalized coordinates of vertices
        # to `sample_grid_features`
        feats = self.grid_features.squeeze(0).permute(1,2,0).reshape(-1, self.feat_dim)
        # A more realistic placeholder:
        # feats = torch.randn(unique_vertices.max() + 1, self.feat_dim, device=pixel_vertices.device)


        vertex_features = feats[pixel_vertices]
        pixel_features = torch.bmm(barycentric_coords.unsqueeze(1),vertex_features).squeeze(1)
        
        if self.feat_embed:
            pixel_features = self.embedding_feat(pixel_features)

        if self.pos_embed:
            xyz_features = self.embedding_xyz(pixel_vertice_coords.reshape(-1, 9))
            pixel_features = torch.cat([pixel_features, xyz_features], -1)

        if self.time_embed and frame_idx is not None:
            time_features = self.embedding_time(frame_idx)
            pixel_features = torch.cat([pixel_features, time_features], -1)

        output = pixel_features
        if self.D == 1:
            output = getattr(self, f"latent_encoding_1")(output)
        else:
            for i in range(self.D):
                if i in self.skips:
                    output = torch.cat([pixel_features, output], -1)
                output = getattr(self, f"latent_encoding_{i+1}")(output)

        if edit_tex:
            feats_src = output[segs_sum[src_ind]:segs_sum[src_ind+1]]
            feats_dst = output[segs_sum[edit_ind]:segs_sum[edit_ind+1]]

            points_src = coord_inmask[segs_sum[src_ind]:segs_sum[src_ind+1]]
            points_dst = coord_inmask[segs_sum[edit_ind]:segs_sum[edit_ind+1]]

            min_vals_src = torch.min(points_src, dim=0).values
            max_vals_src = torch.max(points_src, dim=0).values

            min_vals_dst = torch.min(points_dst, dim=0).values
            max_vals_dst = torch.max(points_dst, dim=0).values

            normalized_points_src = (points_src - min_vals_src) / (max_vals_src - min_vals_src)
            normalized_points_dst = (points_dst - min_vals_dst) / (max_vals_dst - min_vals_dst)

            normalized_points_src = 2 * normalized_points_src - 1
            normalized_points_dst = 2 * normalized_points_dst - 1

            # Update grid features instead of directly modifying the output features
            with torch.no_grad():
                # Get corresponding grid coordinates for dst points
                grid_coords_dst = normalized_points_dst[..., :2].unsqueeze(0).unsqueeze(0)

                # Sample source features at these coordinates
                src_features = F.grid_sample(
                    self.grid_features,
                    grid_coords_dst,
                    mode='bilinear',
                    padding_mode='border',
                    align_corners=True
                ).squeeze().t()

                # Update grid features
                if src_features.ndim == 1:
                    src_features = src_features.unsqueeze(0)

                # This is a simplified update - you might want a more sophisticated approach
                self.grid_features.data += (src_features.mean(0) - self.grid_features.data.mean((2,3))) * tex_alpha

        return self.rgb_layer(output)


class WorldSpaceRenderer(nn.Module):
    """
    A renderer that operates on a 3D world-space feature grid.
    It takes data from TriangleMeshPipeline, calculates the world position
    for each pixel, samples a 3D grid, and renders the color.
    """
    def __init__(self, feat_dim, D, W, embed_config, grid_config):
        """
        Args:
            feat_dim (int): Dimension of the feature vectors in the grid.
            D (int): Number of linear layers in the MLP.
            W (int): Width of the linear layers in the MLP.
            embed_config (dict): Configuration for positional and view direction embeddings.
                - pos_embed (bool): Whether to use positional embedding for world coordinates.
                - pos_freq (int): Number of frequencies for world coordinate embedding.
                - view_embed (bool): Whether to use view direction embedding.
                - view_freq (int): Number of frequencies for view direction embedding.
            grid_config (dict): Configuration for the 3D world grid.
                - grid_size (list or tuple): Resolution of the grid [D, H, W].
                - world_bbox (list or tuple): Bounding box of the grid in world space [min_x, min_y, min_z, max_x, max_y, max_z].
        """
        super().__init__()
        self.D = D
        self.W = W
        self.skips = [D // 2]
        self.embed_config = embed_config

        # Define the 3D feature grid
        self.world_grid = nn.Parameter(torch.randn(1, feat_dim, *grid_config['grid_size']))
        self.register_buffer('world_bbox', torch.tensor(grid_config['world_bbox']))

        input_dim = feat_dim

        # Setup positional embedding for world coordinates (xyz)
        if embed_config['pos_embed']:
            self.embedding_pos = Embedding(embed_config['pos_freq'])
            input_dim += 3 * (2 * embed_config['pos_freq'] + 1)

        # Setup MLP layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(input_dim, W)
            elif i in self.skips:
                layer = nn.Linear(input_dim + W, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)

        # Setup view direction embedding and final layers
        if embed_config['view_embed']:
            self.embedding_view = Embedding(embed_config['view_freq'])
            self.feature_layer = nn.Linear(W, W)
            view_input_dim = W + 3 * (2 * embed_config['view_freq'] + 1)
            self.rgb_layer = nn.Sequential(nn.Linear(view_input_dim, W // 2), nn.ReLU(True), nn.Linear(W // 2, 3), nn.Sigmoid())
        else:
            self.rgb_layer = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

    def _normalize_coords(self, world_coords):
        """Normalize world coordinates to the range [-1, 1] based on the bounding box."""
        bbox_min = self.world_bbox[:3]
        bbox_max = self.world_bbox[3:]
        return 2.0 * (world_coords - bbox_min) / (bbox_max - bbox_min) - 1.0

    def forward(self, data_dict):
        """
        Render pixels based on the data provided by the TriangleMeshPipeline.

        Args:
            data_dict (dict): A dictionary containing:
                - pixel_world_coords (torch.Tensor): World coordinates for each pixel [N, 3].
                - view_dirs (torch.Tensor, optional): View direction for each pixel [N, 3].
                - gt_rgb (torch.Tensor, optional): Ground truth RGB for training [N, 3].

        Returns:
            dict: A dictionary containing:
                - rgb (torch.Tensor): The rendered RGB colors [N, 3].
                - loss (torch.Tensor, optional): The MSE loss if gt_rgb is provided.
        """
        world_coords = data_dict['pixel_world_coords']

        # 1. Normalize world coordinates for grid sampling
        norm_coords = self._normalize_coords(world_coords)

        # 2. Sample features from the 3D world grid
        # grid_sample expects coords in (N, 1, 1, 1, 3) format with (z, y, x) order
        grid_sampler_coords = norm_coords.flip(-1).view(-1, 1, 1, 1, 3)
        sampled_features = F.grid_sample(
            self.world_grid.expand(grid_sampler_coords.shape[0], -1, -1, -1, -1),
            grid_sampler_coords,
            align_corners=True
        ).squeeze()

        # 3. Apply positional embedding
        if self.embed_config['pos_embed']:
            pos_embed = self.embedding_pos(world_coords)
            mlp_input = torch.cat([sampled_features, pos_embed], dim=-1)
        else:
            mlp_input = sampled_features

        # 4. Pass through MLP
        output = mlp_input
        for i in range(self.D):
            if i in self.skips:
                output = torch.cat([mlp_input, output], -1)
            output = getattr(self, f"xyz_encoding_{i+1}")(output)

        # 5. Apply view-dependent effects and get RGB
        if self.embed_config['view_embed']:
            if 'view_dirs' not in data_dict:
                raise ValueError("View directions are required when view_embed is True.")

            features = self.feature_layer(output)
            view_dirs_embed = self.embedding_view(data_dict['view_dirs'])

            rgb_input = torch.cat([features, view_dirs_embed], dim=-1)
            rgb = self.rgb_layer(rgb_input)
        else:
            rgb = self.rgb_layer(output)

        # 6. Prepare output dictionary and compute loss for training
        result = {'rgb': rgb}
        if 'gt_rgb' in data_dict:
            result['loss'] = F.mse_loss(rgb, data_dict['gt_rgb'])

        return result