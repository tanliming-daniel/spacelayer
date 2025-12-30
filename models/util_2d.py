import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from scipy.spatial import Delaunay
import numpy as np

def get_new_sparse_points(xy,          # 现有点 (N,2), [x,y]
                          grid_size,   # 原始 grid_size
                          img_shape,   # (H,W)
                          frame_idx,   # 要附加的帧索引，如 video_tensor.shape[0]-1
                          min_points_per_cell=1,
                          extra_points=0,
                          device="cuda",
                          radius=None):
    """
    根据现有点的分布检测稀疏格并补点
    返回: (N_new, 3) [frame_idx, x, y]
    
    优化说明：
    1. 增加小区域空白检测机制
    2. 对所有区域采用统一的点间距标准
    3. 动态调整补点位置策略
    """
    H, W = img_shape
    # 导入 get_points_on_a_grid
    from submodules.SpaTrackerV2.models.SpaTrackV2.models.utils import get_points_on_a_grid
    # 生成 grid_size x grid_size 的中心点 (1, grid_size*grid_size, 2)
    grid_centers = get_points_on_a_grid(grid_size, (H, W), device=device).reshape(-1, 2)

    # 默认半径：格点间距的一半
    if radius is None:
        if grid_size > 1:
            r_x = (W - W // 64 * 2) / (grid_size - 1) / 2
            r_y = (H - H // 64 * 2) / (grid_size - 1) / 2
            radius = float(min(r_x, r_y))
        else:
            radius = min(H, W) / 4

    new_pts = []
    # 用于统计所有已存在的点（包括新补的）
    all_xy = xy.clone() if xy.numel() > 0 else torch.empty((0,2), device=device)
    
    # 计算平均点密度，用于检测稀疏区域
    avg_density = None
    if all_xy.numel() > 0:
        # 计算所有点之间的平均距离作为密度指标
        dist_matrix = torch.cdist(all_xy, all_xy)
        # 排除自身距离（对角线）和使用top-k获取最近邻距离
        k = min(5, all_xy.shape[0]-1)  # 取5个最近邻或所有可用点
        if k > 0:
            # 获取每个点的k个最近邻距离并计算平均值
            nearest_dists, _ = torch.topk(dist_matrix, k+1, dim=1, largest=False, sorted=True)
            nearest_dists = nearest_dists[:, 1:]  # 排除自身距离
            avg_density = nearest_dists.mean()
    
    # 定义小区域判定阈值：使用固定比例的半径值
    small_region_radius = radius * 0.5
    
    for i, center in enumerate(grid_centers):
        # 统计所有点中距离center小于radius的数量
        n_exist = 0
        neighbor_points = torch.empty((0, 2), device=device)
        
        if all_xy.numel() > 0:
            dists = torch.norm(all_xy - center[None, :], dim=1)
            n_exist = (dists < radius).sum().item()
            # 记录该区域的邻近点
            neighbor_points = all_xy[dists < radius * 2]
        
        if n_exist < min_points_per_cell:
            # 计算该区域需要补多少点
            n_add = (min_points_per_cell - n_exist) + extra_points
            
            # 判断是否为小区域空白
            is_small_region = False
            if neighbor_points.shape[0] > 0:
                # 计算邻近点形成的边界框大小
                min_x, max_x = neighbor_points[:, 0].min(), neighbor_points[:, 0].max()
                min_y, max_y = neighbor_points[:, 1].min(), neighbor_points[:, 1].max()
                region_width = max_x - min_x
                region_height = max_y - min_y
                
                # 如果边界框较小，判定为小区域
                if region_width < small_region_radius * 2 and region_height < small_region_radius * 2:
                    is_small_region = True
            
            # 根据区域类型采用不同的补点位置策略
            if is_small_region and neighbor_points.shape[0] > 0:
                # 小区域策略：更精确地在空白处补点
                # 计算邻近点的中心点作为空白区域参考点
                neighbor_center = neighbor_points.mean(dim=0)
                # 从center向远离neighbor_center的方向移动一定距离作为新中心点
                direction = center - neighbor_center
                if torch.norm(direction) > 1e-5:
                    direction = direction / torch.norm(direction)
                    small_center = center + direction * radius * 0.3
                    # 在小中心附近添加点
                    pts = small_center[None, :].repeat(n_add, 1)
                else:
                    # 如果方向不明确，仍然使用原中心点
                    pts = center[None, :].repeat(n_add, 1)
            else:
                # 普通区域策略：在格点中心补点
                pts = center[None, :].repeat(n_add, 1)
            
            # 所有区域使用相同的高斯扰动
            noise = torch.randn_like(pts) * (radius * 0.01)
            pts = pts + noise
            
            # 确保点在图像范围内
            pts[:, 0].clamp_(0, W-1)
            pts[:, 1].clamp_(0, H-1)
            
            # 检查新点与all_xy的距离，所有区域使用相同的距离阈值
            if all_xy.numel() > 0:
                dists_all = torch.cdist(pts, all_xy)
                mask = (dists_all > radius * 1.8).all(dim=1)
                pts = pts[mask]
            
            # 检查新点之间也不过近，所有区域使用相同的内部点间距
            if pts.shape[0] > 1:
                keep = [0]
                for idx in range(1, pts.shape[0]):
                    if torch.all(torch.norm(pts[idx] - pts[keep], dim=1) > (radius * 0.3)):
                        keep.append(idx)
                pts = pts[keep]
            
            # 添加有效的新点
            if pts.shape[0] > 0:
                fr = torch.full((pts.shape[0], 1), frame_idx, device=device, dtype=torch.float32)
                new_pts.append(torch.cat([fr, pts], dim=1))
                all_xy = torch.cat([all_xy, pts], dim=0)
    
    # 合并所有新点
    if new_pts:
        new_pts = torch.cat(new_pts, dim=0)
    else:
        new_pts = torch.empty((0, 3), device=device)
    
    return new_pts

def compute_loss(pixel_colors_pred, rgb_inmask):
    """Compute loss between predicted colors and ground truth rgb_inmask."""
    loss = torch.mean((pixel_colors_pred - rgb_inmask) ** 2)
    return loss

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

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:], [[0,1],[0,1]])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb 

BARYCENTRIC_EPSILON = 1e-8

def compute_bary_vertices(vertices, points):
    """
    Calculate barycentric coordinates for points with respect to triangles using PyTorch tensors.
    
    Args:
        vertices: Triangle vertex coordinates, shape (N, 3, 3)
        points: Points to check, shape (N, 2)
        
    Returns:
        Three tensors, each of shape (N,), representing barycentric coordinates
    """
    # Ensure inputs are PyTorch tensors on the right device
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, device='cuda')
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, device='cuda')
        
    A = vertices[:, 0]  # First vertex
    B = vertices[:, 1]  # Second vertex
    C = vertices[:, 2]  # Third vertex
    
    # Calculate barycentric coordinates using the same area-based method as util.py
    # Total triangle area calculation
    denominator = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    
    # Add epsilon to avoid division by zero
    denominator = torch.where(torch.abs(denominator) < BARYCENTRIC_EPSILON,
                            torch.ones_like(denominator) * BARYCENTRIC_EPSILON,
                            denominator)
    
    # Lambda1 (ratio of subtriangle area to total area)
    lambda1 = ((B[:, 1] - C[:, 1]) * (points[:, 0] - C[:, 0]) + 
              (C[:, 0] - B[:, 0]) * (points[:, 1] - C[:, 1])) / denominator
    
    # Lambda2 (ratio of subtriangle area to total area)
    lambda2 = ((C[:, 1] - A[:, 1]) * (points[:, 0] - C[:, 0]) + 
              (A[:, 0] - C[:, 0]) * (points[:, 1] - C[:, 1])) / denominator
    
    # Lambda3 (remaining area ratio)
    lambda3 = 1.0 - lambda1 - lambda2
    
    # Check for invalid coordinates
    invalid_coords = torch.isnan(lambda1) | torch.isnan(lambda2) | torch.isnan(lambda3)
    default_value = 1.0 / 3.0  # Equal weights for invalid cases
    
    lambda1 = torch.where(invalid_coords, torch.full_like(lambda1, default_value), lambda1)
    lambda2 = torch.where(invalid_coords, torch.full_like(lambda2, default_value), lambda2)
    lambda3 = torch.where(invalid_coords, torch.full_like(lambda3, default_value), lambda3)
    
    return lambda1, lambda2, lambda3

def compute_bary_triangles(triangles, points):
    """
    Calculate barycentric coordinates for points with respect to triangles using PyTorch tensors.
    Process in batches to reduce memory usage.
    
    Args:
        triangles: Triangle vertex coordinates, shape (M, 3, 3)
        points: Points to check, shape (N, 2)
        
    Returns:
        Three tensors of shape (N, M) representing barycentric coordinates
    """
    # Ensure inputs are PyTorch tensors on the right device
    if not isinstance(triangles, torch.Tensor):
        triangles = torch.tensor(triangles, device=points.device)
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, device=triangles.device)
    
    # Get dimensions
    num_points = points.shape[0]
    num_triangles = triangles.shape[0]
    
    # Process in batches to reduce memory usage
    batch_size = 5000  # Adjust based on available memory
    
    # Pre-allocate output tensors
    lambda1 = torch.zeros((num_points, num_triangles), device=points.device)
    lambda2 = torch.zeros((num_points, num_triangles), device=points.device)
    lambda3 = torch.zeros((num_points, num_triangles), device=points.device)
    
    # Extract triangle vertices
    A = triangles[:, 0]  # First vertex
    B = triangles[:, 1]  # Second vertex
    C = triangles[:, 2]  # Third vertex
    
    # Calculate denominator (triangle area related) - only depends on triangles
    denominator = (B[:, 1] - C[:, 1]) * (A[:, 0] - C[:, 0]) + (C[:, 0] - B[:, 0]) * (A[:, 1] - C[:, 1])
    
    # Add epsilon to avoid division by zero
    denominator = torch.where(torch.abs(denominator) < BARYCENTRIC_EPSILON,
                            torch.ones_like(denominator) * BARYCENTRIC_EPSILON,
                            denominator)
    
    # Process points in batches
    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        batch_points = points[i:end_idx]
        
        # Calculate lambda1 for this batch
        batch_lambda1 = ((B[:, 1] - C[:, 1]) * (batch_points[:, None, 0] - C[:, 0]) + 
                      (C[:, 0] - B[:, 0]) * (batch_points[:, None, 1] - C[:, 1])) / denominator
        
        # Calculate lambda2 for this batch
        batch_lambda2 = ((C[:, 1] - A[:, 1]) * (batch_points[:, None, 0] - C[:, 0]) + 
                      (A[:, 0] - C[:, 0]) * (batch_points[:, None, 1] - C[:, 1])) / denominator
        
        # Calculate lambda3 for this batch
        batch_lambda3 = 1.0 - batch_lambda1 - batch_lambda2
        
        # Check for invalid coordinates
        invalid_coords = torch.isnan(batch_lambda1) | torch.isnan(batch_lambda2) | torch.isnan(batch_lambda3)
        default_value = 1.0 / 3.0  # Equal weights for invalid cases
        
        # Handle invalid values
        batch_lambda1 = torch.where(invalid_coords, torch.full_like(batch_lambda1, default_value), batch_lambda1)
        batch_lambda2 = torch.where(invalid_coords, torch.full_like(batch_lambda2, default_value), batch_lambda2)
        batch_lambda3 = torch.where(invalid_coords, torch.full_like(batch_lambda3, default_value), batch_lambda3)
        
        # Store results
        lambda1[i:end_idx] = batch_lambda1
        lambda2[i:end_idx] = batch_lambda2
        lambda3[i:end_idx] = batch_lambda3
        
        # Free memory
        del batch_lambda1, batch_lambda2, batch_lambda3, invalid_coords

    
    return lambda1, lambda2, lambda3

def mesh_processX(vertices, mask, rgb, coord_c, h, w, start_vertice, visibles=None, frame_idx=None):
    """
    Process the deformed vertices with visibility handling. First filters by visibility, then performs triangulation.
    """
    if visibles is not None:
        visible_indices = torch.nonzero(torch.tensor(visibles)).squeeze()
        vertices_filtered = vertices[visible_indices]
    else:
        vertices_filtered = vertices
        visible_indices = torch.arange(len(vertices))        
        
    delaunay = Delaunay(vertices_filtered.cpu().numpy()[:, :2])
    triangles = delaunay.simplices
    in_mask_triangles_all = torch.tensor([[visible_indices[i] for i in tri] for tri in triangles], device='cpu')
    
    in_mask_triangles_all_vertices = vertices[in_mask_triangles_all]
    in_mask_triangles_all_vertices[:,:,0] /= w
    in_mask_triangles_all_vertices[:,:,1] /= h

    coord_inmask = coord_c[mask.flatten()==0]
    rgb_inmask = rgb[mask.flatten()==0]
    coord_inmask = torch.as_tensor(coord_inmask).cuda()

    lambda1, lambda2, lambda3 = compute_bary_triangles(in_mask_triangles_all_vertices, coord_inmask)
    batch_size = 10000
    num_points = lambda1.shape[0]
    
    pixel_triangle_indices = torch.full((num_points,), -1, dtype=torch.long, device='cuda')
    
    for i in range(0, num_points, batch_size):
        end_idx = min(i + batch_size, num_points)
        batch_lambda1 = lambda1[i:end_idx]
        batch_lambda2 = lambda2[i:end_idx]
        batch_lambda3 = lambda3[i:end_idx]
        
        # Compute valid for this batch
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
        if coord_inmask_invalid.dim() == 1:  
            coord_inmask_invalid = coord_inmask_invalid.unsqueeze(0)  # Make it (1, 2)
    
    in_mask_indices = torch.unique(torch.tensor(in_mask_triangles_all).cuda())
    in_mask_vertices = vertices[in_mask_indices]
    
    in_mask_vertices[:, 0] /= w
    in_mask_vertices[:, 1] /= h

    triangles_tensor = in_mask_triangles_all_vertices
    in_mask_triangles_centers = torch.mean(triangles_tensor, dim=1)

    if coord_inmask_invalid.numel() > 0:
        distances = torch.norm(coord_inmask_invalid[:, None, :] - in_mask_triangles_centers[None, :, :2], dim=2)
        nearest_vertex_indices = torch.argmin(distances, dim=1)
    else:
        nearest_vertex_indices = torch.tensor([], device='cuda', dtype=torch.long)

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
    
    if torch.any(valid_indices):
        pixel_vertices[valid_indices] = triangles_all_tensor[pixel_triangle_indices[valid_indices].long()] + start_vertice
    
    if invalid_pixel_indices.numel() > 0 and nearest_vertex_indices.numel() > 0:
        pixel_vertices[invalid_pixel_indices] = triangles_all_tensor[nearest_vertex_indices.long()] + start_vertice
    

    num_vertices = vertices.shape[0]
    
    pixel_vertice_coords = vertices[pixel_vertices - start_vertice][:, :, :2]
    
    pixel_vertice_coords[:, :, 0] /= w
    pixel_vertice_coords[:, :, 1] /= h
    pixel_vertice_coords = torch.cat([pixel_vertice_coords, torch.ones_like(pixel_vertice_coords[:,:,:1]) * frame_idx], dim=2) 

    return vertices, rgb_inmask, mask, in_mask_vertices, in_mask_triangles_all, coord_inmask, pixel_vertices, pixel_vertice_coords, barycentric_coords, num_vertices


def process_single_frame(
        frame_idx, 
        frame_deformed_vertices_list,
        device, 
        video_segment, 
        gt_frame,
        args,
        visible_list=None,
        delete_mode=False):
    """Process a single frame's deformed points for ShapeLayer input."""
    
    h, w = args.img_h, args.img_w
    global min_x_global, min_y_global, global_h, global_w
    
    gt = gt_frame
    gt = (gt/255).astype(np.float32)
    gt = torch.FloatTensor(gt).permute(2, 0, 1)[None].to(device)
    coord_c, rgb = to_pixel_samples(gt.contiguous())
    coord_c = coord_c.flip(-1)
    coord_c[:, 1] = coord_c[:, 1] * (h / global_h) - (min_y_global / global_h)
    coord_c[:, 0] = coord_c[:, 0] * (w / global_w) - (min_x_global / global_w)    
    start_vertice = 0
    for layer_idx , _ in enumerate(frame_deformed_vertices_list):
        if delete_mode and layer_idx != 0:
            continue
        if_dilate = layer_idx==1
        mask=~base_mask
        
        
        # For layer 0, use the inverse of mask; for others, use mask as is
        if layer_idx == 0:
                mask = 1-mask
            

        frame_deformed_vertices = frame_deformed_vertices_list[layer_idx]     
        vertices_visibility = visible_list[layer_idx][frame_idx,:]
        processed_data = mesh_processX(
            frame_deformed_vertices,
            mask,
            rgb,
            coord_c,
            global_h, global_w,
            start_vertice=start_vertice,
            visibles=vertices_visibility,
            frame_idx=frame_idx/30
        )
        


    return frame_data_per_layer

BARYCENTRIC_EPSILON = 1e-8

class Embedding(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        # self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        # self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)