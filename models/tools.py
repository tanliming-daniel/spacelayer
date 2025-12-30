import torch

def sample_points(all_frame_data, num_points, device):
    """
    Sample points from all pre-processed frames.
    
    Args:
        all_frame_data (list): A list of dictionaries, each containing data for a frame.
        num_points (int): The total number of points to sample.
        device: The torch device.
        
    Returns:
        dict: A dictionary containing the sampled data, batched and on the correct device.
    """
    # Calculate total number of available points and points per frame
    points_per_frame = [len(frame_data['coord_inmask']) for frame_data in all_frame_data]
    total_available_points = sum(points_per_frame)
    
    if total_available_points == 0:
        return None

    # Create a single large tensor for frame indices corresponding to each point
    point_frame_indices = torch.cat([
        torch.full((n_pts,), i, dtype=torch.long)
        for i, n_pts in enumerate(points_per_frame)
    ]).to(device)

    # Randomly select point indices
    num_points_to_sample = min(num_points, total_available_points)
    sampled_indices_global = torch.randint(0, total_available_points, (num_points_to_sample,), device=device)

    # Get the corresponding frame index for each sampled point
    sampled_frame_indices = point_frame_indices[sampled_indices_global]

    # Concatenate all data into large tensors first
    all_coords = torch.cat([d['coord_inmask'] for d in all_frame_data], dim=0)
    all_barycentrics = torch.cat([d['barycentric_coords'] for d in all_frame_data], dim=0)
    all_vertex_indices = torch.cat([d['pixel_vertices_indices'] for d in all_frame_data], dim=0)
    all_gt_rgb = torch.cat([d['gt_rgb'] for d in all_frame_data], dim=0).to(device)

    # Gather the sampled data using the global indices
    sampled_data = {
        'coord_inmask': all_coords[sampled_indices_global],
        'barycentric_coords': all_barycentrics[sampled_indices_global],
        'pixel_vertices_indices': all_vertex_indices[sampled_indices_global],
        'gt_rgb': all_gt_rgb[sampled_indices_global],
        'frame_indices': sampled_frame_indices
    }
    
    return sampled_data