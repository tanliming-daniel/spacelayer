import torch
import numpy as np
import os
from models.pipeline import CameraMotionGenerator

def main():
    # Load data
    data_path = "/mnt/data-2/bob/spacelayer/results/car-turn_background_points/car-turn_background_tensors.pt"
    data = torch.load(data_path, map_location='cpu')

    track3d_pred = data['track3d_pred']  # (T, N, 6) camera coords
    c2w_traj = data['c2w_traj']  # (T, 4, 4)
    track2d_pred = data['track2d_pred']  # (T, N, 3) xyv
    intrs = data['intrs']  # (T, 3, 3)
    extrs = data['extrs']  # (T, 3, 4) w2c

    T, N, _ = track3d_pred.shape
    print("c2w_traj.shape", c2w_traj.shape)
    print("track3d_pred.shape", track3d_pred.shape)
    print("T,N", T, N)

    # --- Final Hypothesis: Test the round-trip consistency of s2w and w2s ---
    print("Final Hypothesis: Testing round-trip consistency of s2w_vggt and w2s_vggt.")

    # Set up CameraMotionGenerator
    cam_gen = CameraMotionGenerator(motion_type='trans 0 0 0', frame_num=T, device='cpu')
    cam_gen.set_intr(intrs)

    # The c2w_traj should be the correct c2w matrices for s2w transformation
    c2w_matrices = c2w_traj

    # The w2c matrices for w2s transformation should be the inverse of c2w
    w2c_matrices = torch.inverse(c2w_traj)

    # 1. Convert screen coordinates (track2d_pred) to world coordinates
    # s2w_vggt needs depth, which we can get from the z-component of track3d_pred in camera space.
    # Let's assume track3d_pred's z is the depth.
    points_s2w_input = torch.cat([track2d_pred[..., :2], track3d_pred[..., 2:3]], dim=-1)
    
    world_points = cam_gen.s2w_vggt(
        points=points_s2w_input.float(),
        extrinsics=c2w_matrices, # s2w expects c2w matrices
        intrinsics=intrs
    )

    # 2. Project world coordinates back to screen coordinates
    projected_uvz = cam_gen.w2s_vggt(
        world_points=world_points.float(),
        extrinsics=w2c_matrices, # w2s expects w2c matrices
        intrinsics=intrs,
        poses=None,
        override_extrinsics=False
    )
    projected_uvz = projected_uvz.cpu().numpy()  # convert to numpy

    # Compare with original track2d_pred
    track2d_numpy = track2d_pred.numpy()  # (T, N, 3)

    # --- Create a mask for valid points (depth > 0) ---
    valid_depth_mask = track3d_pred[..., 2].numpy() > 1e-5
    
    if not np.any(valid_depth_mask):
        print("Error: No points with positive depth found.")
        return

    # Compare only the xy coordinates of valid points
    diff = np.abs(projected_uvz[..., :2] - track2d_numpy[..., :2])
    
    # Apply the mask before calculating statistics
    valid_diff = diff[valid_depth_mask]
    max_diff = np.max(valid_diff)
    mean_diff = np.mean(valid_diff)

    print(f"Max difference: {max_diff:.3f}")
    print(f"Mean difference: {mean_diff:.3f}")

    # Check if projection matches
    threshold = 1.0  # pixel

    if max_diff < threshold:
        print(f"投影逻辑正确，最大差异 {max_diff:.3f} < {threshold}")
    else:
        print(f"投影逻辑有问题，最大差异 {max_diff:.3f} >= {threshold}")

    # Save verification results
    results = {
        'projected_uvz': projected_uvz,
        'original_track2d': track2d_numpy,
        'diff': diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }

    output_path = "spacelayer/results/pipeline_demo/projection_verification.npz"
    np.savez(output_path, **results)
    print(f"验证结果保存到: {output_path}")

if __name__ == '__main__':
    main()
