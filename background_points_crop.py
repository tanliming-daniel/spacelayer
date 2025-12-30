import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(project_root, "submodules/SpaTrackerV2"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print("Warning: SpaTrackerV2 not found")

import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import moviepy.editor as mp
from submodules.SpaTrackerV2.models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from submodules.SpaTrackerV2.models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
from rich import print
import argparse
import decord
from submodules.SpaTrackerV2.models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from submodules.SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from submodules.SpaTrackerV2.models.SpaTrackV2.models.vggt4track.utils.pose_enc import pose_encoding_to_extri_intri
from models.util_2d import get_new_sparse_points
from submodules.SpaTrackerV2.models.SpaTrackV2.models.predictor import Predictor


def find_low_density_regions(vertices, shape, min_radius):
    """
    逐个添加点，每次添加后更新密度图
    
    Args:
        vertices: 现有顶点位置，可以是空数组
        shape: 图像尺寸
        min_radius: 最小采样距离
        
    Returns:
        new_positions: 新添加的点位置
        dist_map: 距离变换图
    """
    new_positions = []
    # 处理空输入的情况
    if len(vertices) == 0:
        current_vertices = np.empty((0, 2))  # 创建一个空的2D数组
    else:
        current_vertices = vertices.copy()
        if len(current_vertices.shape) == 1:
            current_vertices = current_vertices.reshape(-1, 2)  # 确保是2D数组
    
    while True:
        # 创建距离变换图
        vertices_map = np.zeros(shape, dtype=np.uint8)
        vertices_int = current_vertices.astype(int)
        mask = (vertices_int[:, 0] >= 0) & (vertices_int[:, 0] < shape[1]) & \
               (vertices_int[:, 1] >= 0) & (vertices_int[:, 1] < shape[0])
        vertices_int = vertices_int[mask]
        if len(vertices_int) > 0:
            vertices_map[vertices_int[:, 1], vertices_int[:, 0]] = 1
        dist_map = cv2.distanceTransform(1 - vertices_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        
        # 找到距离大于阈值的区域
        threshold = min_radius
        valid_regions = dist_map > threshold
        
        # 在有效区域中找最大值点
        if not np.any(valid_regions):
            break
        max_dist = np.max(dist_map * valid_regions)
        if max_dist <= threshold:  # 如果没有满足条件的点，停止添加
            break
            
        # 找到距离最大的点
        y, x = np.where((dist_map == max_dist) & valid_regions)
        if len(x) == 0:
            break
            
        # 选择第一个最大值点
        new_position = np.array([[x[0], y[0]]], dtype=np.float32)
        new_positions.append(new_position[0])
        
        # 更新当前点集
        current_vertices = np.vstack([current_vertices, new_position])
    
    if len(new_positions) > 0:
        return np.stack(new_positions), dist_map
    return np.array([]), dist_map


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_dir", type=str, default="data/DAVIS/JPEGImages/480p", help="视频帧或视频文件所在目录")
    parser.add_argument("--annot_dir", type=str, default="data/DAVIS/Annotations/480p", help="前景Mask标注所在目录")
    parser.add_argument("--video_name", type=str, default="parkour")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--min_radius", type=int, default=20, help="用于动态补点的最小采样距离")
    parser.add_argument("--vo_points", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--query_frame", type=str, default=None, help="指定要对其做稀疏补点的帧索引（0-based, 支持逗号分隔多个）。若为 None 则只使用首尾 grid 采样）。")
    parser.add_argument("--dilate_kernel_size", type=int, default=5, help="对前景Mask进行膨胀操作的核大小。设为0则不进行膨胀。")
    parser.add_argument("--dilate_iterations", type=int, default=8, help="膨胀操作的迭代次数（iterations 参数）。")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    out_dir = f"results/{args.video_name}_background_points"
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载 VGGT4Track 模型
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to("cuda")

    # 加载视频帧
    img_pattern = os.path.join(args.data_dir, f"{args.video_name}", "*.jpg")
    img_files = sorted(glob.glob(img_pattern))
    if img_files:
        imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in img_files]
        raw_video_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
    else:
        vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
        if not os.path.exists(vid_dir):
            raise FileNotFoundError(f"未找到视频文件或图像序列: {vid_dir} 或 {img_pattern}")
        video_reader = decord.VideoReader(vid_dir)
        raw_video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2).float()
    
    # 以 fps 采样（保留原始帧用于可视化）
    original_video_after_fps = raw_video_tensor[::args.fps]
    
    # 只取后49帧
    start_frame_index_after_fps = 0
    if len(original_video_after_fps) > 49:
        start_frame_index_after_fps = len(original_video_after_fps) - 49
        raw_video_tensor = original_video_after_fps[-49:]
    else:
        raw_video_tensor = original_video_after_fps
        
    num_frames, _, orig_frame_H, orig_frame_W = raw_video_tensor.shape

    # 将 video_tensor 设为已预处理的张量（用于模型输入）
    # 这里先保留原始尺寸(orig_frame_H, orig_frame_W)，
    # 然后从 preprocess_image 的输出中获取处理后的 frame_H/frame_W
    processed_video = preprocess_image(raw_video_tensor.clone())
    # processed_video: (T, C, H, W)
    video_tensor = processed_video[None]
    # 使用预处理后的尺寸作为后续与模型一致的尺寸
    _, _, frame_H, frame_W = processed_video.shape

    # 加载与每一帧对应的Mask，并进行膨胀操作
    print(f"[bold green]正在为 '{args.video_name}' 加载并膨胀前景Mask...[/bold green]")
    mask_dir_path = os.path.join(args.annot_dir, args.video_name)
    mask_files = sorted(glob.glob(os.path.join(mask_dir_path, "*.png")))
    
    masks = []
    if mask_files:
        mask_files_sampled = mask_files[::args.fps]
        
        # 根据视频帧的截取方式，同样截取Mask文件列表
        if start_frame_index_after_fps > 0:
            mask_files_sampled = mask_files_sampled[start_frame_index_after_fps:]

        # 确保mask文件列表与帧数一致
        if len(mask_files_sampled) > num_frames:
            mask_files_sampled = mask_files_sampled[:num_frames]
        
        # 创建膨胀操作的核
        kernel = None
        if args.dilate_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate_kernel_size, args.dilate_kernel_size))
            print(f"Mask膨胀核大小: {args.dilate_kernel_size}x{args.dilate_kernel_size}, 迭代次数: {args.dilate_iterations}")

        for f in mask_files_sampled:
            mask_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # 缩放到与视频帧相同的尺寸
                resized_mask = cv2.resize(mask_img, (frame_W, frame_H))
                
                # 如果需要，进行膨胀操作
                if kernel is not None:
                    dilated_mask = cv2.dilate(resized_mask, kernel, iterations=args.dilate_iterations)
                    masks.append(dilated_mask > 0) # 转换为布尔型 (True为前景)
                else:
                    masks.append(resized_mask > 0)
    
    if not masks or len(masks) != num_frames:
        print("[bold yellow]警告: 未找到足够的Mask或Mask加载失败，将不使用Mask进行过滤。[/bold yellow]")
        masks = [np.zeros((frame_H, frame_W), dtype=bool) for _ in range(num_frames)]

    # 运行 VGGT4Track 获取相机参数和深度
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = vggt4track_model(video_tensor.cuda()/255)
        extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
        depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
    
    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    # 初始化 SpaTracker 跟踪模型
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
    model.spatrack.track_num = args.vo_points
    model.eval().to("cuda")
    
    # --- 初始化查询点 ---
    # 对第一帧进行背景点采样
    initial_points_first, _ = find_low_density_regions(np.array([]), (frame_H, frame_W), min_radius=args.grid_size)
    mask_first = masks[0]
    queries_first_xy_list = []
    for pos in initial_points_first:
        x, y = int(round(pos[0])), int(round(pos[1]))
        if 0 <= x < frame_W and 0 <= y < frame_H and not mask_first[y, x]:
            queries_first_xy_list.append(pos)
    queries_first_xy = np.array(queries_first_xy_list)

    # 可视化第一帧的初始点（使用 processed_video，确保坐标与预处理尺寸一致）
    frame_first_vis = processed_video[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
    for point in queries_first_xy:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame_first_vis, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    vis_frame_path_first = os.path.join(out_dir, "initial_points_frame_00000.jpg")
    Image.fromarray(frame_first_vis).save(vis_frame_path_first)
    print(f"可视化第一帧初始点的图像已保存到: {vis_frame_path_first}")

    queries_first = np.concatenate([np.zeros((len(queries_first_xy), 1)), queries_first_xy], axis=1)

    # 对最后一帧进行背景点采样
    initial_points_last, _ = find_low_density_regions(np.array([]), (frame_H, frame_W), min_radius=args.grid_size)
    mask_last = masks[-1]
    queries_last_xy_list = []
    for pos in initial_points_last:
        x, y = int(round(pos[0])), int(round(pos[1]))
        if 0 <= x < frame_W and 0 <= y < frame_H and not mask_last[y, x]:
            queries_last_xy_list.append(pos)
    queries_last_xy = np.array(queries_last_xy_list)

    # 可视化最后一帧的初始点（使用 processed_video，确保坐标与预处理尺寸一致）
    frame_last_vis = processed_video[-1].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
    for point in queries_last_xy:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame_last_vis, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    vis_frame_path_last = os.path.join(out_dir, f"initial_points_frame_{num_frames - 1:05d}.jpg")
    Image.fromarray(frame_last_vis).save(vis_frame_path_last)
    print(f"可视化最后一帧初始点的图像已保存到: {vis_frame_path_last}")

    queries_last = np.concatenate([np.full((len(queries_last_xy), 1), num_frames - 1), queries_last_xy], axis=1)
    
    first_stage_queries = np.concatenate([queries_first, queries_last], axis=0)
    current_queries = first_stage_queries.copy()
 
    # --- 动态补点逻辑 ---
    if args.query_frame:
        query_frames = sorted([int(x) for x in args.query_frame.split(",")])
        
        for qf in query_frames:
            if qf < 0 or qf >= num_frames:
                print(f"[bold red]错误: query_frame {qf} 超出范围 [0, {num_frames-1}]，已跳过。[/bold red]")
                continue

            print(f"[bold blue]正在为第 {qf} 帧动态补点...[/bold blue]")
            # 使用当前的queries做一次track，并获取置信度
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, _, _, _, track2d_pred_tmp, _, conf_pred_tmp, _ = model.forward(
                    video_tensor.squeeze(0), depth=depth_tensor, intrs=intrs, extrs=extrs,
                    queries=current_queries, fps=1, full_point=True, iters_track=4,
                    query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
                    support_frame=num_frames-1, replace_ratio=0
                )
            
            # --- 置信度筛选 ---
            # 1. 筛选出在当前帧(qf)置信度高的点，用于计算密度
            conf_current_frame = conf_pred_tmp[qf, :, 0].detach().cpu().numpy()
            high_conf_mask_frame = conf_current_frame > 0.5
            xy_current_high_conf = track2d_pred_tmp[qf, :, :2].detach().cpu().numpy()[high_conf_mask_frame]
            print(f"在第 {qf} 帧，有 {xy_current_high_conf.shape[0]} / {current_queries.shape[0]} 个点具有高置信度。")

            # 2. 筛选掉在所有帧中平均置信度都低的点，更新 current_queries
            
            # 首先，确定每个点在每一帧是否在画面内
            track_2d_np = track2d_pred_tmp.detach().cpu().numpy()
            on_screen_mask = (track_2d_np[..., 0] >= 0) & (track_2d_np[..., 0] < frame_W) & \
                             (track_2d_np[..., 1] >= 0) & (track_2d_np[..., 1] < frame_H)
            
            conf_np = conf_pred_tmp.squeeze(-1).detach().cpu().numpy()
            
            # 将画面外的点的置信度设为-1，以便在计算平均值时忽略它们
            conf_on_screen_only = np.where(on_screen_mask, conf_np, -1)
            
            # 计算每个点在画面内的平均置信度
            avg_conf_per_point = np.array([
                np.mean(point_confs[point_confs != -1]) if np.any(point_confs != -1) else 0
                for point_confs in conf_on_screen_only.T
            ])

            high_avg_conf_mask = avg_conf_per_point > 0.1 # 使用一个较低的阈值来避免移除过多的点
            
            num_before = current_queries.shape[0]
            current_queries = current_queries[high_avg_conf_mask]
            num_after = current_queries.shape[0]
            if num_after < num_before:
                print(f"移除了 {num_before - num_after} 个在画面内时平均置信度低的点。")

            # 获取当前帧的点分布 (只使用高置信度的点)
            xy_current = xy_current_high_conf
            
            # 生成稀疏点
            new_pts_xy, _ = find_low_density_regions(xy_current, (frame_H, frame_W), min_radius=args.min_radius)

            if new_pts_xy.shape[0] > 0:
                # 构造 (t,x,y) 格式
                new_pts = np.concatenate([np.full((len(new_pts_xy), 1), qf), new_pts_xy], axis=1)
                new_pts = torch.from_numpy(new_pts)
                
                # 使用当前帧的Mask过滤新点
                mask_qf = masks[qf]
                new_pts_xy_int = new_pts[:, 1:].long()
                
                # 确保坐标在图像范围内
                valid_coords_mask = (new_pts_xy_int[:, 0] >= 0) & (new_pts_xy_int[:, 0] < frame_W) & \
                                    (new_pts_xy_int[:, 1] >= 0) & (new_pts_xy_int[:, 1] < frame_H)
                
                new_pts_valid_coords = new_pts[valid_coords_mask]
                new_pts_xy_int_valid = new_pts_xy_int[valid_coords_mask]

                if new_pts_xy_int_valid.numel() > 0:
                    background_mask_new = ~mask_qf[new_pts_xy_int_valid[:, 1], new_pts_xy_int_valid[:, 0]]
                    final_new_pts = new_pts_valid_coords[background_mask_new].cpu().numpy()
                    
                    if final_new_pts.shape[0] > 0:
                        print(f"在第 {qf} 帧的背景区域补充了 {len(final_new_pts)} 个点。")
                        
                        # 可视化当前帧新增的点（使用 processed_video）
                        frame_to_visualize = processed_video[qf].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
                        # 在图像上绘制新点
                        for point in final_new_pts:
                            x, y = int(point[1]), int(point[2])
                            cv2.circle(frame_to_visualize, (x, y), radius=3, color=(0, 255, 0), thickness=-1) # 绿色
                        
                        # 保存可视化图像
                        vis_frame_path = os.path.join(out_dir, f"added_points_frame_{qf:05d}.jpg")
                        Image.fromarray(frame_to_visualize).save(vis_frame_path)
                        print(f"可视化新增点的帧已保存到: {vis_frame_path}")

                        current_queries = np.concatenate([current_queries, final_new_pts], axis=0)
                        # 可视化所有query点分布
                        all_points_vis = processed_video[qf].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
                        
                        # 绘制 current_queries 中的点
                        qf_queries_mask = (current_queries[:, 0].astype(int) == qf)
                        qf_queries = current_queries[qf_queries_mask]
                        for point in qf_queries:
                            x, y = int(point[1]), int(point[2])
                            cv2.circle(all_points_vis, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # 蓝色

                        # 绘制 track2d_pred_tmp 中的点
                        for point in xy_current:
                            x, y = int(point[0]), int(point[1])
                            cv2.circle(all_points_vis, (x, y), radius=2, color=(0, 0, 255), thickness=-1) # 红色

                        vis_all_points_path = os.path.join(out_dir, f"all_queries_frame_{qf:05d}.jpg")
                        Image.fromarray(all_points_vis).save(vis_all_points_path)
                        print(f"所有query点分布可视化已保存到: {vis_all_points_path}")

    # --- 最终跟踪 ---
    print(f"[bold green]所有点已准备就绪（共 {len(current_queries)} 个），开始最终跟踪...[/bold green]")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor.squeeze(0), depth=depth_tensor,
                            intrs=intrs, extrs=extrs,
                            queries=current_queries,
                            fps=1, full_point=True, iters_track=10,
                            query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=num_frames-1, replace_ratio=0)
    
    # 保存模型输出的原始张量
    tensors_to_save = {
        'c2w_traj': c2w_traj.cpu(),
        'intrs': intrs.cpu(),
        'extrs': torch.from_numpy(extrs),
        'point_map': point_map.cpu(),
        'conf_depth': conf_depth.cpu(),
        'track3d_pred': track3d_pred.cpu(),
        'track2d_pred': track2d_pred.cpu(),
        'vis_pred': vis_pred.cpu(),
        'conf_pred': conf_pred.cpu(),
        'video': video.cpu()
    }
    tensor_output_path = os.path.join(out_dir, f'{args.video_name}_background_tensors.pt')
    torch.save(tensors_to_save, tensor_output_path)
    print(f"[bold green]模型原始输出张量已保存到: {tensor_output_path}[/bold green]")

    print("[bold blue]正在准备可视化...[/bold blue]")
    viser = Visualizer(save_dir=out_dir, grayscale=True, fps=10, pad_value=0, tracks_leave_trace=5)
    
    viser.visualize(video=video[None],
                    tracks=track2d_pred[None][...,:2],
                    visibility=vis_pred[None],
                    filename=f"{args.video_name}_background_track")

    # 保存为 tapip3d 格式
    data_npz_load = {}
    data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
    data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
    data_npz_load["intrinsics"] = intrs.cpu().numpy()
    depth_save = point_map[:,2,...]
    depth_save[conf_depth<0.5] = 0
    data_npz_load["depths"] = depth_save.cpu().numpy()
    # 保存原始和预处理后的视频（归一化到 [0,1]）
    data_npz_load["video_raw"] = (raw_video_tensor).cpu().numpy()/255
    data_npz_load["video"] = (processed_video).cpu().numpy()/255
    data_npz_load["visibs"] = vis_pred.cpu().numpy()
    data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
    
    output_npz_path = os.path.join(out_dir, f'{args.video_name}_result.npz')
    np.savez(output_npz_path, **data_npz_load)

    print(f"\n[bold cyan]所有处理完成！[/bold cyan]")
    print(f"可视化视频已保存到: {os.path.join(out_dir, f'{args.video_name}_background_track.mp4')}")
    print(f"结果数据已保存到: {output_npz_path}")
    print(f"原始张量数据已保存到: {tensor_output_path}")