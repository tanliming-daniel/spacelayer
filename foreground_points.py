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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_dir", type=str, default="data/DAVIS/JPEGImages/480p")
    parser.add_argument("--annot_dir", type=str, default="data/DAVIS/Annotations/480p", help="前景Mask标注所在目录")
    parser.add_argument("--video_name", type=str, default="car-turn")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--vo_points", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--query_frame", type=str, default=None, help="指定要对其做稀疏补点的帧索引（0-based, 支持逗号分隔多个）。若为 None 则只使用首尾 grid 采样）。")
    parser.add_argument("--dilate_kernel_size", type=int, default=5, help="对前景Mask进行膨胀操作的核大小。设为0则不进行膨胀。")
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    out_dir = f"results/{args.video_name}_foreground_points"
    os.makedirs(out_dir, exist_ok=True)
    
    # 加载 VGGT4Track 模型
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front").eval().to("cuda")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")
    # 加载视频帧
    img_pattern = os.path.join(args.data_dir, f"{args.video_name}", "*.jpg")
    img_files = sorted(glob.glob(img_pattern))
    if img_files:
        imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in img_files]
        video_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()
    else:
        vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
        if not os.path.exists(vid_dir):
            raise FileNotFoundError(f"未找到视频文件或图像序列: {vid_dir} 或 {img_pattern}")
        video_reader = decord.VideoReader(vid_dir)
        video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2).float()
    
    video_tensor = video_tensor[::args.fps]
    # 统一预处理流程，和inference.py一致
    video_tensor_processed = preprocess_image(video_tensor)
    video_tensor_processed = video_tensor_processed[None]  # 增加batch维
    # 后续所有点采样、可视化、track都用squeeze后的video_tensor
    video_tensor = video_tensor_processed.squeeze()
    num_frames, _, frame_H, frame_W = video_tensor.shape

    # 加载与每一帧对应的Mask，并进行膨胀操作
    print(f"[bold green]正在为 '{args.video_name}' 加载并膨胀前景Mask...[/bold green]")
    mask_dir_path = os.path.join(args.annot_dir, args.video_name)
    mask_files = sorted(glob.glob(os.path.join(mask_dir_path, "*.png")))
    
    masks = []
    if mask_files:
        mask_files_sampled = mask_files[::args.fps]
        
        # 创建膨胀操作的核
        kernel = None
        if args.dilate_kernel_size > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.dilate_kernel_size, args.dilate_kernel_size))
            print(f"Mask膨胀核大小: {args.dilate_kernel_size}x{args.dilate_kernel_size}")

        for f in mask_files_sampled:
            mask_img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # 缩放到与视频帧相同的尺寸
                resized_mask = cv2.resize(mask_img, (frame_W, frame_H))
                
                # 如果需要，进行膨胀操作
                if kernel is not None:
                    dilated_mask = cv2.dilate(resized_mask, kernel, iterations=1)
                    masks.append(dilated_mask > 0) # 转换为布尔型 (True为前景)
                else:
                    masks.append(resized_mask > 0)
    
    if not masks or len(masks) != num_frames:
        print("[bold yellow]警告: 未找到足够的Mask或Mask加载失败，将不使用Mask进行过滤。[/bold yellow]")
        masks = [np.zeros((frame_H, frame_W), dtype=bool) for _ in range(num_frames)]

    # 运行 VGGT4Track 获取相机参数和深度
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        predictions = vggt4track_model(video_tensor_processed.cuda()/255)
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
    grid_pts_template = get_points_on_a_grid(args.grid_size, (frame_H, frame_W), device="cpu")[0]

    # 对第一帧进行前景点采样
    mask_first = masks[0]
    grid_pts_int_first = grid_pts_template.long()
    foreground_mask_first = mask_first[grid_pts_int_first[..., 1], grid_pts_int_first[..., 0]]
    queries_first_xy = grid_pts_template[foreground_mask_first]

    queries_first = np.concatenate([np.zeros((len(queries_first_xy), 1)), queries_first_xy], axis=1)

    # 对最后一帧进行前景点采样
    mask_last = masks[-1]
    grid_pts_int_last = grid_pts_template.long()
    foreground_mask_last = mask_last[grid_pts_int_last[..., 1], grid_pts_int_last[..., 0]]
    queries_last_xy = grid_pts_template[foreground_mask_last]

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
            # 使用当前的queries做一次track
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, _, _, _, track2d_pred_tmp, _, _, _ = model.forward(
                    video_tensor, depth=depth_tensor, intrs=intrs, extrs=extrs,
                    queries=current_queries, fps=1, full_point=True, iters_track=4,
                    query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
                    support_frame=num_frames-1, replace_ratio=0
                )
            
            # 获取当前帧的点分布
            xy_current = track2d_pred_tmp[qf, :, :2].detach().cpu()
            
            # 生成稀疏点
            new_pts = get_new_sparse_points(xy_current, args.grid_size, (frame_H, frame_W), qf, device="cpu")
            
            if new_pts.numel() > 0:
                # 使用当前帧的Mask过滤新点（只保留前景点）
                mask_qf = masks[qf]
                new_pts_xy_int = new_pts[:, 1:].long()
                
                # 确保坐标在图像范围内
                valid_coords_mask = (new_pts_xy_int[:, 0] >= 0) & (new_pts_xy_int[:, 0] < frame_W) & \
                                    (new_pts_xy_int[:, 1] >= 0) & (new_pts_xy_int[:, 1] < frame_H)
                
                new_pts_valid_coords = new_pts[valid_coords_mask]
                new_pts_xy_int_valid = new_pts_xy_int[valid_coords_mask]

                if new_pts_xy_int_valid.numel() > 0:
                    foreground_mask_new = mask_qf[new_pts_xy_int_valid[:, 1], new_pts_xy_int_valid[:, 0]]
                    final_new_pts = new_pts_valid_coords[foreground_mask_new].cpu().numpy()
                    
                    if final_new_pts.shape[0] > 0:
                        print(f"在第 {qf} 帧的前景区域补充了 {len(final_new_pts)} 个点。")
                        
                        # 可视化当前帧新增的点
                        frame_to_visualize = video_tensor[qf].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
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
                        all_points_vis = video_tensor[qf].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
                        for point in current_queries:
                            x, y = int(point[1]), int(point[2])
                            cv2.circle(all_points_vis, (x, y), radius=2, color=(255, 0, 0), thickness=-1)  # 蓝色
                        vis_all_points_path = os.path.join(out_dir, f"all_queries_frame_{qf:05d}.jpg")
                        Image.fromarray(all_points_vis).save(vis_all_points_path)
                        print(f"所有query点分布可视化已保存到: {vis_all_points_path}")

    # --- 最终跟踪 ---
    print(f"[bold green]所有点已准备就绪（共 {len(current_queries)} 个），开始最终跟踪...[/bold green]")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs,
                            queries=current_queries,
                            fps=1, full_point=True, iters_track=4,
                            query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=num_frames-1, replace_ratio=0)
    
    # 保存模型输出的原始张量
    tensors_to_save = {
        'c2w_traj': c2w_traj.cpu(),
        'intrs': intrs.cpu(),
        'point_map': point_map.cpu(),
        'conf_depth': conf_depth.cpu(),
        'track3d_pred': track3d_pred.cpu(),
        'track2d_pred': track2d_pred.cpu(),
        'vis_pred': vis_pred.cpu(),
        'conf_pred': conf_pred.cpu(),
        'video': video.cpu()
    }
    tensor_output_path = os.path.join(out_dir, f'{args.video_name}_foreground_tensors.pt')
    torch.save(tensors_to_save, tensor_output_path)
    print(f"[bold green]模型原始输出张量已保存到: {tensor_output_path}[/bold green]")

    print("[bold blue]正在手动生成可视化视频...[/bold blue]")
    
    # 准备视频帧
    # video 维度: (T, C, H, W), RGB, 0-255
    # track2d_pred 维度: (T, N, 2), (x, y)
    # vis_pred 维度: (T, N), boolean
    
    num_frames, num_points, _ = track2d_pred.shape
    
    # 为每个点生成一个固定的颜色
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_points, 3), dtype=np.uint8)
    
    output_frames = []
    for t in tqdm.tqdm(range(num_frames), desc="绘制跟踪点"):
        # 将 PyTorch 张量转换为 OpenCV 图像格式
        frame = video[t].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 获取当前帧的所有点和可见性
        points_t = track2d_pred[t].cpu().numpy()
        visibility_t = vis_pred[t].cpu().numpy()
        
        # 绘制可见的点
        for i in range(num_points):
            if visibility_t[i]:
                x, y = int(points_t[i, 0]), int(points_t[i, 1])
                color = tuple(map(int, colors[i]))
                cv2.circle(frame_bgr, (x, y), radius=3, color=color, thickness=-1)
        
        # 将处理后的帧转回 RGB 以便 moviepy 使用
        output_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    # 使用 moviepy 将帧序列保存为视频
    video_clip = mp.ImageSequenceClip(output_frames, fps=10)
    output_video_path = os.path.join(out_dir, f"{args.video_name}_foreground_track.mp4")
    video_clip.write_videofile(output_video_path, codec="libx264")

    # 保存为 tapip3d 格式
    data_npz_load = {}
    data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
    data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
    data_npz_load["intrinsics"] = intrs.cpu().numpy()
    depth_save = point_map[:,2,...]
    depth_save[conf_depth<0.5] = 0
    data_npz_load["depths"] = depth_save.cpu().numpy()
    data_npz_load["video"] = (video_tensor).cpu().numpy()/255
    data_npz_load["visibs"] = vis_pred.cpu().numpy()
    data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
    
    output_npz_path = os.path.join(out_dir, f'{args.video_name}_foreground_result.npz')
    np.savez(output_npz_path, **data_npz_load)

    print(f"\n[bold cyan]所有处理完成！[/bold cyan]")
    print(f"可视化视频已保存到: {os.path.join(out_dir, f'{args.video_name}_foreground_track.mp4')}")
    print(f"结果数据已保存到: {output_npz_path}")
    print(f"原始张量数据已保存到: {tensor_output_path}")
    
    # --- 最终可视化过滤 ---
    print("[bold blue]正在进行最终可视化过滤（只保留前景点）...")
    track2d_pred_np = track2d_pred.cpu().numpy()
    vis_pred_np = vis_pred.cpu().numpy()
    filtered_tracks = []
    filtered_vis = []
    for i in range(num_frames):
        mask_frame = masks[i]
        points_on_frame_int = track2d_pred_np[i, :, :2].astype(int)
        valid_x = (points_on_frame_int[:, 0] >= 0) & (points_on_frame_int[:, 0] < frame_W)
        valid_y = (points_on_frame_int[:, 1] >= 0) & (points_on_frame_int[:, 1] < frame_H)
        valid_coords_mask = valid_x & valid_y
        keep_mask = np.zeros(len(points_on_frame_int), dtype=bool)
        if np.any(valid_coords_mask):
            points_to_check = points_on_frame_int[valid_coords_mask]
            in_foreground = mask_frame[points_to_check[:, 1], points_to_check[:, 0]]
            keep_mask[valid_coords_mask] = in_foreground
        # 只保留前景点
        filtered_tracks.append(track2d_pred[i][keep_mask])
        filtered_vis.append(vis_pred[i][keep_mask])

    # 逐帧可视化（每帧点数不同）
    print("[bold blue]正在生成最终跟踪视频...[/bold blue]")
    output_frames = []
    for t in range(num_frames):
        frame = video[t].cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        points_t = filtered_tracks[t].cpu().numpy() if torch.is_tensor(filtered_tracks[t]) else filtered_tracks[t]
        visibility_t = filtered_vis[t].cpu().numpy() if torch.is_tensor(filtered_vis[t]) else filtered_vis[t]
        for i in range(points_t.shape[0]):
            if visibility_t[i]:
                x, y = int(points_t[i, 0]), int(points_t[i, 1])
                cv2.circle(frame_bgr, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        output_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    video_clip = mp.ImageSequenceClip(output_frames, fps=10)
    output_video_path = os.path.join(out_dir, f"{args.video_name}_foreground_track.mp4")
    video_clip.write_videofile(output_video_path, codec="libx264")