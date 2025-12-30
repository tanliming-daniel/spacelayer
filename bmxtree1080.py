import sys
sys.path.append('/data4/cy/ShapeLayerVideo/tools/sam2')

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tapnet.utils import transforms
import mediapy as media
import matplotlib.pyplot as plt
import cv2
import PIL.Image
import time
from tqdm import tqdm
from util import to_pixel_samples, make_pixel, Embedding
from utils.tools import *
import subprocess
from scipy.ndimage import  binary_fill_holes
import argparse

def preprocess_frames(frames):
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
  visibles = (1 - F.sigmoid(occlusions)) * (1 - F.sigmoid(expected_dist)) > 0.5
  return visibles

def inference(frames, query_points, model):
    frames = preprocess_frames(frames)
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]
    outputs = model(frames, query_points)
    tracks, occlusions, expected_dist = (
      outputs['tracks'][0],
      outputs['occlusion'][0],
      outputs['expected_dist'][0],
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles



def convert_select_points_to_query_points(frame, points):
    points = np.stack(points)
    query_points = np.zeros(shape=(points.shape[0], 3), dtype=np.float32)
    query_points[:, 0] = frame
    query_points[:, 1] = points[:, 1]
    query_points[:, 2] = points[:, 0]
    return query_points


def make_convex_fore(mask, erosion_size=10):
    # 形态学闭合操作（填补小孔）
    kernel = np.ones((5, 5), np.uint8)  # 选择合适的结构元素大小
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    # 填充空洞
    filled_mask = binary_fill_holes(closed_mask).astype(np.uint8)

    # 计算凸包
    contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hull = np.zeros_like(filled_mask)

    if contours:
        cv2.drawContours(convex_hull, contours, -1, (1), thickness=cv2.FILLED)

    # **添加腐蚀操作**
    erosion_kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(convex_hull, erosion_kernel, iterations=1)

    return eroded_mask


def process_single_frame(
        frame_idx, 
        frame_deformed_vertices_list,
        device, 
        video_segment, 
        gt_frame,
        args,
        visible_list=None,
        special_points_info=None,
        special_points_mapper=None,
        eval_mode=False,
        delete_mode=False,
        is_training=False):
    """Process a single frame's deformed points for ShapeLayer input."""
    from utils.tools import mesh_processX
    import cv2
    
    h, w = args.img_h, args.img_w
    global min_x_global, min_y_global, global_h, global_w


    rgb_inmask_list = []
    coord_inmask_list = []
    pixel_vertices_list = []
    pixel_vertice_coords_list = []
    in_mask_triangles_list = []
    barycentric_coords_list = []
    masks = []
    
    # Prepare frame data
    gt = gt_frame
    if len(gt.shape) == 2:
        gt = gt.unsqueeze(dim=-1).repeat(1,1,3)
    if gt.shape[2] == 4:
        gt = gt[:, :, :3]
    gt = (gt/255).astype(np.float32)
    gt = torch.FloatTensor(gt).permute(2, 0, 1)[None].to(device)
    coord_c, rgb = to_pixel_samples(gt.contiguous())
    coord_c = coord_c.flip(-1)
    coord_i = make_pixel((h,w))
    coord_c[:, 1] = coord_c[:, 1] * (h / global_h) - (min_y_global / global_h)
    coord_c[:, 0] = coord_c[:, 0] * (w / global_w) - (min_x_global / global_w)
    base_mask = video_segment[1]
    
    start_feature = 0
    start_vertice = 0
    # Process each layer
    for layer_idx , _ in enumerate(frame_deformed_vertices_list):
        if delete_mode and layer_idx != 0:
            continue
        if_dilate = layer_idx==1

        mask=~base_mask
        
        
        # For layer 0, use the inverse of mask; for others, use mask as is
        if layer_idx == 0:
            if is_training:
                mask = 1-mask  # Invert mask for layer 0
                # kernel = np.ones((8, 8), np.uint8)
                # mask = cv2.dilate(mask, kernel, iterations=3)
            else:
                mask = 1-mask
            

        frame_deformed_vertices = frame_deformed_vertices_list[layer_idx]
        # Save the mask for this frame and layer
        mask_dir = "./test_mask/1"
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"mask-{frame_idx}-layer{layer_idx}.png")
        cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
        
        base_mask_path = os.path.join(mask_dir, f"base_mask-{frame_idx}.png")
        cv2.imwrite(base_mask_path, base_mask.astype(np.uint8) * 255)
        
        start_feature= 0
        if layer_idx == 0:
            # 获取当前可见点的边界
                vertices = frame_deformed_vertices_list[layer_idx]
                vertices_visibility = visible_list[layer_idx][frame_idx,:]
                visible_vertices = vertices[vertices_visibility]
                
                if len(visible_vertices) > 0:
                    visible_min_x = visible_vertices[:, 0].min()
                    visible_max_x = visible_vertices[:, 0].max()
                    min_y = visible_vertices[:, 1].min()
                    max_y = visible_vertices[:, 1].max()
                    
                    # 计算扩展边界
                    extend_min_x = visible_min_x -30
                    extend_max_x = visible_max_x +30
                    extend_min_y = min_y-20
                    extend_max_y = max_y+20

                    # 找到在扩展区域内的vertices
                    vertices_in_left_extend = (vertices[:, 0] >= extend_min_x) & (vertices[:, 0] < visible_min_x+2)
                    vertices_in_right_extend = (vertices[:, 0] > visible_max_x-2) & (vertices[:, 0] <= extend_max_x)
                    vertices_in_up_extend = (vertices[:, 1] >= extend_min_y) & (vertices[:, 1] < min_y+4)
                    vertices_in_down_extend = (vertices[:, 1] > max_y-4) & (vertices[:, 1] <= extend_max_y)
                    vertices_in_y_range = (vertices[:, 1] >= extend_min_y) & (vertices[:, 1] <= extend_max_y)
                    vertices_in_x_range = (vertices[:, 0] >= extend_min_x) & (vertices[:, 0] <= extend_max_x)
                    
                    # 修改扩展区域内的点的可见性
                    vertices_in_extendx = (vertices_in_left_extend | vertices_in_right_extend) & vertices_in_y_range
                    vertices_visibility[vertices_in_extendx] = True
                    vertices_in_extendy =(vertices_in_up_extend | vertices_in_down_extend) & vertices_in_x_range
                    vertices_visibility[vertices_in_extendy] = True
        else:
            vertices_visibility = visible_list[layer_idx][frame_idx,:]
        # 对于layer0，提供特殊点信息
        if layer_idx == 0 :    
            if delete_mode:
                from utils.tools import mesh_processX_delete
                processed_data = mesh_processX_delete(
                    frame_deformed_vertices,
                    mask,
                    rgb,
                    coord_c,
                    coord_i,
                    global_h, global_w,
                    start_vertice=start_vertice,
                    start_feature=start_feature,
                    out_dir=f"{args.output_dir}/masks/frame_{frame_idx}_layer_{layer_idx}",
                    mask_filter=False,
                    if_dilate=if_dilate,
                    visibles=vertices_visibility,
                    retriangle=True,
                    frame_idx=frame_idx/30
                )
            else:
                processed_data = mesh_processX(
                    frame_deformed_vertices,
                    mask,
                    rgb,
                    coord_c,
                    coord_i,
                    global_h, global_w,
                    start_vertice=start_vertice,
                    start_feature=start_feature,
                    out_dir=f"{args.output_dir}/masks/frame_{frame_idx}_layer_{layer_idx}",
                    mask_filter=False,
                    if_dilate=if_dilate,
                    visibles=vertices_visibility,
                    retriangle=True,
                    frame_idx=frame_idx/30)              
        else:
            processed_data = mesh_processX(
                frame_deformed_vertices,
                mask,
                rgb,
                coord_c,
                coord_i,
                global_h, global_w,
                start_vertice=start_vertice,
                start_feature=start_feature,
                out_dir=f"{args.output_dir}/masks/frame_{frame_idx}_layer_{layer_idx}",
                mask_filter=False,
                if_dilate=if_dilate,
                visibles=vertices_visibility,
                retriangle=True,
                frame_idx=frame_idx/30
            )
        
        rgb_inmask_list.append(processed_data[1])
        coord_inmask_list.append(processed_data[5])
        pixel_vertices_list.append(processed_data[6])
        pixel_vertice_coords_list.append(processed_data[7])
        barycentric_coords_list.append(processed_data[8])
        in_mask_triangles_list.append(processed_data[4])
        masks.append(mask)
        start_vertice += processed_data[11]
        
        del processed_data

    # Get layer segment sizes
    layer_segs = [rgb.shape[0] for rgb in rgb_inmask_list]
    
    # Concatenate results and keep on GPU for immediate use
    frame_data_per_layer = []
    for i in range(len(rgb_inmask_list)):
        frame_data_per_layer.append([
            rgb_inmask_list[i],
            coord_inmask_list[i],
            pixel_vertices_list[i],
            pixel_vertice_coords_list[i],
            barycentric_coords_list[i],
            layer_segs[i],
            masks[i],
            in_mask_triangles_list[i]
        ])
    
    del rgb_inmask_list, coord_inmask_list, pixel_vertices_list
    del barycentric_coords_list, coord_c, rgb, coord_i, gt

    return frame_data_per_layer


def compute_loss(pixel_colors_pred, rgb_inmask):
    """Compute loss between predicted colors and ground truth rgb_inmask."""
    loss = torch.mean((pixel_colors_pred - rgb_inmask) ** 2)
    return loss

def compute_metrics(pred_img, gt_img):
    """Compute PSNR, LPIPS, and SSIM metrics between predicted and ground truth images."""
    import lpips
    import torchvision.transforms.functional as TF
    from skimage.metrics import structural_similarity as ssim
    
    # 计算 PSNR
    mse = torch.mean((pred_img - gt_img) ** 2).item()
    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * math.log10(1 / mse)
    
    # 计算 LPIPS
    loss_fn = lpips.LPIPS(net='alex').to(pred_img.device)
    # 转换为LPIPS所需的格式: [-1, 1]范围的 BCHW 格式
    pred_img_lpips = pred_img.reshape(1, gt_img.shape[0], gt_img.shape[1], 3).permute(0, 3, 1, 2) * 2 - 1
    gt_img_lpips = gt_img.reshape(1, gt_img.shape[0], gt_img.shape[1], 3).permute(0, 3, 1, 2) * 2 - 1
    lpips_value = loss_fn(pred_img_lpips, gt_img_lpips).item()
    
    # 计算 SSIM
    pred_np = pred_img.reshape(gt_img.shape[0], gt_img.shape[1], 3).cpu().numpy()
    gt_np = gt_img.reshape(gt_img.shape[0], gt_img.shape[1], 3).cpu().numpy()
    # 使用channel_axis=2替代multichannel=True，因为彩色图像的通道在最后一个维度
    ssim_value = ssim(pred_np, gt_np, channel_axis=2, data_range=1.0)
    
    return psnr, lpips_value, ssim_value

def count_parameters(model):
    """计算模型的参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def visualize_vertices(current_frame_vertices, frame_idx, epoch, args):
    """Visualize vertices positions for each layer."""
    h, w = args.img_h, args.img_w
    
    # 创建空白图像
    plt.figure(figsize=(12, 8))

    
    # 为每个layer的vertices使用不同颜色绘制点
    if frame_idx != -1:
        for layer_idx, vertices in enumerate(current_frame_vertices):
            vertices_cpu = vertices.cpu()  # 将tensor移到CPU
            plt.scatter(vertices_cpu[:,0], vertices_cpu[:,1], c=f'C{layer_idx}', 
                    s=10, alpha=0.5, label=f'Layer {layer_idx}')
    if frame_idx == -2:
        plt.title(f"Initial vertices positions")
    if frame_idx == -1:
        plt.title(f"tapnet vertices positions")
        vertices_cpu = current_frame_vertices.cpu()  # 将tensor移到CPU
        plt.scatter(vertices_cpu[:,0], vertices_cpu[:,1],
                s=10, alpha=0.5, label=f'Layer {-1}')

    plt.legend()
    plt.xlim(0, w)
    plt.ylim(h, 0)  # 翻转y轴以匹配图像坐标系
    plt.axis('equal')
    
    # 保存到单独的文件
    vertices_path = os.path.join(args.output_dir, f"vertice/frame{frame_idx:04d}epoch{epoch}.png")
    os.makedirs(os.path.dirname(vertices_path), exist_ok=True)
    plt.savefig(vertices_path)
    plt.close()

def visualize_results(frame_data, frame_idx, epoch, args, shapelayer_net):
    """Visualize prediction results for a single frame using mask-based visualization."""
    h, w = args.img_h, args.img_w
    device = args.device
    vis_image = torch.zeros((h*w, 3)).to(device)
    
    for layer_idx in range(2):  # 前景和背景两层
        frame = torch.tensor(frame_idx / (num_frames - 1), device=device).float().view(1, 1).repeat(frame_data[layer_idx][2].shape[0], 1)
        # 使用单层数据进行预测, 并传入layer_idx参数
        # Process vertices in two batches to reduce memory usage
        batch_size = frame_data[layer_idx][2].shape[0] // 2
        if batch_size > 0:
            # First batch
            layer_colors_pred1 = shapelayer_net(
                frame_data[layer_idx][2][:batch_size],  # pixel_vertices 
                frame_data[layer_idx][3][:batch_size],  # pixel_vertice_coords
                frame_data[layer_idx][1][:batch_size],  # coord_inmask
                frame_data[layer_idx][4][:batch_size],  # barycentric_coords
                frame_idx=frame[:batch_size] if frame is not None else None
            )
            
            # Second batch
            layer_colors_pred2 = shapelayer_net(
                frame_data[layer_idx][2][batch_size:],  # pixel_vertices 
                frame_data[layer_idx][3][batch_size:],  # pixel_vertice_coords
                frame_data[layer_idx][1][batch_size:],  # coord_inmask
                frame_data[layer_idx][4][batch_size:],  # barycentric_coords
                frame_idx=frame[batch_size:] if frame is not None else None
            )
            
            # Combine results
            layer_colors_pred = torch.cat([layer_colors_pred1, layer_colors_pred2], dim=0)
            
            # Clean up intermediates
            del layer_colors_pred1, layer_colors_pred2
        else:
            # If the batch is too small, process it in one go
            layer_colors_pred = shapelayer_net(
                frame_data[layer_idx][2],  # pixel_vertices 
                frame_data[layer_idx][3],  # pixel_vertice_coords
                frame_data[layer_idx][1],  # coord_inmask
                frame_data[layer_idx][4],  # barycentric_coords
                frame_idx=frame
            )
        
        # 使用mask将预测结果放到对应位置
        mask = frame_data[layer_idx][6]  # 获取当前层的mask
        vis_image[mask.flatten() == 0] = layer_colors_pred
        
        del layer_colors_pred
    # Move to CPU and convert to numpy with memory optimization
    vis_image_np = vis_image.cpu().detach().numpy().reshape(h, w, 3)
    
    # Save visualization
    results_path = os.path.join(args.output_dir, f"frame{frame_idx}result/epoch{epoch}.png")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    plt.imsave(results_path, vis_image_np)
    if epoch==0:
        gt_frame_path = os.path.join(args.output_dir, f"frame{frame_idx}result/gt.png")
        plt.imsave(gt_frame_path, gt_frames[frame_idx])

    
    del vis_image

def load_checkpoint(checkpoint_path, model):
    """Load checkpoint into model."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def init_shapelayerX(mapper, args):
    """Initialize ShapeLayerX network."""
    from util import ShapeLayerX1, ShapeLayerX_MFN  # Import ShapeLayerX from util
    # Initialize ShapeLayerX with new format
    if args.use_mfn:
        print(f"use MFN model")
        net = ShapeLayerX_MFN(
            in_size=args.feat_dim,
            hidden_size=args.W,
            out_size=3,
            n_layers=args.D,
            n_layers_hidden=args.n_layers_hidden,
            feat_embed=args.feat_embed,
            multi_hidden=args.multi_hidden,
            use_relu=args.use_relu,
            feat_freq=args.feat_freq,
            mapper=mapper,
            input_scale=args.f_scale,
            bias=True,
            output_act=False,
        ).to(args.device)
    else:
        print(f"use ShapeLayerX model")
        net = ShapeLayerX1(
            feat_dim=args.feat_dim,
            D=args.D,
            W=args.W, 
            pos_embed=args.pos_embed,
            feat_embed=args.feat_embed,
            time_embed=args.time_embed,
            time_freq =args.time_freq,
            feat_freq=args.feat_freq,
            pos_freq=args.pos_freq,
            mapper=mapper
        ).to(args.device)
    
    # Load checkpoint if specified
    if hasattr(args, 'checkpoint_path') and args.checkpoint_path:
        net = load_checkpoint(args.checkpoint_path, net)
        print(f"Loaded checkpoint from {args.checkpoint_path}")
    
    return net

def save_checkpoints(shapelayer_net,epoch,args):
    """Save network checkpoints."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # In phase 2, save both networks
    torch.save({
        'model_state_dict': shapelayer_net.state_dict(),
        'epoch': epoch,
        'args': args.__dict__
    }, os.path.join(args.checkpoint_dir, f"shapelayer_{epoch}.pth"))
    
def visualize_triangles(vertices, in_mask_triangles, epoch, args, frame_idx, visible_list=None):
    # 如果只有一个layer，就直接处理它
    global min_x_global, min_y_global
    if not isinstance(vertices, list):
        vertices = [vertices]
        in_mask_triangles = [in_mask_triangles]
    
    layer_idx = 0
    for verts_data, tri_data in zip(vertices, in_mask_triangles):
        # 创建图像并设置大小
        figure, ax = plt.subplots(figsize=(10, 6))
        verts = verts_data.cpu()
        
        # 添加gt帧作为背景
        ax.imshow(gt_frames[frame_idx])
        
        # 绘制三角形网格并叠加在gt帧上 - 使用更深颜色和更粗的线条
        ax.axes.set_aspect("equal")
        size = [args.img_w, args.img_h]
        ax.triplot(verts[:, 0]+min_x_global, verts[:, 1]+min_y_global, tri_data, "r-", linewidth=1.2, alpha=0.8)
        
        # 如果有可见性信息，区分可见点和不可见点
        if visible_list is not None and layer_idx < len(visible_list):
            # 获取当前层当前帧的可见性信息
            visibility = visible_list[layer_idx][frame_idx, :].cpu()
            
            # 可见点和不可见点的坐标
            visible_mask = visibility == True
            invisible_mask = ~visible_mask
            
            # 绘制可见点
            if visible_mask.sum() > 0:
                ax.scatter(verts[visible_mask, 0]+min_x_global, 
                          verts[visible_mask, 1]+min_y_global, 
                          c='lime', s=4.0, alpha=1.0, 
                          edgecolors='black', linewidth=0.5,
                          label='Visible')
            
            # 绘制不可见点
            if invisible_mask.sum() > 0:
                ax.scatter(verts[invisible_mask, 0]+min_x_global, 
                          verts[invisible_mask, 1]+min_y_global, 
                          c='red', s=3.0, alpha=0.7, 
                          edgecolors='black', linewidth=0.5,
                          label='Invisible')
                
            # 添加图例
            if visible_mask.sum() > 0 and invisible_mask.sum() > 0:
                ax.legend(loc='upper right', fontsize=8)
        else:
            # 如果没有可见性信息，使用原来的方式绘制所有点
            ax.scatter(verts[:, 0]+min_x_global, verts[:, 1]+min_y_global, 
                      c='cyan', s=3.5, alpha=0.9, 
                      edgecolors='black', linewidth=0.5)
        
        plt.axis("off")
        output_dir = os.path.join(args.output_dir, f"triangles{epoch}layer{layer_idx}")
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"frame{frame_idx:04d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        layer_idx += 1

def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track and optimize ShapeLayer")
    
    # Input/output paths
    parser.add_argument("--video_path", type=str, default="/data4/cy/ShapeLayerVideo/fmbs/bear01_tmp", help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="./output", help="Path to output directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--mask_dir", type=str, default="./masks", help="Path to checkpoint directory")
    parser.add_argument("--vertices_tracks_path", type=str, default="/data4/cy/ShapeLayerVideo/tracks/20251002_160715bear01_tmp/vertices_tracks.npy", help="Path to the vertices tracking results npy file")
    
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to specific checkpoint to load")

    # Network parameters
    parser.add_argument("--net_width", type=int, default=16, help="Network width for deform_network")
    parser.add_argument("--defor_depth", type=int, default=3, help="Network depth for deform_network")
    parser.add_argument("--posebase_pe", type=int, default=4, help="Position encoding dimension for deform_network")
    
    parser.add_argument("--feat_dim", type=int, default=128, help="Feature dimension for ShapeLayer")
    parser.add_argument("--D", type=int, default=16, help="Network depth for ShapeLayer")
    parser.add_argument("--W", type=int, default=256, help="Network width for ShapeLayer")
    parser.add_argument("--feat_embed", action="store_true", help="Whether to use position embedding")
    parser.add_argument("--feat_freq", type=int, default=11, help="Position encoding dimension for ShapeLayer")
    parser.add_argument("--pos_embed", action="store_true", help="Whether to use position embedding")
    parser.add_argument("--pos_freq", type=int, default=8, help="Position encoding dimension for ShapeLayer")
    parser.add_argument("--delete_mode", action="store_true", help="Whether to use position embedding")
    parser.add_argument("--time_embed", action="store_true", help="Whether to use time embedding")
    parser.add_argument("--time_freq", type=int, default=4, help="Time encoding dimension for ShapeLayer")    
    # Image dimensions
    parser.add_argument('--img_h', type=int, default=540, help='Size of the input image.')  # 原来是 480
    parser.add_argument('--img_w', type=int, default=960, help='Size of the input image.')  # 原来是 854
    
    # Training parameters
    parser.add_argument("--deform_lr", type=float, default=0.001, help="Learning rate for deform_network")
    parser.add_argument("--shapelayer_lr", type=float, default=0.001, help="Learning rate for ShapeLayer")
    parser.add_argument("--num_epochs", type=int, default=10000, help="Number of training epochs")
    parser.add_argument("--print_freq", type=int, default=50, help="Print frequency")
    
    # MFN related parameters
    parser.add_argument("--use_mfn", action="store_true", help="Whether to use multi fourier network")
    parser.add_argument("--multi_hidden", action="store_true", help="Whether to use multi hidden network")
    parser.add_argument("--n_layers_hidden", type=int, default=2, help="Network depth for multi hidden network")
    parser.add_argument("--use_relu", action="store_true", help="Whether to use relu")
    parser.add_argument("--f_scale", type=float, default=0.1, help="multi fourier network input scale")
    parser.add_argument("--vis_freq", type=int, default=1000, help="Visualization frequency")
    parser.add_argument("--save_freq", type=int, default=500, help="Checkpoint saving frequency")
    parser.add_argument("--points_per_epoch", type=int, default=150000, help="Number of points to sample per epoch")
    
    args = parser.parse_args()
    return args

def sample_points_from_keyframes(all_frame_data, num_points, keyframe_indices, device, layer_weights=None):
    """
    只从关键帧中采样点，支持为不同层设置不同的采样权重
    
    Args:
        all_frame_data: 所有帧的数据列表
        num_points: 总共要采样的点数
        keyframe_indices: 要采样的关键帧索引列表
        device: 设备（如 'cuda' 或 'cpu'）
        layer_weights: 每个层的采样权重列表，如[0.7, 0.3]表示前景层采样70%的点
    
    Returns:
        sampled_data: 包含采样点的字典
    """
    sampled_data = {
        'rgb_inmask': [],
        'coord_inmask': [],
        'pixel_vertices': [],
        'pixel_vertice_coords': [],
        'barycentric_coords': [],
        'frame_indices': []
    }

    # 如果没有指定layer权重，则平均分配
    if layer_weights is None:
        num_layers = len(all_frame_data[0])  # 每帧数据中的层数
        layer_weights = [1.0 / num_layers] * num_layers

    # 标准化权重
    layer_weights = torch.tensor(layer_weights, device=device)
    layer_weights = layer_weights / layer_weights.sum()
    
    # 计算每层要采样的点数
    points_per_layer = (num_points * layer_weights).int()
    points_per_layer[-1] += num_points - points_per_layer.sum()  # 确保总数正确

    # 对每一层分别处理
    for layer_idx, layer_points in enumerate(points_per_layer):
        layer_total_points = 0
        layer_points_per_frame = []
        
        # 计算该层在关键帧中的总点数
        for frame_idx in keyframe_indices:
            layer_frame_points = all_frame_data[frame_idx][layer_idx][0].shape[0]
            layer_total_points += layer_frame_points
            layer_points_per_frame.append(layer_frame_points)
            
        # 对该层的每个关键帧采样
        for i, frame_idx in enumerate(keyframe_indices):
            layer_data = all_frame_data[frame_idx][layer_idx]  # 获取当前层的数据
            n_to_sample = int(layer_points * (layer_points_per_frame[i] / layer_total_points))
            
            if n_to_sample == 0:
                continue
                
            num_points_in_frame = layer_data[0].shape[0]
            if n_to_sample < num_points_in_frame:
                indices = torch.randperm(num_points_in_frame, device=device)[:n_to_sample]
            else:
                indices = torch.arange(num_points_in_frame, device=device)
            
            # 添加采样的点
            sampled_data['rgb_inmask'].append(layer_data[0][indices])
            sampled_data['coord_inmask'].append(layer_data[1][indices])
            sampled_data['pixel_vertices'].append(layer_data[2][indices])
            sampled_data['pixel_vertice_coords'].append(layer_data[3][indices])
            sampled_data['barycentric_coords'].append(layer_data[4][indices])
            sampled_data['frame_indices'].append(torch.full((indices.shape[0],), frame_idx, device=device))
    
    # 合并所有采样的点
    for key in sampled_data:
        sampled_data[key] = torch.cat(sampled_data[key], dim=0)
    
    return sampled_data

def sample_points_from_all_frames(all_frame_data, num_points, device, layer_weights=None):
    """
    从所有帧中采样点，支持为不同层设置不同的采样权重
    
    Args:
        all_frame_data: 所有帧的数据列表
        num_points: 总共要采样的点数
        device: 设备（如 'cuda' 或 'cpu'）
        layer_weights: 每个层的采样权重列表，如[0.7, 0.3]表示前景层采样70%的点
    
    Returns:
        sampled_data: 包含采样点的字典

    """
    sampled_data = {
        'rgb_inmask': [],
        'coord_inmask': [],
        'pixel_vertices': [],
        'pixel_vertice_coords': [],
        'barycentric_coords': [],
        'frame_indices': []
    }

    # 如果没有指定layer权重，则平均分配
    if layer_weights is None:
        num_layers = len(all_frame_data[0])  # 每帧数据中的层数
        layer_weights = [1.0 / num_layers] * num_layers
    
    # 标准化权重
    layer_weights = torch.tensor(layer_weights, device=device)
    layer_weights = layer_weights / layer_weights.sum()
    
    # 计算每层要采样的点数
    points_per_layer = (num_points * layer_weights).int()
    points_per_layer[-1] += num_points - points_per_layer.sum()  # 确保总数正确
    
    for layer_idx, layer_points in enumerate(points_per_layer):
        layer_total_points = 0
        layer_points_per_frame = []
        
        # 计算该层在所有帧中的总点数
        for frame_data in all_frame_data:
            layer_frame_points = frame_data[layer_idx][0].shape[0]
            layer_total_points += layer_frame_points
            layer_points_per_frame.append(layer_frame_points)
        
        # 对该层的每一帧采样
        for frame_idx, frame_data in enumerate(all_frame_data):
            layer_data = frame_data[layer_idx]  # 获取当前层的数据
            n_to_sample = int(layer_points * (layer_points_per_frame[frame_idx] / layer_total_points))
            
            if n_to_sample == 0:
                continue
                
            num_points_in_frame = layer_data[0].shape[0]
            if n_to_sample < num_points_in_frame:
                indices = torch.randperm(num_points_in_frame, device=device)[:n_to_sample]
            else:
                indices = torch.arange(num_points_in_frame, device=device)
            
            # 添加采样的点
            sampled_data['rgb_inmask'].append(layer_data[0][indices])
            sampled_data['coord_inmask'].append(layer_data[1][indices])
            sampled_data['pixel_vertices'].append(layer_data[2][indices])
            sampled_data['pixel_vertice_coords'].append(layer_data[3][indices])
            sampled_data['barycentric_coords'].append(layer_data[4][indices])
            sampled_data['frame_indices'].append(torch.full((indices.shape[0],), frame_idx, device=device))
    
    # 合并所有采样的点
    for key in sampled_data:
        sampled_data[key] = torch.cat(sampled_data[key], dim=0)
    
    return sampled_data
        
def sample_points_from_edited_frames(edit_frame_data, num_points, device):
    sampled_data = {
        'rgb_inmask': [],
        'coord_inmask': [],
        'pixel_vertices': [],
        'pixel_vertice_coords': [],
        'barycentric_coords': [],
        'frame_indices': []
    }

    total_points = 0
    points_per_frame = []
    
    for _, frame_data in enumerate(edit_frame_data):
        num_frame_points = frame_data[0].shape[0]
        total_points += num_frame_points
        points_per_frame.append(num_frame_points)
    
    for i, frame_data in enumerate(edit_frame_data):
        n_to_sample = int(num_points * (points_per_frame[i] / total_points))
        
        if n_to_sample == 0:
            continue

        num_points_in_frame = frame_data[0].shape[0]
        if n_to_sample < num_points_in_frame:
            indices = torch.randperm(num_points_in_frame, device=device)[:n_to_sample]
        else:
            indices = torch.arange(num_points_in_frame, device=device)

        sampled_data['rgb_inmask'].append(frame_data[0][indices])
        sampled_data['coord_inmask'].append(frame_data[1][indices])
        sampled_data['pixel_vertices'].append(frame_data[2][indices])
        sampled_data['pixel_vertice_coords'].append(frame_data[3][indices])
        sampled_data['barycentric_coords'].append(frame_data[4][indices])
        sampled_data['frame_indices'].append(torch.full((indices.shape[0],), i, device=device))
        
    for key in sampled_data:
        sampled_data[key] = torch.cat(sampled_data[key], dim=0)
    
    return sampled_data
    

def visualize_sampled_points(sampled_data, pred_colors, epoch, args):
    """
    可视化来自不同帧的采样点
    
    Args:
        sampled_data: 采样的点数据
        pred_colors: 预测的颜色
        epoch: 当前训练轮次
        args: 参数
    """
    # 创建目录
    vis_dir = os.path.join(args.output_dir, f'points_vis/epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 选择最多8个不同的帧来可视化
    unique_frames = torch.unique(sampled_data['frame_indices']).cpu().numpy()
    unique_frames = unique_frames[:min(8, len(unique_frames))]
    
    plt.figure(figsize=(12, 8))
    
    for i, frame_idx in enumerate(unique_frames):
        # 获取当前帧的点
        mask = sampled_data['frame_indices'] == frame_idx
        frame_coords = sampled_data['coord_inmask'][mask].cpu().numpy()
        frame_colors = pred_colors[mask].detach().cpu().numpy()
        
        # 创建散点图
        plt.subplot(2, 4, i+1)
        plt.scatter(
            frame_coords[:, 0] * args.img_w, 
            frame_coords[:, 1] * args.img_h, 
            c=frame_colors, 
            s=5, 
            alpha=0.7
        )
        plt.title(f'Frame {frame_idx}')
        plt.xlim(0, args.img_w)
        plt.ylim(args.img_h, 0)  # 翻转y轴以匹配图像坐标
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'sampled_points.png'))
    plt.close()

def visualize_points_distribution(sampled_data, epoch, args):
    """
    可视化采样点在不同帧之间的分布
    
    Args:
        sampled_data: 采样的点数据
        epoch: 当前训练轮次
        args: 参数
    """
    # 创建目录
    vis_dir = os.path.join(args.output_dir, f'points_vis/epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 计算每帧的点数
    frame_indices = sampled_data['frame_indices'].cpu().numpy()
    frames, counts = np.unique(frame_indices, return_counts=True)
    
    # 创建柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(frames, counts, width=0.6)
    plt.title('点采样分布')
    plt.xlabel('帧索引')
    plt.ylabel('点数')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数值标签
    for i, (frame, count) in enumerate(zip(frames, counts)):
        plt.text(frame, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'points_distribution.png'))
    plt.close()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    #debug
    args.feat_embed = True
    h,w = args.img_h, args.img_w
    # Create timestamped output directories
    time_ = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    video_dir = args.video_path
    
    if args.output_dir == "./output":  # If using default output dir
        args.output_dir = os.path.join('./output', time_ + video_name)
    if args.checkpoint_dir == "./checkpoints":  # If using default checkpoint dir
        args.checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    if args.mask_dir == "./masks":  
        args.mask_dir = os.path.join(args.output_dir, 'masks')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    
    # Save run configuration
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')


    # Load jpg frames from directory instead of video
    image_files = sorted(glob.glob(os.path.join(args.video_path, "*.jpg")))
    gt_frames = []
    for img_path in image_files:
        img = PIL.Image.open(img_path).convert("RGB")
        img = img.resize((args.img_w, args.img_h), PIL.Image.BILINEAR)
        img_np = np.array(img)
        gt_frames.append(img_np)
    gt_frames = np.stack(gt_frames, axis=0)

    from tools.sam2.sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "/data4/cy/ShapeLayerVideo/tools/sam2/checkpoints/sam2.1_hiera_large.pt",
        device=device
    )

    # Set up video segmentation
    inference_state = predictor.init_state(video_path=video_dir)
    
    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1 # give a unique id to each object we interact with (it can be any integers)
    # Let's add a positive click at (x, y) = (210, 350) to get started
    #points = np.array([[1000,420],[1000,600],[900,780],[1150,600]], dtype=np.float32)
    points = np.array([[385,224]], dtype=np.float32)
    # points = np.array([[1600,600],[1700,680],[1680,680],[1600,700]], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    #labels = np.array([1,1,1,1], np.int32)
    labels = np.array([1], np.int32)
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        # 将mask调整到目标尺寸
        video_segments[out_frame_idx] = {
            out_obj_id: cv2.resize(
                (out_mask_logits[i] > 0.0).cpu().numpy().squeeze().astype(np.float32), 
                (args.img_w, args.img_h), 
                interpolation=cv2.INTER_LINEAR
            ) > 0.5
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    num_frames = gt_frames.shape[0]
    

    results = np.load(args.vertices_tracks_path, allow_pickle=True).item()
    original_vertices_list = results['original_vertices_list']
    new_vertices_list = results['new_vertices_list']
    original_visible_list = results['original_visible_list']
    new_visible_list = results['new_visible_list']
    # Layer 1: Combine original and new vertices
    tracks = np.concatenate((original_vertices_list[0], new_vertices_list[0]), axis=1)
    visibles = np.concatenate((original_visible_list[0], new_visible_list[0]), axis=1)
    start_vertice = 0  # First layer starts at 0
    vertices_tracks_list = []
    visible_list = []
    vertices_tracks_list.append(torch.tensor(tracks).to(device))
    visible_list.append(torch.tensor(visibles).to(device))

    # Layer 2: Use first layer's total vertices as start_vertice
    tracks = np.concatenate((original_vertices_list[1], new_vertices_list[1]), axis=1)
    visibles = np.concatenate((original_visible_list[1], new_visible_list[1]), axis=1)
    start_vertice = vertices_tracks_list[0].shape[1]  # Set start_vertice for second layer
    vertices_tracks_list.append(torch.tensor(tracks).to(device))
    visible_list.append(torch.tensor(visibles).to(device))
    
    total_vertices = sum([vertices.shape[1] for vertices in vertices_tracks_list])

    combined_mapper = {i:i for i in range(total_vertices)}
    
    shapelayer_net = init_shapelayerX(combined_mapper, args)

    # ====== 模仿outpainting.py的全局坐标体系转换 ======
    print("处理顶点坐标系统（模仿outpainting.py）...")
    # 计算所有layer所有帧的全局最小/最大x/y
    min_x_global = min([vertices[:,:,0].min().item() for vertices in vertices_tracks_list])
    min_y_global = min([vertices[:,:,1].min().item() for vertices in vertices_tracks_list])
    max_x_global = max([vertices[:,:,0].max().item() for vertices in vertices_tracks_list])
    max_y_global = max([vertices[:,:,1].max().item() for vertices in vertices_tracks_list])
    print(f"全局 min_x: {min_x_global}, min_y: {min_y_global}")
    print(f"全局 max_x: {max_x_global}, max_y: {max_y_global}")

    # 重新计算 h, w 以匹配 mask/vertices 的最大范围
    global_h = int(np.ceil(max_y_global - min_y_global))
    global_w = int(np.ceil(max_x_global - min_x_global))
    print(f"全局图像尺寸: {global_w}x{global_h}")

    # 所有layer统一减去全局最小值
    for i in range(len(vertices_tracks_list)):
        vertices_tracks_list[i][:,:,0] = vertices_tracks_list[i][:,:,0] - min_x_global
        vertices_tracks_list[i][:,:,1] = vertices_tracks_list[i][:,:,1] - min_y_global
        print(f"Layer {i} - 坐标已统一调整")

    # edit_frame_index =[0]
    # Pre-process all frames for training
    print("Pre-processing all frames...")
    all_frame_data = []
    edit_frame_data = []
    for frame_idx in tqdm.tqdm(range(num_frames)):
        frame_deformed_vertices_list = [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
        frame_data = process_single_frame(
            frame_idx,
            frame_deformed_vertices_list,
            device,
            video_segments[frame_idx],
            gt_frames[frame_idx],
            args,
            visible_list=visible_list,
            is_training=True
        )
        all_frame_data.append(frame_data)
        del frame_data  # 释放内存
        torch.cuda.empty_cache()
        
    # for i,frame_idx in enumerate(edit_frame_index):
    #     frame_deformed_vertices_list = [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
    #     # 归一化frame_deformed_vertices_list到全局坐标系
    #     frame_deformed_vertices_list = [v.clone() for v in frame_deformed_vertices_list]
    #     frame_data = process_single_frame(
    #         frame_idx,
    #         frame_deformed_vertices_list,
    #         device,
    #         video_segments[frame_idx],
    #         edit_frames[i],
    #         args,
    #         visible_list=visible_list
    #     )
    #     edit_frame_data.append(frame_data)
    # #     torch.cuda.empty_cache()

    for frame_idx in range(num_frames):
        frame_data = all_frame_data[frame_idx]
        current_frame_vertices= [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
        visualize_vertices(current_frame_vertices, frame_idx, 0, args)
        # visualize_triangles(current_frame_vertices, frame_data[7], 0, args, frame_idx,visible_list=visible_list)
    # Set up optimizer
    shapelayer_optimizer = optim.Adam(shapelayer_net.parameters(), lr=args.shapelayer_lr)

    print("开始分阶段训练...")
    
   
    points_per_epoch = args.points_per_epoch  # 每轮采样的点数
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(device) / 1024**2} MB")

    for epoch in tqdm.tqdm(range(args.num_epochs)):
        torch.set_grad_enabled(True)
        epoch_loss = 0  # 仅用于记录，不参与计算
        sampled_data = sample_points_from_all_frames(all_frame_data, points_per_epoch, device,layer_weights=[0.8,0.2])

        
                # 前向传播和反向传播
        shapelayer_optimizer.zero_grad()
        
        
        # 分层计算损失
        total_loss = 0
        layer_losses = []
        
        # 根据采样数据计算每层的损失
        if args.time_embed:
            # 归一化时间到[0, 1]
            frame = sampled_data['frame_indices'][:, None].float() / (len(all_frame_data) - 1)
        else:
            # 如果不使用时间嵌入，则不需要归一化时间
            frame = None                
        # 使用采样的点进行前向传播
        pixel_colors_pred = shapelayer_net(
            sampled_data['pixel_vertices'],
            sampled_data['pixel_vertice_coords'],
            sampled_data['coord_inmask'],
            sampled_data['barycentric_coords'],
            frame_idx=frame
        )
        
        # 计算损失
        if not (torch.isnan(pixel_colors_pred).any() or torch.isinf(pixel_colors_pred).any()):
            loss = compute_loss(pixel_colors_pred, sampled_data['rgb_inmask'])
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                # 梯度裁剪并反向传播
                torch.nn.utils.clip_grad_norm_(shapelayer_net.parameters(), max_norm=1.0)
                loss.backward()
                shapelayer_optimizer.step()
                
                epoch_loss = loss.item()

            else:
                print(f"Warning: Invalid loss detected in epoch {epoch}")
        else:
            print(f"Warning: Invalid predictions detected in epoch {epoch}")
        
    
        del pixel_colors_pred, sampled_data
        if 'loss' in locals():
            del loss
            
        epoch_psnr = 10 * math.log10(1 / epoch_loss)
        # 打印每轮的平均loss
        if epoch % args.print_freq == 0:
            print(f"Phase 2 - Epoch {epoch}, Loss: {epoch_loss:.6f}")
            print(f"Phase 2 - Epoch {epoch}, PSNR: {epoch_psnr:.2f} dB")

        # 保存检查点
        if epoch % args.save_freq == 0:
            save_checkpoints(shapelayer_net, epoch, args)

        # 可视化完整帧结果
        if epoch % args.vis_freq == 0:
            with torch.no_grad():
                for frame_idx in [0, 8]:  # 只对部分帧进行可视化
                    frame_data = all_frame_data[frame_idx]
                    current_frame_vertices = [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
                    
                    
                    visualize_results(frame_data, frame_idx, epoch, args, shapelayer_net)
                    
                    torch.cuda.empty_cache()

    save_checkpoints(shapelayer_net, args.num_epochs,args)
    # shapelayer_optimizer = optim.Adam(shapelayer_net.parameters(), lr=0.0001)

    # print("Phase 3: Training edited frames...")
    # for epoch in tqdm.tqdm(range(300)):
    #     torch.set_grad_enabled(True)
    #     epoch_loss = 0  # 仅用于记录，不参与计算
    #     # 采样来自多个帧的点
    #     sampled_data = sample_points_from_edited_frames(edit_frame_data, points_per_epoch, device)
        
    #     # 前向传播和反向传播
    #     shapelayer_optimizer.zero_grad()
        
    #     # 使用采样的点进行前向传播
    #     pixel_colors_pred = shapelayer_net(
    #         sampled_data['pixel_vertices'],
    #         sampled_data['pixel_vertice_coords'],
    #         sampled_data['coord_inmask'],
    #         sampled_data['barycentric_coords']
    #     )
        
    #     # 计算损失
    #     if not (torch.isnan(pixel_colors_pred).any() or torch.isinf(pixel_colors_pred).any()):
    #         loss = compute_loss(pixel_colors_pred, sampled_data['rgb_inmask'])
            
    #         if not (torch.isnan(loss) or torch.isinf(loss)):
    #             # 梯度裁剪并反向传播
    #             torch.nn.utils.clip_grad_norm_(shapelayer_net.parameters(), max_norm=1.0)
    #             loss.backward()
    #             shapelayer_optimizer.step()
                
    #             epoch_loss = loss.item()
    #         else:
    #             print(f"Warning: Invalid loss detected in epoch {epoch}")
    #     else:
    #         print(f"Warning: Invalid predictions detected in epoch {epoch}")
        
    
    # #     del pixel_colors_pred, sampled_data
    # #     if 'loss' in locals():
    # #         del loss
    # #     torch.cuda.empty_cache()
        
    # #     # 打印每轮的平均loss
    # #     if epoch % args.print_freq == 0:
    # #         print(f"Phase 2 - Epoch {epoch}, Loss: {epoch_loss:.6f}")
        
    # #     # 保存检查点
    # #     if epoch % args.save_freq == 0:
    # #         save_checkpoints(shapelayer_net, epoch, args)
        
    # #     # 可视化完整帧结果
    # #     if epoch % args.vis_freq == 0:
    # #         with torch.no_grad():
    # #             for i , frame_idx in enumerate(edit_frame_index):
    # #                 frame_data = edit_frame_data[i]
    # #                 current_frame_vertices = [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
                    
    # #                 pixel_colors_pred = shapelayer_net(
    # #                     frame_data[2],
    # #                     frame_data[3],
    # #                     frame_data[1],
    # #                     frame_data[4],
    # #                 )
                    
    # #                 visualize_results(pixel_colors_pred, frame_data[6], frame_data[5], frame_idx, epoch, args)
                    
    # #                 # Clean up
    # #                 del pixel_colors_pred
    # #                 torch.cuda.empty_cache()

    # print("Training completed!")
    # Generate final visualization video
    print("Generating final visualization video...")
    final_frames_dir = os.path.join(args.output_dir, 'final_frames')
    os.makedirs(final_frames_dir, exist_ok=True)
    # 创建日志文件
    log_file = os.path.join(args.output_dir, 'metrics.log')
    with open(log_file, 'w') as f:
        # 写入网络参数量信息
        param_count = count_parameters(shapelayer_net)
        f.write(f"ShapeLayer网络参数量: {param_count:,}\n\n")
        f.write("帧评估指标:\n")
        f.write("帧索引,PSNR,LPIPS,SSIM\n")
    
    # 用于计算平均指标
    all_metrics = {
        'psnr': [],
        'lpips': [],
        'ssim': []
    }
    
    with torch.no_grad():
        for frame_idx in tqdm.tqdm(range(num_frames)):
            frame_deformed_vertices_list = [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
            frame_data = process_single_frame(
                frame_idx,
                frame_deformed_vertices_list,
                device,
                video_segments[frame_idx],
                gt_frames[frame_idx],
                args,
                visible_list=visible_list,
                delete_mode=args.delete_mode)
        
            
    
            # Save frame visualization
            frame_path = os.path.join(final_frames_dir, f'frame_{frame_idx:04d}.png')
            h, w = args.img_h, args.img_w
            vis_image = torch.zeros((h*w, 3)).to(device)
             # 用于计算指标的GT图像
            gt_frame_tensor = torch.from_numpy(gt_frames[frame_idx]).float() / 255.0
            gt_frame_tensor = gt_frame_tensor.to(device).reshape(h, w, 3)
            

            if args.time_embed:
                frame = torch.tensor(frame_idx / (num_frames - 1), device=device).float().view(1, 1).repeat(w*h, 1)
            else:
                frame = None             # 分别处理每个layer的预测
            for layer_idx in range(2):  # 前景和背景两层
                if args.delete_mode and layer_idx == 1:
                    continue
                if args.time_embed:
                    layer_frame = frame[:frame_data[layer_idx][2].shape[0]]
                else:
                    layer_frame = None
                # Process vertices in two batches to reduce memory usage
                batch_size = frame_data[layer_idx][2].shape[0] // 2
                if batch_size > 0:
                    # First batch
                    layer_colors_pred1 = shapelayer_net(
                        frame_data[layer_idx][2][:batch_size],  # pixel_vertices 
                        frame_data[layer_idx][3][:batch_size],  # pixel_vertice_coords
                        frame_data[layer_idx][1][:batch_size],  # coord_inmask
                        frame_data[layer_idx][4][:batch_size],  # barycentric_coords
                        frame_idx=layer_frame[:batch_size] if layer_frame is not None else None
                    )
                    
                    # Second batch
                    layer_colors_pred2 = shapelayer_net(
                        frame_data[layer_idx][2][batch_size:],  # pixel_vertices 
                        frame_data[layer_idx][3][batch_size:],  # pixel_vertice_coords
                        frame_data[layer_idx][1][batch_size:],  # coord_inmask
                        frame_data[layer_idx][4][batch_size:],  # barycentric_coords
                        frame_idx=layer_frame[batch_size:] if layer_frame is not None else None
                    )
                    
                    # Combine results
                    layer_colors_pred = torch.cat([layer_colors_pred1, layer_colors_pred2], dim=0)
                    
                    # Clean up intermediates
                    del layer_colors_pred1, layer_colors_pred2
                else:
                    # If the batch is too small, process it in one go
                    layer_colors_pred = shapelayer_net(
                        frame_data[layer_idx][2],  # pixel_vertices 
                        frame_data[layer_idx][3],  # pixel_vertice_coords
                        frame_data[layer_idx][1],  # coord_inmask
                        frame_data[layer_idx][4],  # barycentric_coords
                        frame_idx=layer_frame
                    )
                if args.delete_mode :
                    vis_image=layer_colors_pred
                else:
                    mask = frame_data[layer_idx][6]  # 获取当前层的mask
                    vis_image[mask.flatten() == 0] = layer_colors_pred
                
                del layer_colors_pred
            # 将可视化图像重塑为HWC格式，用于计算指标
            vis_image_reshaped = vis_image.reshape(h, w, 3)
            
            # 计算评估指标
            psnr, lpips_value, ssim_value = compute_metrics(vis_image_reshaped, gt_frame_tensor)
            
            # 添加到总指标中
            all_metrics['psnr'].append(psnr)
            all_metrics['lpips'].append(lpips_value)
            all_metrics['ssim'].append(ssim_value)
            
            # 将指标写入日志
            with open(log_file, 'a') as f:
                f.write(f"{frame_idx},{psnr:.4f},{lpips_value:.4f},{ssim_value:.4f}\n")
            
                        # Generate vertices and triangles visualization
            current_frame_vertices = [vertices_tracks[frame_idx,:,:] for vertices_tracks in vertices_tracks_list]
            
          
            # Create triangles visualization directory
            triangles_dir = os.path.join(args.output_dir, 'triangles_vis')
            os.makedirs(triangles_dir, exist_ok=True)

            # Save RGB visualization
            vis_image_np = vis_image.cpu().detach().numpy().reshape(h, w, 3)
            plt.imsave(frame_path, vis_image_np)
            
            # visualize_triangles(
            #     current_frame_vertices,
            #     frame_data[7],
            #     -1,
            #     args,
            #     frame_idx,
            #     visible_list=visible_list
            # )
            del frame_data,vis_image, vis_image_np
            del current_frame_vertices
            torch.cuda.empty_cache()

            # 计算并写入平均指标
        avg_psnr = sum(all_metrics['psnr']) / len(all_metrics['psnr'])
        avg_lpips = sum(all_metrics['lpips']) / len(all_metrics['lpips'])
        avg_ssim = sum(all_metrics['ssim']) / len(all_metrics['ssim'])
        
        with open(log_file, 'a') as f:
            f.write("\n平均评估指标:\n")
            f.write(f"平均PSNR: {avg_psnr:.4f}\n")
            f.write(f"平均LPIPS: {avg_lpips:.4f}\n")
            f.write(f"平均SSIM: {avg_ssim:.4f}\n")


        # Create final videos using ffmpeg
        output_video = os.path.join(args.output_dir, 'final_visualization.mp4')
        output_triangles = os.path.join(args.output_dir, 'triangles_visualization.mp4')
        
        # Command for rendered frames video
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', '12',
            '-i', os.path.join(final_frames_dir, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_video
        ]
        
        # Command for triangles visualization video
        ffmpeg_triangles_cmd = [
            'ffmpeg',
            '-y',
            '-framerate', '12',
            '-i', os.path.join(args.output_dir, 'triangles-1layer0/frame%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            output_triangles
        ]
       
        subprocess.run(ffmpeg_cmd)
        