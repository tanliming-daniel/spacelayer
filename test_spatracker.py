import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.append(os.path.join(project_root, "submodules/SpaTrackerV2"))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
except:
    print("Warning: SpaTrackerV2 not found")
import pycolmap
from submodules.SpaTrackerV2.models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_mode", type=str, default="offline")
    parser.add_argument("--data_type", type=str, default="RGB")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--video_name", type=str, default="car-turn")
    parser.add_argument("--grid_size", type=int, default=20)
    parser.add_argument("--vo_points", type=int, default=1500)
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args(
    )
    
if __name__ == "__main__":
    args = parse_args()
    out_dir = "results"
    # fps
    fps = int(args.fps)
    mask_dir = args.data_dir + f"/{args.video_name}.png"
    
    vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")

    if args.data_type == "RGBD":
        npz_dir = args.data_dir + f"/{args.video_name}.npz"
        data_npz_load = dict(np.load(npz_dir, allow_pickle=True))
        #TODO: tapip format
        video_tensor = data_npz_load["video"] * 255
        video_tensor = torch.from_numpy(video_tensor)
        video_tensor = video_tensor[::fps]
        depth_tensor = data_npz_load["depths"]
        depth_tensor = depth_tensor[::fps]
        intrs = data_npz_load["intrinsics"]
        intrs = intrs[::fps]
        extrs = np.linalg.inv(data_npz_load["extrinsics"])
        extrs = extrs[::fps]
        unc_metric = None
    elif args.data_type == "RGB":
        # 检查是否为图片序列（%05d.jpg）
        img_pattern = os.path.join(args.data_dir, f"{args.video_name}", "*.jpg")
        img_files = sorted(glob.glob(img_pattern))
        if len(img_files) > 0:
            # 读取所有图片，拼成视频张量
            imgs = [cv2.imread(f) for f in img_files]
            imgs = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im is not None else None for im in imgs]
            imgs = [im for im in imgs if im is not None]
            video_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float()  # (N, C, H, W)
            video_tensor = video_tensor[::fps]
        else:
            vid_dir = os.path.join(args.data_dir, f"{args.video_name}.mp4")
            video_reader = decord.VideoReader(vid_dir)
            video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)  # (N, C, H, W)
            video_tensor = video_tensor[::fps].float()

        # process the image tensor
        video_tensor = preprocess_image(video_tensor)[None]
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Predict attributes including cameras, depth maps, and point maps.
                predictions = vggt4track_model(video_tensor.cuda()/255)
                extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
                depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]
        
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        #NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

        data_npz_load = {}
    
    if os.path.exists(mask_dir):
        mask_files = mask_dir
        mask = cv2.imread(mask_files)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy())>0
        
    # get all data pieces
    viz = True
    os.makedirs(out_dir, exist_ok=True)
        
   
    if args.track_mode == "offline":
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
    else:
        model = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")

    # config the model; the track_num is the number of points in the grid
    model.spatrack.track_num = args.vo_points
    
    model.eval()
    model.to("cuda")
    viser = Visualizer(save_dir=out_dir, grayscale=True, 
                     fps=10, pad_value=0, tracks_leave_trace=5)
    
    grid_size = args.grid_size

    # get frame H W
    if video_tensor is None or video_tensor.shape[0] == 0:
        raise ValueError("video_tensor is empty! 请检查视频或图片序列输入是否正确。")
    else:
        frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    # Sample mask values at grid points and filter out points where mask=0
    if os.path.exists(mask_dir):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    # query_xyt = torch.cat([torch.ones_like(grid_pts[:, :, :1])*(video_tensor.shape[0]-1), grid_pts], dim=2)[0].numpy()
    queries_first = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)
    queries_last = torch.cat([torch.ones_like(grid_pts[:, :, :1]) * (video_tensor.shape[0] - 1), grid_pts], dim=2)
    query_xyt = torch.cat([queries_first, queries_last], dim=1)[0].numpy()

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=True, iters_track=4,
                            query_no_BA=False, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        print("track2d_pred shape:", track2d_pred.shape)
        print("grid_pts shape:", grid_pts.shape)

        # resize the results to avoid too large I/O Burden
        # depth and image, the maximum side is 336
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                if isinstance(depth_tensor, torch.Tensor):
                    depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
                else:
                    depth_tensor = T.Resize((new_h, new_w))(torch.from_numpy(depth_tensor))


        if viz:
            viser.visualize(video=video[None],
                                tracks=track2d_pred[None][...,:2],
                                visibility=vis_pred[None],filename="test")

        # save as the tapip3d format   
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        depth_save = point_map[:,2,...]
        depth_save[conf_depth<0.5] = 0
        data_npz_load["depths"] = depth_save.cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["unc_metric"] = conf_depth.cpu().numpy()
        np.savez(os.path.join(out_dir, f'result.npz'), **data_npz_load)

        print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {out_dir}/result.npz[/bold yellow]")
