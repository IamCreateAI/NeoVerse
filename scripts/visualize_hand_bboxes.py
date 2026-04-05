"""
Visualize hand bounding boxes on video frames.

Uses the exact same dataset, data loading, and bbox computation as training
to ensure what you see matches what the model receives during training.

Also overlays the projected MANO mesh silhouette so you can verify the
bounding boxes correctly enclose each hand.

Usage:
    python -m scripts.visualize_hand_bboxes --config configs/train_hand_head.yaml
    python -m scripts.visualize_hand_bboxes --config configs/train_hand_head.yaml --num_clips 5 --frames_per_clip 8
    python -m scripts.visualize_hand_bboxes --config configs/train_hand_head.yaml --save_video
    python -m scripts.visualize_hand_bboxes --config configs/train_hand_head.yaml --show_mesh
"""

import argparse
import bisect
import json
import os
import random

import cv2
import numpy as np
import torch
import yaml
from decord import VideoReader

from scripts.train_hand_head import (
    HOT3DHandDataset,
    discover_sequences,
)

LEFT_COLOR = (255, 150, 50)   # blue-ish (BGR)
RIGHT_COLOR = (50, 50, 255)   # red-ish (BGR)
LEFT_MESH_COLOR = (255, 200, 120)   # lighter blue for mesh
RIGHT_MESH_COLOR = (120, 120, 255)  # lighter red for mesh


def draw_bboxes_on_frame(img_tensor, bboxes, valid, mesh_pixels=None):
    """Draw hand bounding boxes on a single frame.

    Args:
        img_tensor: [3, H, W] float tensor in [0, 1]
        bboxes: [2, 4] normalised bboxes (x1, y1, x2, y2)
        valid: [2] boolean mask
        mesh_pixels: optional list of 2 arrays, each [N, 2] projected mesh vertices

    Returns:
        BGR uint8 numpy image with bboxes drawn.
    """
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    colors = [LEFT_COLOR, RIGHT_COLOR]
    mesh_colors = [LEFT_MESH_COLOR, RIGHT_MESH_COLOR]
    labels = ["Left", "Right"]

    # Draw mesh silhouette first (underneath bboxes)
    if mesh_pixels is not None:
        for hand_idx in range(2):
            if mesh_pixels[hand_idx] is None:
                continue
            pixels = mesh_pixels[hand_idx]  # [N, 2] in original image coords
            # Scale from original image coords (1408x1408) to display resolution
            scale_x = W / 1408.0
            scale_y = H / 1408.0
            for i in range(len(pixels)):
                px = int(pixels[i, 0] * scale_x)
                py = int(pixels[i, 1] * scale_y)
                if 0 <= px < W and 0 <= py < H:
                    cv2.circle(img, (px, py), 1, mesh_colors[hand_idx], -1)

    # Draw bounding boxes
    for hand_idx in range(2):
        if not valid[hand_idx]:
            continue
        x1, y1, x2, y2 = bboxes[hand_idx].numpy()
        pt1 = (int(x1 * W), int(y1 * H))
        pt2 = (int(x2 * W), int(y2 * H))
        cv2.rectangle(img, pt1, pt2, colors[hand_idx], 2)

        # Draw center cross
        cx = int((x1 + x2) / 2 * W)
        cy = int((y1 + y2) / 2 * H)
        cv2.drawMarker(img, (cx, cy), colors[hand_idx], cv2.MARKER_CROSS, 10, 1)

        # Label with bbox size info
        bw = x2 - x1
        bh = y2 - y1
        label = f"{labels[hand_idx]} ({bw:.2f}x{bh:.2f})"
        cv2.putText(
            img, label,
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[hand_idx], 1, cv2.LINE_AA,
        )

    return img


def compute_mesh_projections_for_frame(seq_path, frame_i, n_video, hand_ts_sorted):
    """Project MANO meshes for a single frame to get 2D vertex positions.

    Returns a list of 2 arrays (left, right), each [N_valid, 2] or None.
    """
    from projectaria_tools.core.sophus import SE3
    from scripts.hand_vis_utils import (
        load_camera_calibration, load_headset_trajectory, find_closest,
        load_hand_poses, MANOModel, project_vertices,
    )

    calib_path   = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
    headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")
    jsonl_path   = os.path.join(seq_path, "hand_data", "mano_hand_pose_trajectory.jsonl")

    # Find MANO folder
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mano_folder = os.path.join(repo_root, "models", "MANO")

    for p in [calib_path, headset_path, jsonl_path, mano_folder]:
        if not os.path.exists(p):
            return [None, None]

    T_device_camera, cam_calib = load_camera_calibration(calib_path)
    headset_poses = load_headset_trajectory(headset_path)
    headset_ts = sorted(headset_poses.keys())

    hand_poses_data = load_hand_poses(jsonl_path)
    hand_ts_data = sorted(hand_poses_data.keys())

    mano_model = MANOModel(mano_folder)

    ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]
    frac = frame_i / max(n_video - 1, 1)
    query_tc = int(ts_start + frac * (ts_end - ts_start))

    closest_ht = find_closest(headset_ts, query_tc)
    t_wd, q_wd = headset_poses[closest_ht]
    T_world_device = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0]

    closest_hand_ts = find_closest(hand_ts_data, query_tc)
    hand_data = hand_poses_data[closest_hand_ts]

    result = [None, None]
    for hand_idx in range(2):
        hand_key = str(hand_idx)
        is_right = hand_idx == 1
        if hand_key not in hand_data or not hand_data[hand_key]:
            continue
        try:
            vertices, _faces = mano_model.get_mesh(hand_data[hand_key], is_right)
            pixels, depths, valid_mask = project_vertices(
                vertices, T_world_device, T_device_camera, cam_calib,
                image_width=1408,
            )
            if valid_mask.sum() >= 10:
                result[hand_idx] = pixels[valid_mask]
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Visualize hand bounding boxes on training frames")
    parser.add_argument("--config", required=True, help="Training config YAML (same as used for training)")
    parser.add_argument("--num_clips", type=int, default=3, help="Number of clips to visualize")
    parser.add_argument("--frames_per_clip", type=int, default=8, help="Max frames to show per clip")
    parser.add_argument("--output_dir", default="outputs/vis_hand_bboxes", help="Output directory for images")
    parser.add_argument("--save_video", action="store_true", help="Save each clip as a video instead of individual frames")
    parser.add_argument("--show_mesh", action="store_true", help="Overlay projected MANO mesh vertices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clip selection")
    parser.add_argument("--rescale_factor", type=float, default=None,
                        help="Override rescale factor from config (HaMeR default: 2.0)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    crop_cfg = cfg.get("hand_crop", {})

    data_root = data_cfg["data_root"]
    num_frames = data_cfg["num_frames"]
    res = tuple(data_cfg["resolution"])
    clip_stride = data_cfg.get("clip_stride", num_frames)

    rescale_factor = args.rescale_factor or crop_cfg.get("rescale_factor", 2.0)

    # Build dataset with hand crop enabled (same as training)
    all_seqs = discover_sequences(data_root)
    if not all_seqs:
        print(f"No sequences found in {data_root}")
        return

    print(f"Building dataset with rescale_factor={rescale_factor} ...")
    dataset = HOT3DHandDataset(
        all_seqs[:1],
        num_frames=num_frames,
        res=res,
        clip_stride=clip_stride,
        use_hand_crop=True,
        rescale_factor=rescale_factor,
    )
    print(f"Dataset: {len(dataset)} clips from {len(all_seqs)} sequences")

    if len(dataset) == 0:
        print("No clips found. Check data_root path and sequence contents.")
        return

    # Pre-load hand timestamps for mesh projection
    seq_hand_ts = {}
    for seq_path in all_seqs:
        jsonl_path = os.path.join(seq_path, "hand_data", "mano_hand_pose_trajectory.jsonl")
        if os.path.exists(jsonl_path):
            with open(jsonl_path) as f:
                ts_list = sorted(json.loads(line)["timestamp_ns"] for line in f)
            seq_hand_ts[seq_path] = ts_list

    # Sample random clips
    random.seed(args.seed)
    num_clips = min(args.num_clips, len(dataset))
    clip_indices = random.sample(range(len(dataset)), num_clips)

    os.makedirs(args.output_dir, exist_ok=True)

    for ci, clip_idx in enumerate(clip_indices):
        data = dataset[clip_idx]
        imgs = data["img"]             # [S, 3, H, W]
        bboxes = data["hand_bboxes"]   # [S, 2, 4]
        valid = data["hand_valid"]     # [S, 2]

        clip_info = dataset.clips[clip_idx]
        seq_name = os.path.basename(clip_info["seq_path"])
        seq_path = clip_info["seq_path"]
        frame_offset = clip_info["frame_offset"]
        n_show = min(args.frames_per_clip, imgs.shape[0])

        # Get video length for mesh projection
        n_video_frames = len(VideoReader(clip_info["video_path"]))
        hand_ts = seq_hand_ts.get(seq_path, [])

        print(f"Clip {ci+1}/{num_clips}: seq={seq_name}, offset={frame_offset}, "
              f"frames={imgs.shape[0]}, showing={n_show}")

        rendered_frames = []
        for fi in range(n_show):
            # Optionally compute mesh projections
            mesh_pixels = None
            if args.show_mesh and hand_ts:
                mesh_pixels = compute_mesh_projections_for_frame(
                    seq_path, frame_offset + fi, n_video_frames, hand_ts,
                )

            frame = draw_bboxes_on_frame(imgs[fi], bboxes[fi], valid[fi], mesh_pixels)

            # Add frame info text
            info = f"frame {frame_offset + fi} | rescale={rescale_factor}"
            cv2.putText(frame, info, (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

            rendered_frames.append(frame)

            if not args.save_video:
                out_path = os.path.join(args.output_dir, f"clip{ci:02d}_frame{fi:02d}.png")
                cv2.imwrite(out_path, frame)

        if args.save_video:
            H, W = rendered_frames[0].shape[:2]
            video_path = os.path.join(args.output_dir, f"clip{ci:02d}_{seq_name}.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 5, (W, H))
            for frame in rendered_frames:
                writer.write(frame)
            writer.release()
            print(f"  Saved video: {video_path}")

        if not args.save_video:
            # Also save a grid image for quick overview
            grid = np.concatenate(rendered_frames, axis=1)
            grid_path = os.path.join(args.output_dir, f"clip{ci:02d}_grid.png")
            cv2.imwrite(grid_path, grid)
            print(f"  Saved grid: {grid_path}")

    print(f"\nDone. Output in {args.output_dir}/")
    print(f"  - Each frame shows bounding boxes (blue=Left, red=Right)")
    if args.show_mesh:
        print(f"  - MANO mesh vertices are overlaid as dots")
    print(f"  - Box labels show normalised width x height")
    print(f"  - Cross markers show box center")


if __name__ == "__main__":
    main()
