"""
Visualize hand bounding boxes on video frames.

Uses the exact same dataset, data loading, and bbox computation as training
to ensure what you see matches what the model receives during training.

Usage:
    python -m scripts.visualize_hand_bboxes --config configs/hamer_hand_head.yaml
    python -m scripts.visualize_hand_bboxes --config configs/hamer_hand_head.yaml --num_clips 5 --frames_per_clip 4
    python -m scripts.visualize_hand_bboxes --config configs/hamer_hand_head.yaml --save_video
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
import yaml

from scripts.train_hand_head import (
    HOT3DHandDataset,
    discover_sequences,
)

LEFT_COLOR = (255, 150, 50)   # blue-ish (BGR)
RIGHT_COLOR = (50, 50, 255)   # red-ish (BGR)


def draw_bboxes_on_frame(img_tensor, bboxes, valid):
    """Draw hand bounding boxes on a single frame.

    Args:
        img_tensor: [3, H, W] float tensor in [0, 1]
        bboxes: [2, 4] normalised bboxes (x1, y1, x2, y2)
        valid: [2] boolean mask

    Returns:
        BGR uint8 numpy image with bboxes drawn.
    """
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    colors = [LEFT_COLOR, RIGHT_COLOR]
    labels = ["Left", "Right"]

    for hand_idx in range(2):
        if not valid[hand_idx]:
            continue
        x1, y1, x2, y2 = bboxes[hand_idx].numpy()
        pt1 = (int(x1 * W), int(y1 * H))
        pt2 = (int(x2 * W), int(y2 * H))
        cv2.rectangle(img, pt1, pt2, colors[hand_idx], 2)
        cv2.putText(
            img, labels[hand_idx],
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[hand_idx], 1, cv2.LINE_AA,
        )

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize hand bounding boxes on training frames")
    parser.add_argument("--config", required=True, help="Training config YAML (same as used for training)")
    parser.add_argument("--num_clips", type=int, default=3, help="Number of clips to visualize")
    parser.add_argument("--frames_per_clip", type=int, default=8, help="Max frames to show per clip")
    parser.add_argument("--output_dir", default="outputs/vis_hand_bboxes", help="Output directory for images")
    parser.add_argument("--save_video", action="store_true", help="Save each clip as a video instead of individual frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clip selection")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg.get("model", {})
    crop_cfg = cfg.get("hand_crop", {})

    data_root = data_cfg["data_root"]
    num_frames = data_cfg["num_frames"]
    res = tuple(data_cfg["resolution"])
    clip_stride = data_cfg.get("clip_stride", num_frames)

    bbox_margin = crop_cfg.get("bbox_margin", 0.15)

    # Build dataset with hand crop enabled (same as training)
    all_seqs = discover_sequences(data_root)
    if not all_seqs:
        print(f"No sequences found in {data_root}")
        return

    dataset = HOT3DHandDataset(
        all_seqs,
        num_frames=num_frames,
        res=res,
        clip_stride=clip_stride,
        use_hand_crop=True,
        bbox_margin=bbox_margin,
    )
    print(f"Dataset: {len(dataset)} clips from {len(all_seqs)} sequences")

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
        frame_offset = clip_info["frame_offset"]
        n_show = min(args.frames_per_clip, imgs.shape[0])

        print(f"Clip {ci+1}/{num_clips}: seq={seq_name}, offset={frame_offset}, "
              f"frames={imgs.shape[0]}, showing={n_show}")

        rendered_frames = []
        for fi in range(n_show):
            frame = draw_bboxes_on_frame(imgs[fi], bboxes[fi], valid[fi])

            # Add frame info text
            info = f"frame {frame_offset + fi}"
            cv2.putText(frame, info, (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

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

    print(f"Done. Output in {args.output_dir}/")


if __name__ == "__main__":
    main()
