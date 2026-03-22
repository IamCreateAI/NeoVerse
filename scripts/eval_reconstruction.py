"""
Evaluate a reconstructed video against the original on standard image quality metrics.

Metrics computed per frame, then aggregated (mean ± std):
  MSE   - Mean Squared Error (lower is better)
  PSNR  - Peak Signal-to-Noise Ratio in dB (higher is better)
  SSIM  - Structural Similarity Index (higher is better, max 1.0)
  LPIPS - Learned Perceptual Image Patch Similarity (lower is better)

Both videos are resized to the resolution of the original (or --height/--width if given).
Frame counts are matched by taking the minimum of the two, unless --num_frames is set.

Usage:
  python scripts/eval_reconstruction.py --original examples/videos/driving.mp4 \\
      --reconstruction outputs/reconstruction/render.mp4
  python scripts/eval_reconstruction.py --original examples/videos/driving.mp4 \\
      --reconstruction outputs/reconstruction/render.mp4 --output_csv results.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
from torchvision.transforms import functional as F
from tqdm import tqdm

from diffsynth.utils.auxiliary import load_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True,
                   help="Path to the original video or image directory")
    p.add_argument("--reconstruction", required=True,
                   help="Path to the reconstructed video or image directory")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Number of frames to evaluate (default: min of both video lengths)")
    p.add_argument("--height", type=int, required=True,
                   help="Resize frames to this height before evaluation (default: original video height)")
    p.add_argument("--width", type=int, required=True,
                   help="Resize frames to this width before evaluation (default: original video width)")
    p.add_argument("--output_csv", default=None,
                   help="If set, save per-frame metric values to this CSV file")
    p.add_argument("--lpips_net", choices=["alex", "vgg"], default="alex",
                   help="Backbone for LPIPS (default: alex)")
    p.add_argument("--no_gpu", action="store_true",
                   help="Force CPU even if CUDA is available")
    return p.parse_args()


def load_frames(path, num_frames, resolution):
    """Load num_frames from path, returned as a list of [H, W, 3] uint8 numpy arrays."""
    pil_images = load_video(
        path,
        num_frames=num_frames,
        resolution=resolution,
        resize_mode="resize",
        sampling="first",
    )
    return [np.array(img) for img in pil_images]


def video_length(path):
    """Return the number of frames in a video file or image directory."""
    if os.path.isdir(path):
        return len(os.listdir(path))
    from decord import VideoReader
    return len(VideoReader(path))


def to_tensor(frame_np, device):
    """Convert [H, W, 3] uint8 numpy array to [1, 3, H, W] float32 tensor in [-1, 1] (for LPIPS)."""
    t = F.to_tensor(Image.fromarray(frame_np)).unsqueeze(0).to(device)  # [1, 3, H, W], [0, 1]
    return t * 2.0 - 1.0  # LPIPS expects [-1, 1]


def compute_mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def compute_psnr(mse):
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0 ** 2 / mse))


def compute_ssim(a, b):
    # skimage expects HxWxC uint8; data_range=255
    return float(ssim_fn(a, b, data_range=255, channel_axis=2))


def main():
    args = parse_args()
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

    # Resolve frame count
    n_orig = video_length(args.original)
    n_recon = video_length(args.reconstruction)
    num_frames = args.num_frames or min(n_orig, n_recon)
    if num_frames > min(n_orig, n_recon):
        print(f"Warning: requested {num_frames} frames but shortest video has "
              f"{min(n_orig, n_recon)} frames. Clamping.")
        num_frames = min(n_orig, n_recon)

    height, width = args.height, args.width
    resolution = (width, height)

    print(f"Evaluating {num_frames} frames at {width}x{height} ...")
    print(f"  Original      : {args.original}")
    print(f"  Reconstruction: {args.reconstruction}")

    orig_frames = load_frames(args.original, num_frames, resolution)
    recon_frames = load_frames(args.reconstruction, num_frames, resolution)

    # LPIPS model
    lpips_model = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_model.eval()

    rows = []
    mses, psnrs, ssims, lpipss = [], [], [], []

    for i, (orig, recon) in enumerate(tqdm(zip(orig_frames, recon_frames), total=num_frames, desc="Frames")):
        mse = compute_mse(orig, recon)
        psnr = compute_psnr(mse)
        s = compute_ssim(orig, recon)

        with torch.no_grad():
            lp = float(lpips_model(to_tensor(orig, device), to_tensor(recon, device)).item())

        mses.append(mse)
        psnrs.append(psnr)
        ssims.append(s)
        lpipss.append(lp)
        rows.append({"frame": i, "mse": mse, "psnr": psnr, "ssim": s, "lpips": lp})

    # Summary
    def stats(vals):
        return np.mean(vals), np.std(vals)

    print("\n── Results ─────────────────────────────────────")
    for name, vals, better in [
        ("MSE  ", mses, "↓"),
        ("PSNR ", psnrs, "↑"),
        ("SSIM ", ssims, "↑"),
        ("LPIPS", lpipss, "↓"),
    ]:
        mu, sd = stats(vals)
        print(f"  {name} ({better})  mean={mu:.4f}  std={sd:.4f}")
    print("─────────────────────────────────────────────────")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "mse", "psnr", "ssim", "lpips"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-frame results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
