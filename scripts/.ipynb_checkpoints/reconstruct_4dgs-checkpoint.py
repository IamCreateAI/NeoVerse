"""
Standalone 4DGS reconstruction script for NeoVerse / FF-4DGS-Ego.

Runs only the reconstructor (WorldMirror backbone + heads) without loading
the WAN diffusion model, VAE, or text encoder.

Outputs (saved to --output_dir):
  gaussians.pt          - list of per-group dicts (means, harmonics, scales, rotations, opacities, timestamp)
  camera_params.json    - per-frame cam2world [S,4,4] and intrinsics [S,3,3]
  gaussians_frame0000.ply  - Gaussian splat PLY (frame 0 by default)
  render.mp4            - re-rendered input views (with --render_video, for sanity checking)

Usage:
  python scripts/reconstruct_4dgs.py --input_path examples/videos/robot.mp4
  python scripts/reconstruct_4dgs.py --input_path examples/videos/robot.mp4 --render_video
  python scripts/reconstruct_4dgs.py --input_path examples/videos/robot.mp4 --static_scene
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from diffsynth.models import ModelManager
from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse
from diffsynth.auxiliary_models.worldmirror.utils.save_utils import (
    save_gs_ply,
    save_camera_params,
)
from diffsynth.data import save_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True,
                   help="Path to input video (.mp4, etc.) or image directory")
    p.add_argument("--reconstructor_path", default="models/NeoVerse/reconstructor.ckpt",
                   help="Path to reconstructor checkpoint")
    p.add_argument("--output_dir", default="outputs/reconstruction",
                   help="Directory where outputs are saved")
    p.add_argument("--num_frames", type=int, default=120,
                   help="Number of frames to sample from the video")
    p.add_argument("--sampling", choices=["uniform", "first"], default="uniform",
                   help="Frame sampling strategy: 'uniform' spreads frames across the video, "
                        "'first' takes the first num_frames frames (default: uniform)")
    p.add_argument("--frame_offset", type=int, default=0,
                   help="Skip this many frames from the start before applying the sampling strategy (default: 0)")
    p.add_argument("--height", type=int, default=336)
    p.add_argument("--width", type=int, default=336)
    p.add_argument("--resize_mode", choices=["center_crop", "resize"], default="center_crop",
                   help="How to fit frames to the target resolution (default: center_crop)")
    p.add_argument("--static_scene", action="store_true",
                   help="All frames share the same timestamp (static camera / single image)")
    p.add_argument("--save_all_frames_ply", action="store_true",
                   help="Save one PLY per frame instead of only frame 0")
    p.add_argument("--render_video", action="store_true",
                   help="Re-render the scene from input camera poses and save as render.mp4")
    p.add_argument("--fps", type=int, default=16,
                   help="FPS for the rendered output video (default: 16)")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # 1. Load only the reconstructor
    # ------------------------------------------------------------------
    print(f"Loading reconstructor from {args.reconstructor_path} ...")
    model_manager = ModelManager()
    model_manager.load_model(args.reconstructor_path, device=device, torch_dtype=torch.bfloat16)
    reconstructor = model_manager.fetch_model("reconstructor")
    reconstructor.eval()
    print("Reconstructor loaded.")
    # --- CARICA I TUOI PESI ADDESTRATI ---
    hand_weights_path = "hand_head_weights.pt"
    if os.path.exists(hand_weights_path):
        print(f"🔥 Caricamento pesi custom della Hand Head da {hand_weights_path}...")
        custom_hand_weights = torch.load(hand_weights_path, map_location=device)
        # Carichiamo i pesi dentro la testa del modello
        reconstructor.load_state_dict(custom_hand_weights, strict=False)
    else:
        print("⚠️ Attenzione: hand_head_weights.pt non trovato. Uso pesi originali.")
    # -------------------------------------

    # ------------------------------------------------------------------
    # 2. Load and preprocess video frames
    # ------------------------------------------------------------------
    resolution = (args.width, args.height)
    pil_images = load_video(
        args.input_path,
        num_frames=args.num_frames,
        resolution=resolution,
        resize_mode=args.resize_mode,
        static_scene=args.static_scene,
        sampling=args.sampling,
        frame_offset=args.frame_offset,
    )
    S = len(pil_images)
    print(f"Loaded {S} frames at {args.width}x{args.height}.")

    img_tensor = torch.stack(
        [F.to_tensor(img)[None] for img in pil_images], dim=1
    ).to(device)  # [1, S, 3, H, W]

    views = {
        "img": img_tensor,
        "is_target": torch.zeros((1, S), dtype=torch.bool, device=device),
        "is_static": torch.ones((1, S), dtype=torch.bool, device=device)
                     if args.static_scene
                     else torch.zeros((1, S), dtype=torch.bool, device=device),
        "timestamp": torch.zeros((1, S), dtype=torch.int64, device=device)
                     if args.static_scene
                     else torch.arange(0, S, dtype=torch.int64, device=device).unsqueeze(0),
    }

    # ------------------------------------------------------------------
    # 3. Run reconstruction
    # ------------------------------------------------------------------
    print("Running 4DGS reconstruction ...")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        predictions = reconstructor(views, is_inference=True, use_motion=False)

    # predictions["splats"] is List[List[Gaussians]] — batch × frames
    # Each Gaussians has: .means, .harmonics, .opacities, .scales, .rotations, .timestamp
    gaussian_list = predictions["splats"]       # List[List[Gaussians]], len = B
    cam2world  = predictions["rendered_extrinsics"][0]  # [S, 4, 4]
    intrinsics = predictions["rendered_intrinsics"][0]  # [S, 3, 3]

    frame_gaussians = gaussian_list[0]  # List[Gaussians] for batch 0
    print(f"Reconstruction done. Got {len(frame_gaussians)} Gaussian groups "
          f"(one per dynamic frame + optional static group).")

    # ------------------------------------------------------------------
    # 4. Save outputs
    # ------------------------------------------------------------------

    # 4a. Camera parameters as JSON
    cam_path = save_camera_params(
        extrinsics=cam2world.cpu().numpy(),
        intrinsics=intrinsics.cpu().numpy(),
        target_dir=args.output_dir,
    )
    print(f"Saved camera params -> {cam_path}")

    # 4b. Gaussian PLY per group (standard 3DGS format)
    #     Groups with timestamp >= 0 are dynamic (one per source frame).
    #     The group with timestamp == -1 is the fused static background.
    SH_C0 = 0.28209479177387814
    groups_to_save = frame_gaussians if args.save_all_frames_ply else frame_gaussians[:1]
    saved_plys = []
    for gi, gs in enumerate(groups_to_save):
        ts = gs.timestamp
        label = f"static" if ts == -1 else f"frame{ts:04d}"
        ply_path = os.path.join(args.output_dir, f"gaussians_{label}.ply")

        rgbs = (0.5 + SH_C0 * gs.harmonics[..., 0, :]).clamp(0, 1)  # [N, 3]

        save_gs_ply(
            path=ply_path,
            means=gs.means.float(),
            scales=gs.scales.float(),
            rotations=gs.rotations.float(),
            rgbs=rgbs.float(),
            opacities=gs.opacities.float(),
        )
        saved_plys.append(ply_path)
        print(f"Saved Gaussian PLY (ts={ts}) -> {ply_path}")

    # 4c. Save all Gaussians groups as a single .pt for downstream use
    splats_path = os.path.join(args.output_dir, "gaussians.pt")
    torch.save(
        [
            {
                "means": gs.means.cpu(),
                "harmonics": gs.harmonics.cpu(),
                "opacities": gs.opacities.cpu(),
                "scales": gs.scales.cpu(),
                "rotations": gs.rotations.cpu(),
                "timestamp": gs.timestamp,
            }
            for gs in frame_gaussians
        ],
        splats_path,
    )
    print(f"Saved raw splats -> {splats_path}")

    # 4d. Optional: re-render from input cameras for visual sanity check
    if args.render_video:
        print("Rendering input views from reconstructed Gaussians ...")
        timestamps = predictions["rendered_timestamps"][0]   # [S]
        world2cam = homo_matrix_inverse(cam2world.unsqueeze(0))  # [1, S, 4, 4]
        # --- FIX EMERGENZA PER MATRICI SINGOLARI ---
        # Se le matrici sono nulle o "rotte", usiamo l'identità
        if torch.any(torch.isnan(world2cam)) or world2cam.abs().sum() < 1e-3:
            print("⚠️ Attenzione: Pose della camera invalide rilevate. Uso matrici identità per il render.")
            world2cam = torch.eye(4, device=device).view(1, 1, 4, 4).expand(1, S, 4, 4)
        # -------------------------------------------
        target_rgb, _depth, _alpha = reconstructor.gs_renderer.rasterizer.forward(
            gaussian_list,
            render_viewmats=[world2cam[0]],
            render_Ks=[intrinsics],
            render_timestamps=[timestamps],
            sh_degree=0,
            width=args.width,
            height=args.height,
        )
        # target_rgb: [1, S, H, W, 3] in [0, 1]
        frames = [
            Image.fromarray(
                (target_rgb[0, i].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            )
            for i in range(target_rgb.shape[1])
        ]
        render_path = os.path.join(args.output_dir, "render.mp4")
        save_video(frames, render_path, fps=args.fps)
        print(f"Saved render      -> {render_path}")

    print("\nDone. Output files:")
    print(f"  {splats_path}   (list of dicts, load with torch.load)")
    print(f"  {cam_path}")
    for p in saved_plys:
        print(f"  {p}")


if __name__ == "__main__":
    main()
