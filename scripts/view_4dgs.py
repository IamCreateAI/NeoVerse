"""
Interactive 4DGS viewer for NeoVerse / FF-4DGS-Ego reconstruction outputs.

Uses viser's native Gaussian splat rendering — actual 3D ellipsoids in the browser.

Controls (in browser):
  - Left-drag   : orbit
  - Right-drag  : pan
  - Scroll      : zoom
  - Frame slider: scrub through time

Requires only the files produced by reconstruct_4dgs.py:
  gaussians.pt        - saved Gaussians groups
  camera_params.json  - per-frame intrinsics & cam2world poses

Usage:
  python scripts/view_4dgs.py --output_dir outputs/my_reconstruction
  python scripts/view_4dgs.py --output_dir outputs/my_reconstruction --host 0.0.0.0 --port 8080
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf
from scipy.spatial.transform import Rotation

from diffsynth.auxiliary_models.worldmirror.models.models.rasterization import Gaussians


SH_C0 = 0.28209479177387814


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gaussians(pt_path: str, device: str) -> list:
    data = torch.load(pt_path, map_location=device, weights_only=False)
    return [
        Gaussians(
            means=d["means"].to(device),
            harmonics=d["harmonics"].to(device),
            opacities=d["opacities"].to(device),
            scales=d["scales"].to(device),
            rotations=d["rotations"].to(device),
            timestamp=d["timestamp"],
        )
        for d in data
    ]


def load_cameras(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    c2w = np.array([e["matrix"] for e in data["extrinsics"]], dtype=np.float32)
    K   = np.array([e["matrix"] for e in data["intrinsics"]], dtype=np.float32)
    return c2w, K


# ---------------------------------------------------------------------------
# Gaussian covariance computation
# ---------------------------------------------------------------------------

def splats_to_viser(gaussian_list: list, timestamp: int, opacity_thresh: float):
    """
    Collect all Gaussians relevant to `timestamp` (static + matching dynamic)
    and return arrays in the format expected by viser.scene.add_gaussian_splats().
    """
    all_centers    = []
    all_covs       = []
    all_rgbs       = []
    all_opacities  = []

    for gs in gaussian_list:
        if gs.timestamp != -1 and gs.timestamp != timestamp:
            continue

        opacities = gs.opacities.cpu().float().numpy()          # [N]
        mask = opacities > opacity_thresh
        if not mask.any():
            continue

        means      = gs.means.cpu().float().numpy()[mask]       # [N, 3]
        scales     = gs.scales.cpu().float().numpy()[mask]      # [N, 3]
        quats_wxyz = gs.rotations.cpu().float().numpy()[mask]   # [N, 4] wxyz
        harmonics  = gs.harmonics.cpu().float().numpy()[mask]   # [N, deg, 3]
        opacities  = opacities[mask]                            # [N]

        # RGB from DC spherical harmonics coefficient
        rgbs = np.clip(0.5 + SH_C0 * harmonics[:, 0, :3], 0.0, 1.0)  # [N, 3]

        # 3D covariance:  Σ = R S S^T R^T,  S = diag(scales)
        quats_xyzw = quats_wxyz[:, [1, 2, 3, 0]]               # wxyz → xyzw for scipy
        R = Rotation.from_quat(quats_xyzw).as_matrix()          # [N, 3, 3]
        RS = R * scales[:, np.newaxis, :]                       # [N, 3, 3]
        cov = RS @ RS.transpose(0, 2, 1)                        # [N, 3, 3]

        all_centers.append(means)
        all_covs.append(cov)
        all_rgbs.append(rgbs)
        all_opacities.append(opacities)

    if not all_centers:
        return None

    opacities = np.concatenate(all_opacities, axis=0).astype(np.float32)
    return dict(
        centers    = np.concatenate(all_centers,   axis=0).astype(np.float32),
        covariances= np.concatenate(all_covs,      axis=0).astype(np.float32),
        rgbs       = np.concatenate(all_rgbs,      axis=0).astype(np.float32),
        opacities  = opacities.reshape(-1, 1),   # viser requires shape (N, 1)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True,
                   help="Directory produced by reconstruct_4dgs.py")
    p.add_argument("--host",   type=str, default="localhost",
                   help="Host to bind to. Use 0.0.0.0 to expose on all interfaces.")
    p.add_argument("--port",   type=int, default=8080)
    p.add_argument("--opacity_thresh", type=float, default=0.05,
                   help="Minimum opacity to display a Gaussian (filters noise)")
    p.add_argument("--show_frustums", action="store_true",
                   help="Show input camera frustums in the scene")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)

    print("Loading Gaussians ...")
    gaussian_list = load_gaussians(out_dir / "gaussians.pt", device)

    print("Loading camera parameters ...")
    cam2world, intrinsics = load_cameras(out_dir / "camera_params.json")
    S = cam2world.shape[0]

    dynamic_timestamps = sorted(
        set(gs.timestamp for gs in gaussian_list if gs.timestamp >= 0)
    )
    num_frames = len(dynamic_timestamps) if dynamic_timestamps else 1
    print(f"  {num_frames} dynamic frame(s) + "
          f"{sum(1 for gs in gaussian_list if gs.timestamp == -1)} static group(s)")

    # ------------------------------------------------------------------
    # Viser server
    # ------------------------------------------------------------------
    server = viser.ViserServer(host=args.host, port=args.port)
    host_display = args.host if args.host != "0.0.0.0" else "<remote-ip>"
    print(f"\nOpen your browser at  http://{host_display}:{args.port}\n")

    # GUI controls
    time_slider = server.gui.add_slider(
        "Frame", min=0, max=num_frames - 1, step=1, initial_value=0
    )
    opacity_slider = server.gui.add_slider(
        "Opacity threshold", min=0.0, max=0.5, step=0.01,
        initial_value=args.opacity_thresh,
    )

    # Optional camera frustums
    if args.show_frustums:
        for i in range(S):
            R = cam2world[i, :3, :3]
            t = cam2world[i, :3, 3]
            wxyz = vtf.SO3.from_matrix(R).wxyz
            fy = intrinsics[i, 1, 1]
            H = intrinsics[i, 1, 2] * 2   # approximate from principal point
            server.scene.add_camera_frustum(
                f"/cameras/frame_{i:04d}",
                fov=float(2 * np.arctan2(H / 2, fy)),
                aspect=float(intrinsics[i, 0, 0] / fy),
                scale=0.05,
                wxyz=wxyz,
                position=t,
                color=(180, 180, 255),
            )

    # ------------------------------------------------------------------
    # State & render loop
    # ------------------------------------------------------------------
    state = {"dirty": True, "splat_handle": None}

    @time_slider.on_update
    def _(_): state["dirty"] = True

    @opacity_slider.on_update
    def _(_): state["dirty"] = True

    @server.on_client_connect
    def _(client):
        state["dirty"] = True   # send scene to new client

    print("Viewer running. Press Ctrl-C to quit.")
    try:
        while True:
            if state["dirty"]:
                state["dirty"] = False
                frame_idx = int(time_slider.value)
                timestamp = dynamic_timestamps[frame_idx] if dynamic_timestamps else 0
                thresh    = float(opacity_slider.value)

                splat_data = splats_to_viser(gaussian_list, timestamp, thresh)
                if splat_data is not None:
                    n = len(splat_data["centers"])
                    print(f"  Frame {frame_idx} (ts={timestamp}): "
                          f"rendering {n:,} Gaussians", end="\r")
                    server.scene.add_gaussian_splats(
                        "/splats",
                        centers     = splat_data["centers"],
                        covariances = splat_data["covariances"],
                        rgbs        = splat_data["rgbs"],
                        opacities   = splat_data["opacities"],
                    )

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nShutting down viewer.")


if __name__ == "__main__":
    main()
