"""
Visualization utilities for MANO hand mesh overlay on Hot3D Aria RGB frames.

Provides functions to generate MANO meshes from the 44-dim parameter vectors
used in training, project them onto fisheye camera images, and render overlays
comparing ground truth vs predicted hand poses.
"""

import bisect
import csv
import json
import os

import cv2
import numpy as np
import smplx
import torch
from decord import VideoReader
from projectaria_tools.core.calibration import CameraCalibration, FISHEYE624
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# MANO mesh generation
# ---------------------------------------------------------------------------

class MANOModel:
    """Wrapper around smplx MANO for left/right hand mesh generation."""

    def __init__(self, mano_model_folder):
        self.left = smplx.create(
            os.path.join(mano_model_folder, "MANO_LEFT.pkl"),
            "mano",
            use_pca=True,
            is_rhand=False,
            num_pca_comps=15,
        )
        self.right = smplx.create(
            os.path.join(mano_model_folder, "MANO_RIGHT.pkl"),
            "mano",
            use_pca=True,
            is_rhand=True,
            num_pca_comps=15,
        )
        # Fix left hand shapedirs bug (https://github.com/vchoutas/smplx/issues/48)
        if (
            torch.sum(
                torch.abs(self.left.shapedirs[:, 0, :] - self.right.shapedirs[:, 0, :])
            )
            < 1
        ):
            self.left.shapedirs[:, 0, :] *= -1

    def get_mesh_from_params(self, params_32, is_right):
        """Generate mesh from the flat 32-dim training parameter vector.

        Args:
            params_32: tensor [32] = [pos(3), rot_qwxyz(4), pose(15), betas(10)]
            is_right: bool

        Returns:
            vertices: (778, 3) numpy array in world coordinates
            faces: (F, 3) numpy int array
        """
        params = params_32.detach().cpu().float()
        pos = params[:3].numpy()
        q_wxyz = params[3:7].numpy()
        pose = params[7:22].unsqueeze(0)
        betas = params[22:32].unsqueeze(0)

        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        rotvec = Rotation.from_quat(q_xyzw).as_rotvec().astype(np.float32)

        global_orient = torch.from_numpy(rotvec).unsqueeze(0)
        transl = torch.tensor(pos, dtype=torch.float32).unsqueeze(0)

        layer = self.right if is_right else self.left
        output = layer(
            betas=betas,
            global_orient=global_orient,
            hand_pose=pose,
            transl=transl,
            return_verts=True,
        )

        vertices = output.vertices[0].detach().numpy()
        faces = layer.faces.astype(np.int32)
        return vertices, faces


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_camera_calibration(jsonl_path, camera_label="camera-rgb"):
    """Load camera-rgb calibration from online_calibration.jsonl (first entry)."""
    with open(jsonl_path) as f:
        entry = json.loads(f.readline())
        for cam in entry["CameraCalibrations"]:
            if cam["Label"] == camera_label:
                params = np.array(cam["Projection"]["Params"], dtype=np.float64)
                t_dc = np.array(cam["T_Device_Camera"]["Translation"])
                q_dc = cam["T_Device_Camera"]["UnitQuaternion"]
                T_device_camera = SE3.from_quat_and_translation(
                    q_dc[0], np.array(q_dc[1]), t_dc
                )[0]
                cam_calib = CameraCalibration(
                    camera_label, FISHEYE624, params, T_device_camera,
                    1408, 1408, None, 3.14159, "",
                )
                return T_device_camera, cam_calib
    raise RuntimeError(f"Camera '{camera_label}' not found in calibration")


def load_headset_trajectory(csv_path):
    """Load ground truth headset trajectory (timecode domain).
    Returns dict: timecode_ns -> (translation, quaternion_wxyz).
    """
    poses = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ns = int(row["timestamp[ns]"])
            t = np.array([
                float(row["t_wo_x[m]"]),
                float(row["t_wo_y[m]"]),
                float(row["t_wo_z[m]"]),
            ])
            q_wxyz = np.array([
                float(row["q_wo_w"]),
                float(row["q_wo_x"]),
                float(row["q_wo_y"]),
                float(row["q_wo_z"]),
            ])
            poses[ts_ns] = (t, q_wxyz)
    return poses


def load_jsonl_timestamps(jsonl_path, max_entries=None):
    """Load just the timestamp_ns from each JSONL line."""
    timestamps = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if max_entries is not None and i >= max_entries:
                break
            entry = json.loads(line)
            timestamps.append(entry["timestamp_ns"])
    return timestamps


def find_closest(sorted_keys, query):
    """Find the closest key in a sorted list."""
    idx = bisect.bisect_left(sorted_keys, query)
    if idx == 0:
        return sorted_keys[0]
    if idx >= len(sorted_keys):
        return sorted_keys[-1]
    before = sorted_keys[idx - 1]
    after = sorted_keys[idx]
    return before if (query - before) <= (after - query) else after


# ---------------------------------------------------------------------------
# Projection and rendering
# ---------------------------------------------------------------------------

def project_vertices(vertices_world, T_world_device, T_device_camera, cam_calib,
                     image_width=1408):
    """Project 3D world vertices to 2D pixel coordinates.

    Applies 90 deg CW rotation to match MP4 video orientation.
    """
    T_camera_world = T_device_camera.inverse().to_matrix() @ T_world_device.inverse().to_matrix()

    N = vertices_world.shape[0]
    verts_homo = np.hstack([vertices_world, np.ones((N, 1))])
    verts_cam = (T_camera_world @ verts_homo.T).T[:, :3]

    depths = verts_cam[:, 2]
    pixels = np.zeros((N, 2))
    valid = np.zeros(N, dtype=bool)
    margin = 100

    for i in np.where(depths > 0.01)[0]:
        p = cam_calib.project(verts_cam[i])
        if p is not None:
            u = (image_width - 1) - p[1]
            v = p[0]
            if -margin <= u <= image_width + margin and -margin <= v <= image_width + margin:
                pixels[i] = [u, v]
                valid[i] = True

    return pixels, depths, valid


def render_mesh_overlay(image, pixels, faces, depths, valid, color, alpha, wireframe):
    """Render a mesh overlay on the image using filled triangles with alpha blending."""
    overlay = image.copy()

    face_depths = []
    valid_faces = []
    for i, face in enumerate(faces):
        if valid[face[0]] and valid[face[1]] and valid[face[2]]:
            face_depths.append(
                (depths[face[0]] + depths[face[1]] + depths[face[2]]) / 3.0
            )
            valid_faces.append(i)

    if not valid_faces:
        return image

    sorted_indices = np.argsort([-d for d in face_depths])

    for idx in sorted_indices:
        face = faces[valid_faces[idx]]
        pts = pixels[face].astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)

    result = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    if wireframe:
        edge_color = tuple(max(0, int(c * 0.6)) for c in color)
        for idx in sorted_indices:
            face = faces[valid_faces[idx]]
            pts = pixels[face].astype(np.int32)
            for j in range(3):
                cv2.line(
                    result, tuple(pts[j]), tuple(pts[(j + 1) % 3]),
                    edge_color, 1, cv2.LINE_AA,
                )

    return result


# ---------------------------------------------------------------------------
# High-level API for training integration
# ---------------------------------------------------------------------------

GT_LEFT_COLOR = (255, 150, 50)    # blue-ish (BGR)
GT_RIGHT_COLOR = (50, 50, 255)    # red-ish (BGR)
PRED_LEFT_COLOR = (50, 255, 50)   # green (BGR)
PRED_RIGHT_COLOR = (50, 165, 255) # orange (BGR)


def setup_vis_context(seq_path, mano_model_folder=None, mano_model=None):
    """One-time setup: load camera calibration, headset trajectory.

    Provide either mano_model_folder (creates new model) or mano_model (reuses existing).
    Returns a context dict, or None if required files are missing.
    """
    video_path = os.path.join(seq_path, "video_main_rgb.mp4")
    jsonl_path = os.path.join(seq_path, "hand_data", "mano_hand_pose_trajectory.jsonl")
    calib_path = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
    headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")

    for p in [video_path, jsonl_path, calib_path, headset_path]:
        if not os.path.exists(p):
            print(f"[VIS] Skipping visualization: missing {p}")
            return None

    if mano_model is None:
        mano_model = MANOModel(mano_model_folder)
    T_device_camera, cam_calib = load_camera_calibration(calib_path)
    headset_poses = load_headset_trajectory(headset_path)
    headset_ts_sorted = sorted(headset_poses.keys())

    n_video = len(VideoReader(video_path))
    jsonl_timestamps = load_jsonl_timestamps(jsonl_path, max_entries=n_video)

    return {
        "mano_model": mano_model,
        "T_device_camera": T_device_camera,
        "cam_calib": cam_calib,
        "headset_poses": headset_poses,
        "headset_ts_sorted": headset_ts_sorted,
        "jsonl_timestamps": jsonl_timestamps,
        "video_path": video_path,
    }


def render_hand_comparison(vis_context, frame_idx, gt_params_44, pred_params_44):
    """Render GT and predicted MANO hands overlaid on a full-resolution frame.

    Args:
        vis_context: dict from setup_vis_context
        frame_idx: video frame index (matches JSONL line index)
        gt_params_44: tensor [64] ground truth (2 hands x 32)
        pred_params_44: tensor [64] predicted (2 hands x 32)

    Returns:
        RGB numpy image (H, W, 3) uint8 for wandb.Image, or None on failure.
    """
    try:
        vr = VideoReader(vis_context["video_path"])
        frame_rgb = vr[frame_idx].asnumpy()
        image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"[VIS] Failed to read frame {frame_idx}: {e}")
        return None

    # Get device pose for this frame via headset trajectory
    tc_ns = vis_context["jsonl_timestamps"][frame_idx]
    closest_ts = find_closest(vis_context["headset_ts_sorted"], tc_ns)
    t_wd, q_wd_wxyz = vis_context["headset_poses"][closest_ts]
    T_world_device = SE3.from_quat_and_translation(
        q_wd_wxyz[0], q_wd_wxyz[1:], t_wd
    )[0]

    mano = vis_context["mano_model"]
    T_dev_cam = vis_context["T_device_camera"]
    cam_calib = vis_context["cam_calib"]

    # Render GT hands (solid fill)
    for is_right, color in [(False, GT_LEFT_COLOR), (True, GT_RIGHT_COLOR)]:
        offset = 32 if is_right else 0
        params = gt_params_44[offset:offset + 32]
        if params.abs().sum() < 1e-6:
            continue
        try:
            verts, faces = mano.get_mesh_from_params(params, is_right)
            pixels, depths, valid = project_vertices(verts, T_world_device, T_dev_cam, cam_calib)
            if valid.sum() >= 10:
                image = render_mesh_overlay(image, pixels, faces, depths, valid, color, 0.35, False)
        except Exception as e:
            print(f"[VIS] GT {'right' if is_right else 'left'} failed: {e}")

    # Render predicted hands (wireframe)
    for is_right, color in [(False, PRED_LEFT_COLOR), (True, PRED_RIGHT_COLOR)]:
        offset = 32 if is_right else 0
        params = pred_params_44[offset:offset + 32]
        if params.abs().sum() < 1e-6:
            continue
        try:
            verts, faces = mano.get_mesh_from_params(params, is_right)
            pixels, depths, valid = project_vertices(verts, T_world_device, T_dev_cam, cam_calib)
            if valid.sum() >= 10:
                image = render_mesh_overlay(image, pixels, faces, depths, valid, color, 0.35, True)
        except Exception as e:
            print(f"[VIS] Pred {'right' if is_right else 'left'} failed: {e}")

    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "GT Left", (10, 30), font, 0.7, GT_LEFT_COLOR, 2)
    cv2.putText(image, "GT Right", (10, 60), font, 0.7, GT_RIGHT_COLOR, 2)
    cv2.putText(image, "Pred Left", (10, 90), font, 0.7, PRED_LEFT_COLOR, 2)
    cv2.putText(image, "Pred Right", (10, 120), font, 0.7, PRED_RIGHT_COLOR, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
