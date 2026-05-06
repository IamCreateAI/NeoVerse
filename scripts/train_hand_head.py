"""
Train MANO hand parameter prediction head on Hot3D Aria data.

Example:
    python -m scripts.train_hand_head --config configs/train_hand_head.yaml
"""

import argparse
import bisect
import json
import os
import random

import numpy as np

import torch
import torch.nn.functional as F
import numpy as np
import wandb
import yaml
from decord import VideoReader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TVF
from tqdm import tqdm

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.utils.auxiliary import load_video

from scripts.hamer_losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from scripts.hand_gs_consistency import HandGSConsistencyLoss, reset_diag_flag


HAND_PARAM_DIM = 32  # per hand: pos(3) + rot(4) + pose(15) + betas(10)
NUM_HANDS = 2


# ------------------------------------------------------------------
# Bounding-box utilities
# ------------------------------------------------------------------



def default_full_image_bboxes(num_frames: int) -> tuple:
    """Fallback: return full-image bboxes (no effective cropping).

    Useful as a baseline or when real bboxes are not yet available.
    """
    bboxes = torch.zeros(num_frames, NUM_HANDS, 4)
    bboxes[:, :, 2] = 1.0  # x2
    bboxes[:, :, 3] = 1.0  # y2
    valid = torch.ones(num_frames, NUM_HANDS, dtype=torch.bool)
    return bboxes, valid


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class HOT3DHandDataset(Dataset):
    """Sliding-window clips over a list of sequences."""

    def __init__(self, seq_dirs, mano_model, num_frames=16, res=(224, 224), clip_stride=None,
                 use_hand_crop=False, rescale_factor=2.0):
        self.num_frames = num_frames
        self.mano_model = mano_model
        self.res = res
        self.use_hand_crop = use_hand_crop
        self.rescale_factor = rescale_factor
        self.clips = []
        
        if clip_stride is None:
            clip_stride = num_frames

        for seq_path in tqdm(seq_dirs):
            video_path = os.path.join(seq_path, "video_main_rgb.mp4")
            hand_data_root = os.path.join(seq_path, "hand_data")
            jsonl_path = os.path.join(hand_data_root, "mano_hand_pose_trajectory.jsonl")
            # The joint cache depends on whether we've rewritten the params into
            # camera frame. Bump the filename so stale world-frame caches don't
            # get silently reused after switching crop mode.
            # v2 suffix = MANO transl semantics fixed (joint_0_canonical-aware).
            joint_cache_name = (
                "gt_joints_cache_cam_v2.pt" if self.use_hand_crop
                else "gt_joints_cache_world.pt"
            )
            joint_cache_path = os.path.join(hand_data_root, joint_cache_name)

            if not os.path.exists(video_path) or not os.path.exists(jsonl_path):
                print(f"Skipping {seq_path} because it doesn't have a video or jsonl file")
                continue
            
            # if seq_path[0:5] in ["P0004", "P0005", "P0006", "P0008", "P0016", "P0020"]:
            #     print(f"Skipping {seq_path} as it doesn't have MANO data")
            #     continue

            # Load all JSONL entries keyed by timestamp
            hand_entries = {}  # timestamp_ns -> hand_poses dict
            with open(jsonl_path) as f:
                for line in f:
                    entry = json.loads(line)
                    hand_entries[entry["timestamp_ns"]] = entry["hand_poses"]
            hand_ts_sorted = sorted(hand_entries.keys())

            if len(hand_ts_sorted) < 2:
                continue

            n_video = len(VideoReader(video_path))
            if n_video < num_frames:
                continue

            # Map each video frame to the closest JSONL entry via linear interpolation
            ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]

            def _closest_ts(frame_i):
                frac = frame_i / max(n_video - 1, 1)
                query = int(ts_start + frac * (ts_end - ts_start))
                idx = bisect.bisect_left(hand_ts_sorted, query)
                if idx == 0:
                    return hand_ts_sorted[0]
                if idx >= len(hand_ts_sorted):
                    return hand_ts_sorted[-1]
                before, after = hand_ts_sorted[idx - 1], hand_ts_sorted[idx]
                return before if (query - before) <= (after - query) else after

            def _hand_to_vec(hand_poses):
                vecs = []
                for hand_id in ["0", "1"]:
                    hand = hand_poses.get(hand_id, {})
                    if hand:
                        vecs.append(torch.cat([
                            torch.tensor(hand["wrist_xform"]["t_xyz"],  dtype=torch.float32),
                            torch.tensor(hand["wrist_xform"]["q_wxyz"], dtype=torch.float32),
                            torch.tensor(hand["pose"],                  dtype=torch.float32),
                            torch.tensor(hand["betas"],                 dtype=torch.float32),
                        ]))
                    else:
                        vecs.append(torch.zeros(HAND_PARAM_DIM))
                return torch.cat(vecs)

            gt_per_frame = []
            for frame_i in range(n_video):
                ts = _closest_ts(frame_i)
                gt_per_frame.append(_hand_to_vec(hand_entries[ts]))

            # Handle 2D GT joints + camera data. If we have to compute this
            # from scratch, we need world-frame joints (project_vertices applies
            # its own world→camera transform). If the cache exists we skip that
            # entirely, which matters because after use_hand_crop below we
            # rewrite gt_per_frame into camera frame.
            cam_2d_cache_path   = os.path.join(hand_data_root, "gt_joints_2d_cache.pt")
            cam_extr_cache_path = os.path.join(hand_data_root, "cam_extrinsics_cache.pt")
            cam_intr_cache_path = os.path.join(hand_data_root, "cam_intrinsics.pt")

            if (os.path.exists(cam_2d_cache_path)
                    and os.path.exists(cam_extr_cache_path)
                    and os.path.exists(cam_intr_cache_path)):
                seq_gt_joints_2d   = torch.load(cam_2d_cache_path,   weights_only=True)
                seq_cam_extrinsics = torch.load(cam_extr_cache_path,  weights_only=True)
                seq_cam_intrinsics = torch.load(cam_intr_cache_path,  weights_only=True)
                print(f"Loaded 2D GT joints + cam data for {seq_path}.")
            else:
                # gt_per_frame is still in world frame here (crop transform runs below),
                # so joints computed from it are in world frame as project_vertices expects.
                seq_gt_joints_world = self._compute_seq_joints_from_params(gt_per_frame)
                seq_gt_joints_2d, seq_cam_extrinsics, seq_cam_intrinsics = (
                    HOT3DHandDataset._compute_2d_cam_data(
                        seq_path, n_video, hand_ts_sorted, seq_gt_joints_world
                    )
                )
                if seq_gt_joints_2d is not None:
                    torch.save(seq_gt_joints_2d,   cam_2d_cache_path)
                    torch.save(seq_cam_extrinsics, cam_extr_cache_path)
                    torch.save(seq_cam_intrinsics, cam_intr_cache_path)
                    print(f"Computed and saved 2D GT joints + cam data for {seq_path}.")
                else:
                    print(f"No calibration for {seq_path} — 2D loss unavailable for this sequence.")

            # Handle Bounding Boxes (rewrites gt_per_frame into camera frame on a hit).
            if self.use_hand_crop:
                cache_name = f"hand_bboxes_v2_rf{self.rescale_factor}_res{res[0]}x{res[1]}.pt"
                bbox_cache_path = os.path.join(seq_path, "hand_data", cache_name)

                if os.path.exists(bbox_cache_path):
                    cached = torch.load(bbox_cache_path, weights_only=True)
                    bbox_frames = list(cached["bboxes"])
                    valid_frames = list(cached["valid"])
                    gt_per_frame[:] = list(cached["gt"])
                else:
                    bbox_frames, valid_frames = HOT3DHandDataset._compute_projected_bboxes(
                        seq_path, n_video, hand_ts_sorted, gt_per_frame,
                        rescale_factor=self.rescale_factor,
                    )
                    if bbox_frames is None:
                        print(f"Skipping {seq_path}: missing calibration for hand crop")
                        continue

                    # world → camera frame so predicted and ground-truth params
                    # (and the joints we derive from them) live in the same frame.
                    ok = self._transform_gt_to_crop_local(
                        seq_path, n_video, hand_ts_sorted, gt_per_frame,
                        bbox_frames, valid_frames, res=res,
                    )
                    if not ok:
                        print(f"Skipping {seq_path}: missing calibration for GT crop-local transform")
                        continue

                    torch.save({
                        "bboxes": torch.stack(bbox_frames),
                        "valid": torch.stack(valid_frames),
                        "gt": torch.stack(gt_per_frame),
                    }, bbox_cache_path)
                    print(f"Cached hand bboxes -> {bbox_cache_path}")
            else:
                bbox_frames = valid_frames = None

            # Build GT joints from the (possibly camera-frame) params.
            if os.path.exists(joint_cache_path):
                seq_gt_joints = torch.load(joint_cache_path, weights_only=True)
                print(f"Loaded GT joints for {seq_path}.")
            else:
                seq_gt_joints = self._compute_seq_joints_from_params(gt_per_frame)
                torch.save(seq_gt_joints, joint_cache_path)
                print(f"Computed and saved GT joints for {seq_path}.")

            # Create Sliding Window Clips
            for start in range(0, n_video - num_frames + 1, clip_stride):
                end = start + num_frames
                clip = {
                    "video_path":   video_path,
                    "gt_frames":    gt_per_frame[start : end],
                    "gt_joints":    seq_gt_joints[start : end],
                    "frame_offset": start,
                    "seq_path":     seq_path,
                }
                if self.use_hand_crop:
                    clip["hand_bboxes"] = bbox_frames[start : start + num_frames]
                    clip["hand_valid"]  = valid_frames[start : start + num_frames]
                if seq_gt_joints_2d is not None:
                    clip["gt_joints_2d"]   = seq_gt_joints_2d[start : end]    # [S, 2, 16, 3]
                    clip["cam_extrinsics"] = seq_cam_extrinsics[start : end]  # [S, 4, 4]
                    clip["cam_intrinsics"] = seq_cam_intrinsics                # [3]
                self.clips.append(clip)

    def _compute_seq_joints_from_params(self, gt_per_frame):
        """Run MANO once per (frame, hand) to produce [N, 2, 16, 3] joints.

        Used for both the world-frame joints fed to the 2D cache generator and
        the camera-frame joints that become the 3D-loss ground truth.
        """
        seq_list = []
        for frame_p in gt_per_frame:
            frame_joints = []
            for h_idx in range(NUM_HANDS):
                offset = h_idx * HAND_PARAM_DIM
                p = frame_p[offset : offset + HAND_PARAM_DIM]
                if p.abs().sum() < 1e-6:
                    j3d = np.zeros((16, 3), dtype=np.float32)
                else:
                    j3d = self.mano_model.get_joints_from_tensor(
                        p, is_right=(h_idx == 1), return_tensor=False,
                    )
                    if isinstance(j3d, np.ndarray) and j3d.ndim == 3:
                        j3d = j3d.squeeze(0)
                    elif torch.is_tensor(j3d) and j3d.dim() == 3:
                        j3d = j3d.squeeze(0)
                frame_joints.append(torch.as_tensor(j3d, dtype=torch.float32))
            seq_list.append(torch.stack(frame_joints))
        return torch.stack(seq_list)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        pil_images = load_video(
            clip["video_path"],
            num_frames=self.num_frames,
            resolution=self.res,
            sampling="first",
            frame_offset=clip["frame_offset"],
        )
        
        imgs = torch.stack([TVF.to_tensor(img) for img in pil_images])
        gt_params = torch.stack(clip["gt_frames"])
        gt_joints = clip["gt_joints"] 

        # # Pre-compute GT joints for the whole clip
        # all_gt_joints = []
        # for frame_p in gt_params:
        #     frame_joints = []
        #     for h_idx in range(NUM_HANDS):
        #         offset = h_idx * HAND_PARAM_DIM
        #         p = frame_p[offset : offset + HAND_PARAM_DIM]
                
        #         # Use CPU/Numpy version for the Dataset
        #         if p.abs().sum() < 1e-6:
        #             j3d = np.zeros((21, 3), dtype=np.float32)
        #         else:
        #             j3d = self.mano_model.get_joints_from_tensor(p, is_right=(h_idx==1), return_tensor=False)
        #         frame_joints.append(torch.from_numpy(j3d))
        #     all_gt_joints.append(torch.stack(frame_joints))
            

        # gt_joints_3d = torch.stack(all_gt_joints) 
        # gt_joints_3d = torch.squeeze(gt_joints_3d, 2)
        # # [16, 2, 21, 3], [num_frames, num_hands, num_joints, dims]
        
        # print(f"gt_joints_3d.shape: {gt_joints_3d.shape}")

        out = {
            "img": imgs, 
            "gt": gt_params, 
            "gt_joints": gt_joints
        }

        if self.use_hand_crop:
            out["hand_bboxes"] = torch.stack(clip["hand_bboxes"])  # [S, 2, 4]
            out["hand_valid"]  = torch.stack(clip["hand_valid"])   # [S, 2]

        if "gt_joints_2d" in clip:
            out["gt_joints_2d"]   = clip["gt_joints_2d"]    # [S, 2, 16, 3]
            out["cam_extrinsics"] = clip["cam_extrinsics"]  # [S, 4, 4]
            out["cam_intrinsics"] = clip["cam_intrinsics"]  # [3]

        return out

    @staticmethod
    def _compute_projected_bboxes(seq_path, n_video, hand_ts_sorted, gt_per_frame,
                                   rescale_factor=2.0, **_kwargs):
        """Compute per-frame hand bboxes by projecting the full MANO mesh.

        Follows the same approach as HaMeR's hand detection pipeline:
        1. Project all hand mesh vertices to 2D (like HaMeR uses ViTPose keypoints)
        2. Compute a tight bounding box around all valid projected vertices
        3. Apply a rescale factor to pad the box (HaMeR default: 2.0x)

        This produces bboxes that tightly enclose the visible hand and are
        centered on the hand (not just the wrist), matching HaMeR's ViTPose
        keypoint-based bbox extraction.

        Returns lists of [2, 4] bbox tensors (normalised x1,y1,x2,y2 in [0,1])
        and [2] bool valid tensors, or (None, None) if calibration files are missing.
        """
        import numpy as np
        from projectaria_tools.core.sophus import SE3
        from scripts.hand_vis_utils import (
            load_camera_calibration, load_headset_trajectory, find_closest,
            load_hand_poses, MANOModel, project_vertices,
        )

        calib_path   = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
        headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")
        jsonl_path   = os.path.join(seq_path, "hand_data", "mano_hand_pose_trajectory.jsonl")
        mano_folder  = os.path.join(os.path.dirname(os.path.dirname(seq_path)),
                                     "models", "MANO")

        # Also accept mano_folder from the repo root (common layout)
        if not os.path.exists(mano_folder):
            # Try relative to the FF-4DGS-Ego repo root
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mano_folder = os.path.join(repo_root, "models", "MANO")

        for p in [calib_path, headset_path, jsonl_path]:
            if not os.path.exists(p):
                return None, None
        if not os.path.exists(mano_folder):
            print(f"[WARN] MANO model folder not found at {mano_folder}, "
                  "cannot compute mesh-based bboxes")
            return None, None

        T_device_camera, cam_calib = load_camera_calibration(calib_path)
        headset_poses = load_headset_trajectory(headset_path)
        headset_ts    = sorted(headset_poses.keys())

        hand_poses_data = load_hand_poses(jsonl_path)
        hand_ts_data    = sorted(hand_poses_data.keys())

        mano_model = MANOModel(mano_folder)

        ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]
        IMAGE_WIDTH = 1408  # Aria sensor resolution before resize

        bboxes_list = []
        valid_list  = []

        for frame_i in range(len(gt_per_frame)):
            frac     = frame_i / max(n_video - 1, 1)
            query_tc = int(ts_start + frac * (ts_end - ts_start))

            # Find closest headset pose
            closest_ht = find_closest(headset_ts, query_tc)
            t_wd, q_wd = headset_poses[closest_ht]
            T_world_device = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0]

            # Find closest hand pose entry (raw JSONL data with full MANO params)
            closest_hand_ts = find_closest(hand_ts_data, query_tc)
            hand_data = hand_poses_data[closest_hand_ts]

            frame_bboxes = torch.zeros(NUM_HANDS, 4)
            frame_valid  = torch.zeros(NUM_HANDS, dtype=torch.bool)

            for hand_idx in range(NUM_HANDS):
                hand_key = str(hand_idx)  # "0" = left, "1" = right
                is_right = hand_idx == 1

                if hand_key not in hand_data or not hand_data[hand_key]:
                    # Hand absent: safe fallback bbox
                    frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                    continue

                try:
                    # Generate full MANO mesh in world coordinates
                    vertices, _faces = mano_model.get_mesh(hand_data[hand_key], is_right)

                    # Project all 778 vertices to 2D using the same projection
                    # as hand_vis_utils (fisheye + 90° rotation)
                    pixels, depths, valid_mask = project_vertices(
                        vertices, T_world_device, T_device_camera, cam_calib,
                        image_width=IMAGE_WIDTH,
                    )

                    if valid_mask.sum() < 10:
                        frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                        continue

                    # Tight bbox around all valid projected vertices (in pixels)
                    valid_pixels = pixels[valid_mask]  # [N_valid, 2] — (u, v) in pixel coords
                    u_min, v_min = valid_pixels.min(axis=0)
                    u_max, v_max = valid_pixels.max(axis=0)

                    # Compute center and size (Hamer-style: center + scale)
                    center_u = (u_min + u_max) / 2.0
                    center_v = (v_min + v_max) / 2.0
                    bbox_w = u_max - u_min
                    bbox_h = v_max - v_min

                    # Apply rescale factor (Hamer default = 2.0) to pad the bbox
                    # This matches HaMeR's ViTDetDataset: scale = rescale_factor * (box_size) / 200
                    # but we directly expand the bbox by the factor
                    bbox_w *= rescale_factor
                    bbox_h *= rescale_factor

                    # Make square (take max side, like Hamer's expand_to_aspect_ratio)
                    bbox_size = max(bbox_w, bbox_h)

                    # Convert to normalised [0,1] coords (x1, y1, x2, y2)
                    x1 = (center_u - bbox_size / 2.0) / IMAGE_WIDTH
                    y1 = (center_v - bbox_size / 2.0) / IMAGE_WIDTH
                    x2 = (center_u + bbox_size / 2.0) / IMAGE_WIDTH
                    y2 = (center_v + bbox_size / 2.0) / IMAGE_WIDTH

                    # Clamp to [0, 1]
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))

                    # Sanity check: box must have positive area
                    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                        frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                        continue

                    frame_bboxes[hand_idx] = torch.tensor([x1, y1, x2, y2])
                    frame_valid[hand_idx] = True

                except Exception as e:
                    # Fallback on MANO mesh generation failure
                    frame_bboxes[hand_idx] = torch.tensor([0.25, 0.25, 0.75, 0.75])
                    continue

            bboxes_list.append(frame_bboxes)
            valid_list.append(frame_valid)

        return bboxes_list, valid_list

    @staticmethod
    def _load_camera_seq_data(seq_path):
        """Load per-sequence camera calibration + headset trajectory.

        Returns (T_device_camera, cam_calib, headset_poses, headset_ts, calib_path)
        or None if the required calibration files are missing.
        """
        from scripts.hand_vis_utils import load_camera_calibration, load_headset_trajectory

        calib_path = os.path.join(seq_path, "mps_slam_calibration", "online_calibration.jsonl")
        headset_path = os.path.join(seq_path, "ground_truth", "headset_trajectory.csv")
        if not os.path.exists(calib_path) or not os.path.exists(headset_path):
            return None

        T_device_camera, cam_calib = load_camera_calibration(calib_path)
        headset_poses = load_headset_trajectory(headset_path)
        headset_ts = sorted(headset_poses.keys())
        return T_device_camera, cam_calib, headset_poses, headset_ts, calib_path

    @staticmethod
    def _frame_camera_transforms(frame_i, n_video, hand_ts_sorted,
                                 headset_poses, headset_ts, T_device_camera):
        """Build T_world_device and T_camera_world (numpy [4, 4]) for one frame."""
        from projectaria_tools.core.sophus import SE3
        from scripts.hand_vis_utils import find_closest

        ts_start, ts_end = hand_ts_sorted[0], hand_ts_sorted[-1]
        frac = frame_i / max(n_video - 1, 1)
        query_tc = int(ts_start + frac * (ts_end - ts_start))

        closest_ht = find_closest(headset_ts, query_tc)
        t_wd, q_wd = headset_poses[closest_ht]
        T_world_device = SE3.from_quat_and_translation(q_wd[0], q_wd[1:], t_wd)[0]
        T_camera_world = (
            T_device_camera.inverse().to_matrix()
            @ T_world_device.inverse().to_matrix()
        )
        return T_world_device, T_camera_world

    @staticmethod
    def _compute_2d_cam_data(seq_path, n_video, hand_ts_sorted, seq_gt_joints_3d):
        """Compute GT 2D keypoints and per-frame camera extrinsics for the 2D loss.

        Projects all GT 3D joints to pixel coordinates using project_vertices and
        records the world-to-camera extrinsic matrix for each frame so that predicted
        joints can be projected differentiably at training time.

        Returns:
            gt_joints_2d   (torch.Tensor | None): [N, 2, 16, 3]  — (u, v, confidence)
            cam_extrinsics (torch.Tensor | None): [N, 4, 4]       — T_camera_world per frame
            cam_intrinsics (torch.Tensor | None): [3]             — [f, cx, cy]
            or (None, None, None) if calibration files are missing.
        """
        import numpy as np
        from scripts.hand_vis_utils import project_vertices

        seq_data = HOT3DHandDataset._load_camera_seq_data(seq_path)
        if seq_data is None:
            return None, None, None
        T_device_camera, cam_calib, headset_poses, headset_ts, calib_path = seq_data

        # FISHEYE624 params layout: [f, cx, cy, k1..k6, p1, p2, s0..s3]
        with open(calib_path) as fh:
            entry = json.loads(fh.readline())
            for cam in entry["CameraCalibrations"]:
                if cam["Label"] == "camera-rgb":
                    raw_params = np.array(cam["Projection"]["Params"], dtype=np.float64)
                    break
        cam_intrinsics = torch.tensor(
            [raw_params[0], raw_params[1], raw_params[2]], dtype=torch.float32
        )  # [f, cx, cy]

        IMAGE_WIDTH = 1408

        gt_joints_2d_list   = []
        cam_extrinsics_list = []

        for frame_i in range(n_video):
            T_world_device, T_cam_world_np = HOT3DHandDataset._frame_camera_transforms(
                frame_i, n_video, hand_ts_sorted, headset_poses, headset_ts, T_device_camera,
            )
            cam_extrinsics_list.append(
                torch.tensor(T_cam_world_np, dtype=torch.float32)
            )

            # Project all joints for this frame
            frame_joints_3d = seq_gt_joints_3d[frame_i]   # [2, 16, 3]
            frame_joints_2d = torch.zeros(NUM_HANDS, 16, 3)  # [u, v, confidence]

            for h_idx in range(NUM_HANDS):
                joints_w = frame_joints_3d[h_idx].numpy()  # [16, 3]
                if np.abs(joints_w).sum() < 1e-6:
                    continue  # hand absent — confidence stays 0

                pixels, _, valid = project_vertices(
                    joints_w, T_world_device, T_device_camera, cam_calib, IMAGE_WIDTH
                )
                frame_joints_2d[h_idx, :, 0] = torch.from_numpy(pixels[:, 0].astype(np.float32))
                frame_joints_2d[h_idx, :, 1] = torch.from_numpy(pixels[:, 1].astype(np.float32))
                frame_joints_2d[h_idx, :, 2] = torch.from_numpy(valid.astype(np.float32))

            gt_joints_2d_list.append(frame_joints_2d)

        gt_joints_2d   = torch.stack(gt_joints_2d_list)    # [N, 2, 16, 3]
        cam_extrinsics = torch.stack(cam_extrinsics_list)  # [N, 4, 4]
        return gt_joints_2d, cam_extrinsics, cam_intrinsics

    def _transform_gt_to_crop_local(self, seq_path, n_video, hand_ts_sorted, gt_per_frame,
                                     bbox_frames, valid_frames, res=(224, 224)):
        """Transform GT wrist position and orientation from world to camera frame.

        MANO's convention is `joint_0_final = joint_0_canonical(betas) + transl`,
        i.e. `transl` is an offset from the beta-specific canonical wrist location,
        NOT an absolute world position. So naively doing `R_cw @ transl + t_cw`
        places joint 0 at `joint_0_canonical + R_cw @ transl + t_cw` in camera
        frame, which differs from the true camera-frame wrist position by
        `(I - R_cw) @ joint_0_canonical`. This manifested as a visible
        rotation+mirror offset in the GT-vs-pred overlay.

        Correct transform (per frame per hand):
            1. joint_0_world = joint_0_canonical(betas) + transl_world
            2. joint_0_cam   = R_cw @ joint_0_world + t_cw
            3. transl_cam    = joint_0_cam - joint_0_canonical(betas)
            4. R_cam         = R_cw @ R_world   (unchanged)

        Modifies gt_per_frame in-place.
        Returns True on success, False if calibration files are missing.
        """
        from scipy.spatial.transform import Rotation

        seq_data = HOT3DHandDataset._load_camera_seq_data(seq_path)
        if seq_data is None:
            return False
        T_device_camera, _cam_calib, headset_poses, headset_ts, _calib_path = seq_data

        # Cache joint_0_canonical per (is_right, betas) since betas are usually
        # constant across a sequence and MANO is fairly slow.
        self.mano_model._ensure_device(torch.device("cpu"))
        canon_cache = {}

        def canonical_joint_0(betas_np, is_right):
            key = (is_right, tuple(np.round(betas_np, 5)))
            if key in canon_cache:
                return canon_cache[key]
            layer = self.mano_model.right if is_right else self.mano_model.left
            out = layer(
                betas=torch.tensor([betas_np], dtype=torch.float32),
                global_orient=torch.zeros(1, 3),
                hand_pose=torch.zeros(1, 15),
                transl=torch.zeros(1, 3),
                return_verts=True,
            )
            j0 = out.joints[0, 0].detach().numpy().astype(np.float64)
            canon_cache[key] = j0
            return j0

        for frame_i in range(len(gt_per_frame)):
            _T_world_device, T_camera_world = HOT3DHandDataset._frame_camera_transforms(
                frame_i, n_video, hand_ts_sorted, headset_poses, headset_ts, T_device_camera,
            )
            R_cw = T_camera_world[:3, :3]
            t_cw = T_camera_world[:3, 3]

            gt_vec = gt_per_frame[frame_i]  # [64] = 2 hands x 32

            for hand_idx in range(NUM_HANDS):
                off = hand_idx * HAND_PARAM_DIM
                t_world = gt_vec[off:off + 3].numpy().astype(np.float64)
                q_wxyz  = gt_vec[off + 3:off + 7].numpy().astype(np.float64)
                betas   = gt_vec[off + 22:off + 32].numpy().astype(np.float64)

                if np.abs(t_world).sum() < 1e-8 and np.abs(q_wxyz).sum() < 1e-8:
                    continue

                j0_canon = canonical_joint_0(betas, is_right=(hand_idx == 1))

                # Joint 0 in world → camera → back to transl-offset form
                j0_world = j0_canon + t_world
                j0_cam   = R_cw @ j0_world + t_cw
                t_cam    = j0_cam - j0_canon

                gt_vec[off]     = float(t_cam[0])
                gt_vec[off + 1] = float(t_cam[1])
                gt_vec[off + 2] = float(t_cam[2])

                # Rotation transforms as before
                q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
                R_world = Rotation.from_quat(q_xyzw).as_matrix()
                R_cam = R_cw @ R_world
                q_cam_xyzw = Rotation.from_matrix(R_cam).as_quat()
                q_cam_wxyz = np.array([
                    q_cam_xyzw[3], q_cam_xyzw[0], q_cam_xyzw[1], q_cam_xyzw[2]
                ])

                gt_vec[off + 3:off + 7] = torch.from_numpy(q_cam_wxyz.astype(np.float32))

        return True


def discover_sequences(data_root):
    seqs = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if not os.path.isdir(path):
            continue
        if (os.path.exists(os.path.join(path, "video_main_rgb.mp4")) and
                os.path.exists(os.path.join(path, "hand_data/mano_hand_pose_trajectory.jsonl"))):
            seqs.append(path)
    return seqs


# ------------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------------

def build_views(imgs, num_frames, device, hand_bboxes=None, hand_valid=None,
                 crop_local_output=False):
    B, _, _, H, W = imgs.shape
    views = {
        "img":          imgs,
        "is_target":    torch.zeros((B, num_frames), dtype=torch.bool, device=device),
        "timestamp":    torch.arange(num_frames, device=device).unsqueeze(0).expand(B, -1),
        "is_static":    torch.zeros((B, num_frames), dtype=torch.bool, device=device),
        "valid_mask":   torch.ones((B, num_frames, H, W), dtype=torch.bool, device=device),
        "camera_poses": torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, num_frames, 4, 4),
        "camera_intrs": torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, num_frames, 3, 3),
        "depthmap":     torch.ones((B, num_frames, H, W), device=device),
    }
    if hand_bboxes is not None:
        views["hand_bboxes"] = hand_bboxes
    if hand_valid is not None:
        views["hand_valid"] = hand_valid
    if crop_local_output:
        views["crop_local_output"] = True
    return views


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------

def setup_vis_items(dataset, num_vis_frames, seq_cache, mano_model, preload=False, draw_joint_dots=True):
    """Set up visualization entries for a dataset.

    Args:
        dataset: HOT3DHandDataset
        preload: if True, also load img/gt tensors (for train vis)
        draw_joint_dots: if False, skip 2D joint dot rendering in render_hand_comparison

    Returns:
        List of dicts with 'clip_idx', 'ctx', and optionally 'img'/'gt'.
    """
    from scripts.hand_vis_utils import setup_vis_context

    n = len(dataset.clips)
    step = max(1, n // num_vis_frames)
    items = []
    for clip_idx in torch.arange(0, n, step).tolist()[:num_vis_frames]:
        clip = dataset.clips[clip_idx]
        seq_path = clip["seq_path"]
        if seq_path not in seq_cache:
            seq_cache[seq_path] = setup_vis_context(seq_path, mano_model=mano_model)
        ctx = seq_cache[seq_path]
        if ctx is None:
            continue

        entry = {
            "clip_idx": clip_idx,
            "ctx": {**ctx, "frame_offset": clip["frame_offset"], "draw_joint_dots": draw_joint_dots},
        }
        if preload:
            data = dataset[clip_idx]
            entry["img"] = data["img"]
            entry["gt"] = data["gt"]
            if "hand_bboxes" in data:
                entry["hand_bboxes"] = data["hand_bboxes"]
                entry["hand_valid"] = data["hand_valid"]
        items.append(entry)
    return items


def render_vis_list(vis_items, gt_pred_pairs, render_fn):
    """Render visualization images from gt/pred pairs.

    Args:
        vis_items: list of dicts with 'ctx' (containing frame_offset)
        gt_pred_pairs: list of (gt_tensor, pred_tensor) aligned with vis_items
        render_fn: render_hand_comparison function

    Returns:
        List of wandb.Image objects.
    """
    images = []
    for i, (item, (gt, pred)) in enumerate(zip(vis_items, gt_pred_pairs)):
        ctx = item["ctx"]
        vis_img = render_fn(ctx, ctx["frame_offset"], gt, pred)
        if vis_img is not None:
            images.append(wandb.Image(vis_img, caption=f"Frame {i}: Solid=GT, Wireframe=Pred"))
    return images


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def run_validation(model, val_loader, num_frames, device, criterion_kp3d, criterion_kp2d, criterion_param, mano_model, loss_weights, vis_clip_indices=None, consistency_fn=None):
    """Run validation and optionally capture gt/pred at specific clip indices.

    Returns (supervised_val_loss, val_terms, val_consistency, captured). The
    consistency value is 0.0 when ``consistency_fn`` is None.
    """
    model.eval()
    # Phase 3: free cached allocator blocks before validation so the renderer
    # has a clean slate — training accumulation leaves a lot of fragmented
    # reserved memory behind, and GS rendering is the next peak.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    val_loss = 0.0
    val_consistency = 0.0
    val_terms = {
        "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
        "kp3d": 0.0, "kp2d": 0.0,
    }
    captured = {}
    batch_size = val_loader.batch_size
    with torch.no_grad():
        for batch_idx, vbatch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
            imgs = vbatch["img"].to(device)
            gt = vbatch["gt"].to(device)
            hb = vbatch["hand_bboxes"].to(device) if "hand_bboxes" in vbatch else None
            hv = vbatch["hand_valid"].to(device)  if "hand_valid"  in vbatch else None
            views = build_views(imgs, num_frames, device, hb, hv)
            preds = model(views, is_inference=False, use_motion=False)
            pred_params = preds["hand_joints"]

            pred_joints = compute_joints_from_batch(pred_params, mano_model, device)
            B, S, H, J, _ = pred_joints.shape

            if hv is not None:
                has_hand = hv.float()
            else:
                gt_pack = gt.view(*gt.shape[:-1], NUM_HANDS, HAND_PARAM_DIM)
                has_hand = (gt_pack.abs().sum(dim=-1) > 1e-6).float()

            param_losses = criterion_param(pred_params, gt, has_hand)

            gt_joints = vbatch["gt_joints"].to(device)
            gt_conf = has_hand.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, J, 1)
            gt_input = torch.cat([gt_joints, gt_conf], dim=-1)
            pred_flat = pred_joints.view(B * S * H, 1, J, 3)
            gt_flat   = gt_input.view(B * S * H, 1, J, 4)
            loss_kp3d = criterion_kp3d(pred_flat, gt_flat, pelvis_id=0)

            loss_kp2d = torch.zeros((), device=device)
            if "gt_joints_2d" in vbatch:
                cam_intr = vbatch["cam_intrinsics"].to(device)            # [B, 3]
                N = B * S
                pred_j  = pred_joints.view(N, H, J, 3)
                focal   = cam_intr[:, 0].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cx      = cam_intr[:, 1].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cy      = cam_intr[:, 2].unsqueeze(1).expand(B, S).reshape(N, 1, 1)

                # Clamp at 5 cm (below camera focal distance is nonphysical)
                # to keep focal*x/z finite when early pred params are degenerate.
                z = pred_j[..., 2].clamp_min(0.05)
                col = focal * pred_j[..., 0] / z + cx
                row = focal * pred_j[..., 1] / z + cy
                IMAGE_WIDTH = 1408.0
                # 90° CW to match project_vertices: (col, row) → (W-1-row, col)
                u = (IMAGE_WIDTH - 1.0) - row
                v = col
                pred_2d = torch.stack([u, v], dim=-1)

                pred_2d_norm = pred_2d / IMAGE_WIDTH - 0.5
                gt_2d        = vbatch["gt_joints_2d"].to(device)
                gt_2d_norm   = gt_2d.clone()
                gt_2d_norm[..., :2] = gt_2d[..., :2] / IMAGE_WIDTH - 0.5
                gt_2d_norm[..., 2]  = gt_2d_norm[..., 2] * has_hand.unsqueeze(-1)

                pred_2d_flat = pred_2d_norm.view(N * H, 1, J, 2)
                gt_2d_flat   = gt_2d_norm.view(N * H, 1, J, 3)
                loss_kp2d    = criterion_kp2d(pred_2d_flat, gt_2d_flat)

            loss = (
                loss_weights["transl"]        * param_losses["transl"]
                + loss_weights["global_orient"] * param_losses["global_orient"]
                + loss_weights["hand_pose"]     * param_losses["hand_pose"]
                + loss_weights["betas"]         * param_losses["betas"]
                + loss_weights["kp3d"]          * loss_kp3d
                + loss_weights["kp2d"]          * loss_kp2d
            )

            val_loss += loss.item()
            for k in ("transl", "global_orient", "hand_pose", "betas"):
                val_terms[k] += param_losses[k].item()
            val_terms["kp3d"]  += loss_kp3d.item()
            val_terms["kp2d"]  += loss_kp2d.item()

            if consistency_fn is not None:
                val_consistency += consistency_fn(preds, views).item()

            if vis_clip_indices:
                # For vis we need camera-space predictions for rendering
                vis_preds = model(build_views(imgs, num_frames, device, hb, hv, crop_local_output=False), is_inference=False, use_motion=False)
                for item_idx in range(imgs.shape[0]):
                    clip_idx = batch_idx * batch_size + item_idx
                    if clip_idx in vis_clip_indices:
                        captured[clip_idx] = {
                            "gt": gt[item_idx, 0].cpu(),
                            "pred": vis_preds["hand_joints"][item_idx, 0].cpu(),
                        }

    n = max(len(val_loader), 1)
    val_terms = {k: v / n for k, v in val_terms.items()}
    return val_loss / n, val_terms, val_consistency / n, captured


def render_train_vis(model, train_vis_items, num_frames, device, render_fn):
    """Forward-pass fixed train clips and render visualizations."""
    model.eval()
    with torch.no_grad():
        imgs = torch.stack([it["img"] for it in train_vis_items]).to(device)
        hb = torch.stack([it["hand_bboxes"] for it in train_vis_items]).to(device) if "hand_bboxes" in train_vis_items[0] else None
        hv = torch.stack([it["hand_valid"]  for it in train_vis_items]).to(device) if "hand_valid"  in train_vis_items[0] else None
        # crop_local_output=False so predictions are in camera space for rendering
        preds = model(build_views(imgs, num_frames, device, hb, hv, crop_local_output=False), is_inference=False, use_motion=False)
        pairs = [
            (item["gt"][0], preds["hand_joints"][i, 0].cpu())
            for i, item in enumerate(train_vis_items)
        ]
    return render_vis_list(train_vis_items, pairs, render_fn)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def compute_joints_from_batch(params, mano_model, device):
    """Differentiable, batched conversion of 32-D MANO params to 3D joints.

    Args:
        params: [B, S, 64] — two hands packed as (left[32], right[32]).
        mano_model: MANOModel wrapper.

    Returns:
        [B, S, 2, 16, 3] joint tensor with autograd linked back to `params`.
    """
    B, S, D = params.shape
    assert D == NUM_HANDS * HAND_PARAM_DIM
    N = B * S
    flat = params.view(N, NUM_HANDS, HAND_PARAM_DIM)  # [N, 2, 32]
    left  = mano_model.get_joints_batched(flat[:, 0], is_right=False, device=device)  # [N, 16, 3]
    right = mano_model.get_joints_batched(flat[:, 1], is_right=True,  device=device)
    joints = torch.stack([left, right], dim=1)  # [N, 2, 16, 3]
    return joints.view(B, S, NUM_HANDS, joints.shape[-2], 3)


def _save_hand_ckpt(model, path):
    """Save hand-head weights, plus refiner weights when Phase 2+ is active."""
    ckpt = {"hand_head": model.hand_head.state_dict()}
    refiner = getattr(model, "hand_gs_refiner", None)
    if refiner is not None:
        ckpt["hand_gs_refiner"] = refiner.state_dict()
    torch.save(ckpt, path)


def _apply_overrides(cfg, overrides):
    """Apply dotted-key overrides like 'training.lr=3e-4' to a nested dict."""
    for ov in overrides:
        key, val = ov.split("=", 1)
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)



def train():

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hand_head.yaml")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VAL",
                        help="Config overrides, e.g. training.lr=3e-4 model.hamer_head_kwargs.depth=4")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.overrides:
        _apply_overrides(cfg, args.overrides)
        print(f"Config overrides: {args.overrides}")

    data_cfg     = cfg["data"]
    model_cfg    = cfg["model"]
    training_cfg = cfg["training"]
    wandb_cfg    = cfg.get("wandb", {})
    debug_cfg    = cfg.get("debug", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model ---
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("reconstructor", checkpoint))
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. New (hand head) keys: {len(missing)}")
    model.to(device)

    vis_cfg = cfg.get("visualization", {})
    mano_folder = vis_cfg.get("mano_model_folder")
    if not mano_folder:
        raise RuntimeError("MANO model folder must be specified in config for training and visualization")
    from scripts.hand_vis_utils import MANOModel
    mano_model = MANOModel(mano_folder)

    criterion_kp3d  = Keypoint3DLoss(loss_type='l2').to(device)
    criterion_kp2d  = Keypoint2DLoss(loss_type='l1').to(device)
    criterion_param = ParameterLoss().to(device)

    hand_params = list(model.hand_head.parameters())
    refiner = getattr(model, "hand_gs_refiner", None)
    if refiner is not None:
        hand_params += list(refiner.parameters())
        print(f"Hand head + refiner parameters: {sum(p.numel() for p in hand_params):,}")
    else:
        print(f"Hand head parameters: {sum(p.numel() for p in hand_params):,}")

    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))

    # --- 3D self-supervised consistency loss (Phase 3) ---
    consistency_cfg = training_cfg.get("consistency_loss", {}) or {}
    consistency_fn = None
    cons_weight_target = 0.0
    cons_warmup = 0
    if consistency_cfg.get("enabled", False):
        if refiner is None:
            print("[WARN] consistency_loss.enabled=true but hand_gs_refiner is absent "
                  "(enable_gs=false or enable_hand_gs_refiner=false) — skipping.")
        else:
            consistency_fn = HandGSConsistencyLoss(
                mano_folder=consistency_cfg["mano_folder"],
                target=consistency_cfg.get("target", "vertices"),
                mask_threshold=consistency_cfg.get("mask_threshold", 0.5),
                max_samples_per_frame=consistency_cfg.get("max_samples_per_frame", 128),
            ).to(device)
            cons_weight_target = float(consistency_cfg.get("weight", 0.0005))
            cons_warmup = int(consistency_cfg.get("warmup_steps", 0))
            print(f"Consistency loss ENABLED | weight={cons_weight_target} "
                  f"warmup_steps={cons_warmup} target={consistency_cfg.get('target', 'vertices')}")

    # --- Data ---
    all_seqs = discover_sequences(data_cfg["data_root"])
    if not all_seqs:
        raise RuntimeError(f"No sequences found in {data_cfg['data_root']}")
    print(f"Found {len(all_seqs)} sequences")

    if debug_cfg.get("enabled", False):
        max_seqs = debug_cfg.get("max_sequences", 5)
        all_seqs = all_seqs[:max_seqs]
        print(f"[DEBUG] Limited to {len(all_seqs)} sequences")

    num_frames       = data_cfg["num_frames"]
    res              = tuple(data_cfg["resolution"])
    clip_stride      = data_cfg.get("clip_stride", num_frames)
    batch_size       = training_cfg.get("batch_size", 2)
    grad_accum_steps = training_cfg.get("grad_accum_steps", 1)
    num_workers      = data_cfg.get("num_workers", 4)

    # Hand-crop dataset options (mirror the model flag)
    use_hand_crop = model_cfg.get("hand_head_type") == "hand_crop" or model_cfg.get("use_hand_crop", False)
    rescale_factor = cfg.get("hand_crop", {}).get("rescale_factor", 2.0)

    ds_kwargs = dict(
        num_frames=num_frames, res=res, clip_stride=clip_stride,
        use_hand_crop=use_hand_crop, rescale_factor=rescale_factor,
    )

    if debug_cfg.get("single_frame", False):
        # Overfit on a single clip from the middle of the first sequence
        single_set = HOT3DHandDataset(all_seqs[:1], mano_model, **ds_kwargs)
        mid = len(single_set.clips) // 2
        single_set.clips = [single_set.clips[mid]]
        train_set = val_set = single_set
        print(f"[DEBUG] Single-frame overfit: seq={os.path.basename(all_seqs[0])}, clip offset={single_set.clips[0]['frame_offset']}")
    else:
        random.seed(training_cfg.get("seed", 42))
        random.shuffle(all_seqs)
        n_val = int(len(all_seqs) * float(data_cfg.get("val_split", 0.1)))
        if n_val == 0:
            print("[WARN] No validation sequences — validation disabled, no best checkpoint will be saved")
        val_seqs, train_seqs = all_seqs[:n_val], all_seqs[n_val:]
        train_set = HOT3DHandDataset(train_seqs, mano_model, **ds_kwargs)
        val_set = HOT3DHandDataset(val_seqs, mano_model, **ds_kwargs) if val_seqs else None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)

    if val_set is not None and len(val_set.clips) > 0:
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=False)
        print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")
    else:
        val_set = None
        val_loader = None
        print(f"Train clips: {len(train_set)} | Val clips: 0")

    # --- Visualization setup ---
    num_vis_frames = vis_cfg.get("num_vis_frames", 4)
    draw_joint_dots = vis_cfg.get("draw_joint_dots", True)
    render_fn = None
    val_vis_items = []
    train_vis_items = []

    has_val_clips = val_set is not None and len(val_set.clips) > 0
    if mano_folder and (has_val_clips or len(train_set.clips) > 0):
        from scripts.hand_vis_utils import render_hand_comparison

        render_fn = render_hand_comparison
        seq_cache = {}

        if has_val_clips:
            val_vis_items = setup_vis_items(val_set, num_vis_frames, seq_cache, mano_model, draw_joint_dots=draw_joint_dots)
        train_vis_items = setup_vis_items(train_set, num_vis_frames, seq_cache, mano_model, preload=True, draw_joint_dots=draw_joint_dots)

        if val_vis_items or train_vis_items:
            print(f"[VIS] {len(val_vis_items)} val + {len(train_vis_items)} train frames across {len(seq_cache)} sequences")

    val_vis_clip_indices = {it["clip_idx"] for it in val_vis_items} or None

    # --- Optimizer & scheduler ---
    epochs     = training_cfg["epochs"]
    steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = epochs * steps_per_epoch
    optimizer  = Adam(hand_params, lr=float(training_cfg["lr"]))
    scheduler  = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=float(training_cfg.get("min_lr", 1e-6)))

    log_every  = training_cfg.get("log_every", 500)
    val_every  = training_cfg.get("val_every", 2000)
    save_every = training_cfg.get("save_every", 2000)
    output_dir = training_cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training on {device} | {epochs} epochs | batch_size={batch_size} | grad_accum_steps={grad_accum_steps}")

    # --- W&B ---
    use_wandb = wandb_cfg.get("enabled", False)
    if use_wandb:
        wandb.init(
            project=wandb_cfg.get("project", "hand-head-training"),
            entity=wandb_cfg.get("entity") or None,
            name=wandb_cfg.get("run_name") or None,
            tags=wandb_cfg.get("tags") or [],
            notes=wandb_cfg.get("notes") or None,
            config=cfg,
        )

    # --- Diagnostic: first-batch GT translation stats (sanity check the
    # crop-local / camera-frame transform after the fisheye fix). ---
    _diag_batch = next(iter(train_loader))
    _diag_gt = _diag_batch["gt"]  # [B, S, 64]
    for _hand_idx, _name in enumerate(("left", "right")):
        _off = _hand_idx * HAND_PARAM_DIM
        _t = _diag_gt[..., _off:_off + 3]
        _nz = _t.abs().sum(dim=-1) > 1e-6
        if _nz.any():
            _tv = _t[_nz]
            print(f"[DIAG] {_name} hand GT t_cam: "
                  f"min={_tv.min().item():.4f} max={_tv.max().item():.4f} "
                  f"mean={_tv.mean().item():.4f} std={_tv.std().item():.4f} "
                  f"(N={_nz.sum().item()})")
        else:
            print(f"[DIAG] {_name} hand GT t_cam: all-zero in first batch")
    del _diag_batch, _diag_gt

    # --- Sanity check: MANO path must be differentiable w.r.t. pred_params.
    # If this assert fires, loss_kp3d / loss_kp2d will silently contribute no
    # gradient to the head and training will look fine on the param losses only.
    _probe = 0.01 * torch.randn(1, num_frames, NUM_HANDS * HAND_PARAM_DIM,
                                device=device)
    _probe = _probe.detach().clone().requires_grad_(True)
    _probe_joints = compute_joints_from_batch(_probe, mano_model, device)
    _probe_joints.sum().backward()
    assert _probe.grad is not None and _probe.grad.abs().sum().item() > 0, (
        "MANO joint computation is not differentiable — check "
        "MANOModel.get_joints_batched / quat_wxyz_to_axis_angle_torch."
    )
    del _probe, _probe_joints

    best_val_loss = float("inf")
    global_step = 0

    # --- Training loop ---
    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_consistency = 0.0
        accum_terms = {
            "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
            "kp3d": 0.0, "kp2d": 0.0,
        }
        reset_diag_flag()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}", leave=False)):
            imgs = batch["img"].to(device)
            gt_params = batch["gt"].to(device)
            gt_joints = batch["gt_joints"].to(device)

            hb = batch["hand_bboxes"].to(device) if "hand_bboxes" in batch else None
            hv = batch["hand_valid"].to(device)  if "hand_valid"  in batch else None

            views = build_views(imgs, num_frames, device, hb, hv)
            preds = model(views, is_inference=False, use_motion=False)
            pred_params = preds["hand_joints"] # [B, S, 64]

            pred_joints = compute_joints_from_batch(pred_params, mano_model, device)

            loss_kp2d = torch.zeros((), device=device)

            # Per-hand presence mask — prefer the dataset's hand_valid (derived from
            # the projected-bbox pipeline); fall back to a params-are-nonzero check.
            if hv is not None:
                has_hand = hv.to(device).float()                 # [B, S, 2]
            else:
                gt_pack = gt_params.view(*gt_params.shape[:-1], NUM_HANDS, HAND_PARAM_DIM)
                has_hand = (gt_pack.abs().sum(dim=-1) > 1e-6).float()

            # Parameter loss — split per MANO key, each masked per-hand.
            # Returns dict with 'transl', 'global_orient', 'hand_pose', 'betas'.
            param_losses = criterion_param(pred_params, gt_params, has_hand)

            # 3D keypoint loss. Confidence is 1 where the hand is present, 0 otherwise —
            # absent hands have zero GT joints but MANO's default pose would otherwise
            # produce a constant ~0.2 residual that never decays.
            B, S, H, J, _ = pred_joints.shape
            gt_conf = has_hand.unsqueeze(-1).unsqueeze(-1).expand(B, S, H, J, 1)
            gt_input = torch.cat([gt_joints, gt_conf], dim=-1)                # [B, S, H, J, 4]
            pred_flat = pred_joints.view(B * S * H, 1, J, 3)
            gt_flat   = gt_input.view(B * S * H, 1, J, 4)
            loss_kp3d = criterion_kp3d(pred_flat, gt_flat, pelvis_id=0)

            # 2D reprojection loss. pred_joints are camera-frame (post-transform),
            # so skip the world→camera extrinsic and project with intrinsics only.
            if "gt_joints_2d" in batch:
                cam_intr = batch["cam_intrinsics"].to(device)              # [B, 3]
                N = B * S
                pred_j  = pred_joints.view(N, H, J, 3)                     # [N, H, J, 3]
                focal   = cam_intr[:, 0].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cx      = cam_intr[:, 1].unsqueeze(1).expand(B, S).reshape(N, 1, 1)
                cy      = cam_intr[:, 2].unsqueeze(1).expand(B, S).reshape(N, 1, 1)

                # Clamp at 5 cm (below camera focal distance is nonphysical)
                # to keep focal*x/z finite when early pred params are degenerate.
                z = pred_j[..., 2].clamp_min(0.05)
                col = focal * pred_j[..., 0] / z + cx
                row = focal * pred_j[..., 1] / z + cy
                IMAGE_WIDTH = 1408.0
                # 90° CW to match project_vertices: (col, row) → (W-1-row, col)
                u = (IMAGE_WIDTH - 1.0) - row
                v = col
                pred_2d = torch.stack([u, v], dim=-1)                      # [N, H, J, 2]

                # Normalize to [-0.5, 0.5] so residuals match HaMeR's convention.
                pred_2d_norm = pred_2d / IMAGE_WIDTH - 0.5
                gt_2d        = batch["gt_joints_2d"].to(device)            # [B, S, H, J, 3]
                gt_2d_norm   = gt_2d.clone()
                gt_2d_norm[..., :2] = gt_2d[..., :2] / IMAGE_WIDTH - 0.5
                gt_2d_norm[..., 2]  = gt_2d_norm[..., 2] * has_hand.unsqueeze(-1)

                pred_2d_flat = pred_2d_norm.view(N * H, 1, J, 2)
                gt_2d_flat   = gt_2d_norm.view(N * H, 1, J, 3)
                loss_kp2d    = criterion_kp2d(pred_2d_flat, gt_2d_flat)

            w = cfg["loss_weights"]
            supervised_loss = (
                w["transl"]        * param_losses["transl"]
                + w["global_orient"] * param_losses["global_orient"]
                + w["hand_pose"]     * param_losses["hand_pose"]
                + w["betas"]         * param_losses["betas"]
                + w["kp3d"]          * loss_kp3d
                + w["kp2d"]          * loss_kp2d
            )

            # 3D consistency: pulls refined Gaussian centers toward predicted
            # MANO vertices. Ramp the weight 0 → target over warmup_steps so
            # the supervised head stabilises before the refiner starts tugging
            # on Gaussian positions.
            if consistency_fn is not None:
                ramp = min(1.0, global_step / cons_warmup) if cons_warmup > 0 else 1.0
                cons_weight = cons_weight_target * ramp
                consistency_loss = consistency_fn(preds, views)
                loss = supervised_loss + cons_weight * consistency_loss
            else:
                cons_weight = 0.0
                consistency_loss = supervised_loss.new_zeros(())
                loss = supervised_loss

            # Circuit breaker: skip this accumulation window if the loss is
            # non-finite (NaN/Inf). Clears any partial grads so we don't poison
            # the next step.
            if not torch.isfinite(loss):
                tqdm.write(
                    f"[WARN] Non-finite loss at step {global_step} "
                    f"(supervised={supervised_loss.item():.4f} "
                    f"consistency={consistency_loss.item():.4f}) — skipping update."
                )
                optimizer.zero_grad()
                accum_loss = 0.0
                accum_consistency = 0.0
                accum_terms = {
                    "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
                    "kp3d": 0.0, "kp2d": 0.0,
                }
                continue

            (loss / grad_accum_steps).backward()
            accum_loss += loss.item()
            accum_consistency += consistency_loss.item()
            for k in ("transl", "global_orient", "hand_pose", "betas"):
                accum_terms[k] += param_losses[k].item()
            accum_terms["kp3d"]  += loss_kp3d.item()
            accum_terms["kp2d"]  += loss_kp2d.item()

            if (batch_idx + 1) % grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(hand_params, max_norm=grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                avg_loss = accum_loss / grad_accum_steps
                avg_consistency = accum_consistency / grad_accum_steps
                avg_terms = {k: v / grad_accum_steps for k, v in accum_terms.items()}
                accum_loss = 0.0
                accum_consistency = 0.0
                accum_terms = {
                    "transl": 0.0, "global_orient": 0.0, "hand_pose": 0.0, "betas": 0.0,
                    "kp3d": 0.0, "kp2d": 0.0,
                }
                global_step += 1

                # --- Train logging ---
                if use_wandb:
                    wandb.log({"train/loss": avg_loss,
                               "train/loss_transl":        avg_terms["transl"],
                               "train/loss_global_orient": avg_terms["global_orient"],
                               "train/loss_hand_pose":     avg_terms["hand_pose"],
                               "train/loss_betas":         avg_terms["betas"],
                               "train/loss_kp3d":          avg_terms["kp3d"],
                               "train/loss_kp2d":          avg_terms["kp2d"],
                               "train/consistency_loss":   avg_consistency,
                               "train/consistency_weight": cons_weight,
                               "train/grad_norm":          grad_norm.item(),
                               "lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % log_every == 0 or global_step == 1:
                    lr = scheduler.get_last_lr()[0]
                    tqdm.write(
                        f"  step {global_step} | train_loss={avg_loss:.4f} "
                        f"(t={avg_terms['transl']:.4f} o={avg_terms['global_orient']:.4f} "
                        f"p={avg_terms['hand_pose']:.4f} b={avg_terms['betas']:.4f} "
                        f"kp3d={avg_terms['kp3d']:.4f} kp2d={avg_terms['kp2d']:.4f} "
                        f"cons={avg_consistency:.4f} w={cons_weight:.2e}) "
                        f"| grad_norm={grad_norm.item():.4f} | lr={lr:.2e}"
                    )
                    if use_wandb and train_vis_items:
                        train_images = render_train_vis(model, train_vis_items, num_frames, device, render_fn)
                        if train_images:
                            wandb.log({"train/hand_overlay": train_images}, step=global_step)
                        model.train()

                # --- Validation ---
                if val_loader and (global_step % val_every == 0 or global_step == 1):
                    val_loss, val_terms, val_consistency, captured = run_validation(
                        model, val_loader, num_frames, device,
                        criterion_kp3d, criterion_kp2d, criterion_param,
                        mano_model, cfg["loss_weights"], val_vis_clip_indices,
                        consistency_fn=consistency_fn,
                    )
                    tqdm.write(
                        f"  step {global_step} | val_loss={val_loss:.4f} "
                        f"(t={val_terms['transl']:.4f} o={val_terms['global_orient']:.4f} "
                        f"p={val_terms['hand_pose']:.4f} b={val_terms['betas']:.4f} "
                        f"kp3d={val_terms['kp3d']:.4f} kp2d={val_terms['kp2d']:.4f}"
                        f" cons={val_consistency:.4f})"
                    )

                    if use_wandb:
                        log_dict = {"val/loss": val_loss,
                                    "val/loss_transl":        val_terms["transl"],
                                    "val/loss_global_orient": val_terms["global_orient"],
                                    "val/loss_hand_pose":     val_terms["hand_pose"],
                                    "val/loss_betas":         val_terms["betas"],
                                    "val/loss_kp3d":          val_terms["kp3d"],
                                    "val/loss_kp2d":          val_terms["kp2d"]}
                        if consistency_fn is not None:
                            log_dict["val/consistency_loss"] = val_consistency
                        if val_vis_items:
                            pairs = [
                                (captured[it["clip_idx"]]["gt"], captured[it["clip_idx"]]["pred"])
                                for it in val_vis_items if it["clip_idx"] in captured
                            ]
                            val_images = render_vis_list(val_vis_items, pairs, render_fn)
                            if val_images:
                                log_dict["val/hand_overlay"] = val_images
                        wandb.log(log_dict, step=global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        _save_hand_ckpt(model, os.path.join(output_dir, "best_val_loss.pt"))
                        tqdm.write("  -> New best. Saved.")
                    model.train()

                if global_step % save_every == 0:
                    _save_hand_ckpt(model, os.path.join(output_dir, f"checkpoint_{global_step}.pt"))

        # Flush leftover gradients from an incomplete accumulation window
        if (batch_idx + 1) % grad_accum_steps != 0:
            optimizer.zero_grad()

    # --- Save final ---
    final_path = os.path.join(output_dir, "hand_head_final.pt")
    _save_hand_ckpt(model, final_path)
    print(f"Final weights saved to: {final_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
