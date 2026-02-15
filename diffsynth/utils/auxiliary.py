import torch
import numpy as np
import json
import os
from PIL import Image
from decord import VideoReader
from scipy.spatial.transform import Rotation, Slerp


def homo_matrix_inverse(homo_matrix):
    """
    Computes the inverse of a batch of 4x4 (or 3x4) homogeneous transformation matrices.
    """
    assert homo_matrix.shape[-2:] == (4, 4) or homo_matrix.shape[-2:] == (3, 4), "Input must be a batch of 4x4 or 3x4 matrices"

    R, T = homo_matrix[..., :3, :3].reshape(-1, 3, 3), homo_matrix[..., :3, 3:4].reshape(-1, 3, 1)

    with torch.amp.autocast("cuda", enabled=False):
        R_inv = R.transpose(-1, -2)
        T_inv = -torch.bmm(R_inv, T)

    homo_inv = torch.eye(4, device=homo_matrix.device, dtype=homo_matrix.dtype)[None].repeat(R_inv.shape[0], 1, 1)
    homo_inv[:, :3, :3] = R_inv
    homo_inv[:, :3, 3:4] = T_inv
    homo_inv = homo_inv.reshape(*homo_matrix.shape[:-2], 4, 4)
    return homo_inv


def average_filter(depth_map, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1

    device = depth_map.device
    dtype = depth_map.dtype
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device, dtype=dtype) / (kernel_size * kernel_size)

    # Prepare depth map for convolution
    depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Apply padding to preserve spatial dimensions
    padding = kernel_size // 2
    depth_map_padded = torch.nn.functional.pad(depth_map, (padding, padding, padding, padding), mode="replicate")

    # Apply convolution
    smoothed_depth = torch.nn.functional.conv2d(depth_map_padded, kernel, padding=0)

    return smoothed_depth.squeeze(0).squeeze(0)


def fast_perceptual_color_distance(color1, color2):
    """
    Fast RGB perceptual color distance approximation.
    Based on the formula in "A low-cost approximation" at https://www.compuphase.com/cmetric.htm.

    Args:
        color1, color2: [*, 3] tensors with RGB values in [0, 1]
    Returns:
        distance: [*] tensor with perceptual color distances
    """
    # Convert to [0, 255] range for the formula
    c1 = color1 * 255.0
    c2 = color2 * 255.0

    # Calculate mean red value
    r_bar = (c1[..., 0] + c2[..., 0]) / 2.0  # [N]

    # Calculate color differences
    delta_r = c1[..., 0] - c2[..., 0]  # [N]
    delta_g = c1[..., 1] - c2[..., 1]  # [N]
    delta_b = c1[..., 2] - c2[..., 2]  # [N]

    # Calculate weighted distance according to the formula
    # ΔC = sqrt((2 + r̄/256) × ΔR² + 4 × ΔG² + (2 + (255-r̄)/256) × ΔB²)
    weight_r = 2.0 + r_bar / 256.0
    weight_g = 4.0
    weight_b = 2.0 + (255.0 - r_bar) / 256.0

    distance = torch.sqrt(
        weight_r * delta_r**2 +
        weight_g * delta_g**2 +
        weight_b * delta_b**2
    )

    return distance


def pixel_to_world_coords(pixel_x, pixel_y, depths, intrinsic, extrinsic):
    """
    Convert pixel coordinates with depths to world coordinates.

    Args:
        pixel_x: Pixel x-coordinates [N]
        pixel_y: Pixel y-coordinates [N]
        depths: Depth values at each pixel [N]
        intrinsic: Camera intrinsic matrix [3, 3]
        extrinsic: Camera extrinsic matrix (world-to-camera) [3, 4] or [4, 4]

    Returns:
        world_coords: 3D coordinates in world space [N, 3]
    """
    # Extract intrinsic parameters
    fu = intrinsic[0, 0]
    fv = intrinsic[1, 1]
    cu = intrinsic[0, 2]
    cv = intrinsic[1, 2]

    # Convert pixel coordinates to camera coordinates
    x_cam = (pixel_x - cu) * depths / (fu + 1e-6)
    y_cam = (pixel_y - cv) * depths / (fv + 1e-6)
    z_cam = depths
    cam_coords = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [N, 3]

    # Extract rotation and translation from extrinsic matrix
    R = extrinsic[:3, :3]  # [3, 3]
    T = extrinsic[:3, 3:4]  # [3, 1]

    # Convert camera coordinates to world coordinates
    # world = R^T @ (cam - T) = R^T @ cam - R^T @ T
    R_transposed = R.transpose(0, 1)  # [3, 3]
    t_world = -torch.matmul(R_transposed, T).squeeze(-1)  # [3]
    world_coords = torch.matmul(cam_coords, R) + t_world  # [N, 3]

    return world_coords


def center_crop(image, resolution):
    """Center crop a PIL Image to target resolution, scaling first to cover."""
    width, height = image.size
    target_width, target_height = resolution

    scale_width = target_width / width
    scale_height = target_height / height
    scale_final = max(scale_width, scale_height)
    output_width = int(width * scale_final)
    output_height = int(height * scale_final)
    scaled_image = image.resize((output_width, output_height), resample=Image.LANCZOS)

    left = (output_width - target_width) // 2
    top = (output_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    return scaled_image.crop((left, top, right, bottom))


def load_video(data, num_frames, resolution=(560, 336), resize_mode="center_crop", static_scene=False):
    """Load video frames from a video file or image directory or image set.

    Args:
        data: Path to a video file or a directory of images or a list of images.
        num_frames: Number of frames to sample.
        resolution: (width, height) to resize/crop to.
        resize_mode: "center_crop" or "resize".
        static_scene: Whether the scene is static (default: False).
    Returns:
        List of PIL Images.
    """
    def _process_frame(image):
        if resize_mode == "resize":
            return image.resize(resolution, resample=Image.LANCZOS)
        return center_crop(image, resolution)

    if isinstance(data, str) and os.path.isdir(data):
        image_names = sorted(os.listdir(data))
        end_index = min(len(image_names), num_frames)
        if static_scene:
            sample_indices = np.arange(end_index)
        else:
            sample_indices = np.linspace(0, end_index - 1, num_frames, dtype=int)
        images = []
        for idx in sample_indices:
            img_path = os.path.join(data, image_names[idx])
            images.append(_process_frame(Image.open(img_path)))
    elif isinstance(data, list):
        image_paths = sorted(data)
        end_index = min(len(image_paths), num_frames)
        if static_scene:
            sample_indices = np.arange(end_index)
        else:
            sample_indices = np.linspace(0, end_index - 1, num_frames, dtype=int)
        images = []
        for idx in sample_indices:
            images.append(_process_frame(Image.open(image_paths[idx])))
    elif isinstance(data, str) and os.path.isfile(data):
        video_reader = VideoReader(data)
        end_index = min(len(video_reader), num_frames)
        if static_scene:
            sample_indices = np.arange(end_index)
        else:
            sample_indices = np.linspace(0, end_index - 1, num_frames, dtype=int)
        raw_frames = video_reader.get_batch(sample_indices).asnumpy()
        images = [_process_frame(Image.fromarray(f)) for f in raw_frames]
    else:
        raise ValueError(f"Invalid data input: {data} (must be video file, image directory, or list of images)")
    return images


class CameraTrajectory:
    """Unified camera trajectory representation.

    Attributes:
        c2w: Camera-to-world matrices [N, 4, 4]
        mode: "relative" or "global"
        name: Human-readable name
        zoom_ratio: Zoom factor (1.0 = no zoom)
        use_first_frame: Whether to use the first frame as reference
    """

    VALID_TRAJECTORY_TYPES = {
        "pan_left", "pan_right", "tilt_up", "tilt_down",
        "move_left", "move_right", "push_in", "pull_out",
        "boom_up", "boom_down", "orbit_left", "orbit_right",
        "static",
    }

    def __init__(self, c2w, mode="relative", name="trajectory", zoom_ratio=1.0, use_first_frame=False):
        if isinstance(c2w, np.ndarray):
            c2w = torch.from_numpy(c2w)
        assert c2w.ndim == 3 and c2w.shape[1:] == (4, 4), f"c2w must be [N, 4, 4], got {c2w.shape}"
        assert mode in ("relative", "global"), f"mode must be 'relative' or 'global', got '{mode}'"
        self.c2w = c2w.float()
        self.mode = mode
        self.name = name
        self.zoom_ratio = zoom_ratio
        self.use_first_frame = use_first_frame

    @classmethod
    def from_predefined(cls, trajectory_type, num_frames=81, mode="relative",
                        angle=None, distance=None, orbit_radius=None, zoom_ratio=1.0):
        """Create from a predefined trajectory type."""
        default_params = {
            "pan_left": {"angle": 15},
            "pan_right": {"angle": 15},
            "tilt_up": {"angle": 15},
            "tilt_down": {"angle": 15},
            "move_left": {"distance": 0.1},
            "move_right": {"distance": 0.1},
            "push_in": {"distance": 0.1},
            "pull_out": {"distance": 0.1},
            "boom_up": {"distance": 0.1},
            "boom_down": {"distance": 0.1},
            "orbit_left": {"angle": 15, "orbit_radius": 1.0},
            "orbit_right": {"angle": 15, "orbit_radius": 1.0},
            "static": {},
        }
        if trajectory_type not in default_params:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}. "
                             f"Must be one of: {sorted(default_params.keys())}")

        params = default_params[trajectory_type].copy()
        if angle is not None:
            params["angle"] = angle
        if distance is not None:
            params["distance"] = distance
        if orbit_radius is not None:
            params["orbit_radius"] = orbit_radius

        keyframes = [
            {0: [{"static": {}}]},
            {num_frames - 1: [{trajectory_type: params}]},
        ]
        cameras = cls._compose_keyframes(keyframes, num_frames)
        return cls(torch.from_numpy(cameras), mode=mode, name=trajectory_type, zoom_ratio=zoom_ratio, use_first_frame=True)

    @classmethod
    def from_keyframes(cls, keyframes, num_frames=81, mode="relative", name="keyframes", zoom_ratio=1.0, use_first_frame=False):
        """Create from programmatic keyframe list.

        Args:
            keyframes: List of single-key dicts {frame_index: [operations]}, where
                       each operation is a single-key dict {traj_type: params_dict}.
                       Example: [{0: [{"static": {}}]}, {80: [{"pan_left": {"angle": 20}}]}]
            num_frames: Number of output frames.
            mode: "relative" or "global".
            name: Human-readable name.
            use_first_frame: Whether to use the first frame as a reference.
        """
        cameras = cls._compose_keyframes(keyframes, num_frames)
        return cls(torch.from_numpy(cameras), mode=mode, name=name, zoom_ratio=zoom_ratio, use_first_frame=use_first_frame)

    @classmethod
    def from_json(cls, filepath):
        """Load trajectory from a JSON file (keyframe or matrix format)."""
        data = cls.validate_json(filepath)
        num_frames = data.get("num_frames", 81)
        mode = data.get("mode", "relative")
        name = data.get("name", os.path.splitext(os.path.basename(filepath))[0])
        zoom_ratio = data.get("zoom_ratio", 1.0)
        use_first_frame = data.get("use_first_frame", False)
        if "keyframes" in data:
            keyframes_list = []
            for kf in data["keyframes"]:
                ts_str, operations = next(iter(kf.items()))
                frame_idx = int(ts_str)
                keyframes_list.append({frame_idx: operations})
            cameras = cls._compose_keyframes(keyframes_list, num_frames)
        else:
            traj = data["trajectory"]
            if "file" in traj:
                npz_path = traj["file"]
                npz_data = np.load(npz_path)
                frame_indices = npz_data["frame_indices"]
                frame_matrices = npz_data["frame_matrices"]
            else:
                frame_indices = traj["frame_indices"]
                frame_matrices = traj["frame_matrices"]
            cameras = cls._interpolate_sparse_matrices(frame_indices, frame_matrices, num_frames)

        return cls(torch.from_numpy(cameras), mode=mode, name=name, zoom_ratio=zoom_ratio, use_first_frame=use_first_frame)

    def __len__(self):
        return self.c2w.shape[0]

    @staticmethod
    def validate_json(filepath):
        """Validate a trajectory JSON file and return parsed data."""
        if not os.path.exists(filepath):
            raise ValueError(f"Trajectory file not found: {filepath}")
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

        mode = data.get("mode", "relative")
        if mode not in ("relative", "global"):
            raise ValueError(f"Invalid mode: {mode} (must be 'relative' or 'global')")

        if "keyframes" in data:
            CameraTrajectory._validate_keyframe_format(data)
        elif "trajectory" in data:
            CameraTrajectory._validate_matrix_format(data)
        else:
            raise ValueError("Must contain either 'keyframes' or 'trajectory' field")

        return data

    @staticmethod
    def _get_camera_matrix(trajectory_type, distance=0.1, orbit_radius=0.5, angle=30):
        angle_rad = np.deg2rad(angle)
        camera_pose = np.eye(4, dtype=np.float32)

        if trajectory_type == "move_left":
            camera_pose[0, 3] = -distance
        elif trajectory_type == "move_right":
            camera_pose[0, 3] = distance
        elif trajectory_type == "push_in":
            camera_pose[2, 3] = distance
        elif trajectory_type == "pull_out":
            camera_pose[2, 3] = -distance
        elif trajectory_type == "boom_up":
            camera_pose[1, 3] = -distance
        elif trajectory_type == "boom_down":
            camera_pose[1, 3] = distance
        elif trajectory_type == "orbit_left":
            theta = -angle_rad
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            camera_pose[0, 3] = orbit_radius * sin_a
            camera_pose[2, 3] = orbit_radius * (1 - cos_a)
            camera_pose[0, 0] = cos_a
            camera_pose[0, 2] = -sin_a
            camera_pose[2, 0] = sin_a
            camera_pose[2, 2] = cos_a
        elif trajectory_type == "orbit_right":
            theta = angle_rad
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            camera_pose[0, 3] = orbit_radius * sin_a
            camera_pose[2, 3] = orbit_radius * (1 - cos_a)
            camera_pose[0, 0] = cos_a
            camera_pose[0, 2] = -sin_a
            camera_pose[2, 0] = sin_a
            camera_pose[2, 2] = cos_a
        elif trajectory_type == "pan_left":
            theta = -angle_rad
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            camera_pose[0, 0] = cos_a
            camera_pose[0, 2] = sin_a
            camera_pose[2, 0] = -sin_a
            camera_pose[2, 2] = cos_a
        elif trajectory_type == "pan_right":
            theta = angle_rad
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            camera_pose[0, 0] = cos_a
            camera_pose[0, 2] = sin_a
            camera_pose[2, 0] = -sin_a
            camera_pose[2, 2] = cos_a
        elif trajectory_type == "tilt_up":
            theta = angle_rad
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            camera_pose[1, 1] = cos_a
            camera_pose[1, 2] = -sin_a
            camera_pose[2, 1] = sin_a
            camera_pose[2, 2] = cos_a
        elif trajectory_type == "tilt_down":
            theta = -angle_rad
            cos_a, sin_a = np.cos(theta), np.sin(theta)
            camera_pose[1, 1] = cos_a
            camera_pose[1, 2] = -sin_a
            camera_pose[2, 1] = sin_a
            camera_pose[2, 2] = cos_a
        elif trajectory_type in ("static",):
            pass

        return camera_pose

    @staticmethod
    def _compose_keyframes(keyframes, num_frames):
        """Compose keyframes into a full camera trajectory.

        Args:
            keyframes: List of single-key dicts {frame_index: [operations]}.
                       Frame indices are integers in [0, num_frames-1].
            num_frames: Number of output frames.
        """
        frame_indices = [next(iter(kf)) for kf in keyframes]
        assert frame_indices == sorted(frame_indices), \
            f"Keyframe frame indices must be in ascending order, got {frame_indices}"

        keyframe_times = []
        keyframe_poses = []
        accumulated_pose = np.eye(4, dtype=np.float32)

        max_idx = num_frames - 1 if num_frames > 1 else 1
        for kf in keyframes:
            frame_idx, operations = next(iter(kf.items()))
            delta_pose = np.eye(4, dtype=np.float32)
            for op in operations:
                traj_type, params = next(iter(op.items()))
                distance = params.get("distance", 0.1)
                angle = params.get("angle", 30)
                orbit_radius = params.get("orbit_radius", 0.5)
                delta_pose = delta_pose @ CameraTrajectory._get_camera_matrix(
                    traj_type, distance, orbit_radius, angle)
            accumulated_pose = accumulated_pose @ delta_pose
            keyframe_times.append(frame_idx / max_idx)
            keyframe_poses.append(accumulated_pose.copy())

        keyframe_rotations = Rotation.from_matrix([pose[:3, :3] for pose in keyframe_poses])
        keyframe_positions = np.array([pose[:3, 3] for pose in keyframe_poses])

        slerp = Slerp(keyframe_times, keyframe_rotations)
        frame_times = np.linspace(0, 1, num_frames)
        rot_interp = slerp(frame_times)
        pos_interp = np.array([
            np.interp(frame_times, keyframe_times, keyframe_positions[:, i])
            for i in range(3)
        ]).T

        cameras = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
        cameras[:, :3, :3] = rot_interp.as_matrix()
        cameras[:, :3, 3] = pos_interp
        return cameras

    @staticmethod
    def _interpolate_sparse_matrices(indices, matrices, num_frames):
        keyframe_indices = np.array(indices)
        keyframe_matrices = np.array(matrices, dtype=np.float32)
        if len(indices) == num_frames:
            return keyframe_matrices

        keyframe_rotations = Rotation.from_matrix(keyframe_matrices[:, :3, :3])
        keyframe_positions = keyframe_matrices[:, :3, 3]

        max_frame = keyframe_indices[-1]
        if max_frame == 0:
            max_frame = num_frames - 1
        keyframe_times = keyframe_indices / max_frame

        slerp = Slerp(keyframe_times, keyframe_rotations)
        frame_times = np.linspace(0, 1, num_frames)
        rot_interp = slerp(frame_times)
        pos_interp = np.array([
            np.interp(frame_times, keyframe_times, keyframe_positions[:, i])
            for i in range(3)
        ]).T

        cameras = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
        cameras[:, :3, :3] = rot_interp.as_matrix()
        cameras[:, :3, 3] = pos_interp
        return cameras

    @staticmethod
    def _validate_keyframe_format(data):
        keyframes = data["keyframes"]
        num_frames = data.get("num_frames", 81)
        if not isinstance(keyframes, list) or len(keyframes) < 2:
            raise ValueError("keyframes must be a list with at least 2 keyframes")

        frame_indices = []
        for i, kf in enumerate(keyframes):
            if not isinstance(kf, dict) or len(kf) != 1:
                raise ValueError(f"Keyframe {i}: must be a single-key dict {{frame_index: [operations]}}")
            key_str, operations = next(iter(kf.items()))
            try:
                idx = int(key_str)
            except (TypeError, ValueError):
                raise ValueError(f"Keyframe {i}: key must be an integer frame index, got '{key_str}'")
            if idx < 0 or idx >= num_frames:
                raise ValueError(f"Keyframe {i}: frame index must be >= 0 and < {num_frames}, got {idx}")
            frame_indices.append(idx)

            if not isinstance(operations, list) or len(operations) == 0:
                raise ValueError(f"Keyframe {i}: operations must be non-empty list")

            for j, op in enumerate(operations):
                if not isinstance(op, dict) or len(op) != 1:
                    raise ValueError(f"Keyframe {i}, operation {j}: must be a single-key dict {{type: params}}")
                op_type, params = next(iter(op.items()))
                if op_type not in CameraTrajectory.VALID_TRAJECTORY_TYPES:
                    raise ValueError(f"Keyframe {i}, operation {j}: invalid type '{op_type}'")
                if not isinstance(params, dict):
                    raise ValueError(f"Keyframe {i}, operation {j}: params must be a dict")

        if frame_indices != sorted(frame_indices):
            raise ValueError("Keyframe frame indices must be in ascending order")
        if frame_indices[0] != 0:
            raise ValueError("First keyframe frame index must be 0")
        if frame_indices[-1] != num_frames - 1:
            raise ValueError(f"Last keyframe frame index must be {num_frames - 1}")

    @staticmethod
    def _validate_matrix_format(data):
        trajectory = data["trajectory"]
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be a dict")

        if "file" in trajectory:
            file_path = trajectory["file"]
            if not isinstance(file_path, str):
                raise ValueError("trajectory.file must be a string path")
            return

        if "frame_indices" not in trajectory:
            raise ValueError("trajectory: missing 'frame_indices'")
        if "frame_matrices" not in trajectory:
            raise ValueError("trajectory: missing 'frame_matrices'")

        frame_indices = trajectory["frame_indices"]
        frame_matrices = trajectory["frame_matrices"]

        if not isinstance(frame_indices, list) or len(frame_indices) < 2:
            raise ValueError("trajectory.frame_indices must be a list with at least 2 entries")
        if not isinstance(frame_matrices, list) or len(frame_matrices) != len(frame_indices):
            raise ValueError(f"trajectory.frame_matrices length ({len(frame_matrices)}) "
                             f"must match frame_indices length ({len(frame_indices)})")

        for i, frame in enumerate(frame_indices):
            if not isinstance(frame, int) or frame < 0:
                raise ValueError(f"trajectory.frame_indices[{i}]: must be non-negative integer, got {frame}")

        for i, matrix in enumerate(frame_matrices):
            if not isinstance(matrix, list) or len(matrix) != 4:
                raise ValueError(f"trajectory.frame_matrices[{i}]: must be 4x4 array")
            for j, row in enumerate(matrix):
                if not isinstance(row, list) or len(row) != 4:
                    raise ValueError(f"trajectory.frame_matrices[{i}][{j}]: must have 4 elements")
                for k, val in enumerate(row):
                    if not isinstance(val, (int, float)):
                        raise ValueError(f"trajectory.frame_matrices[{i}][{j}][{k}]: must be numeric")

        if frame_indices != sorted(frame_indices):
            raise ValueError("trajectory.frame_indices must be in ascending order")
