import numpy as np
import trimesh
from matplotlib import colormaps
from ..auxiliary_models.worldmirror.utils.visual_util import integrate_camera_into_scene


def extract_point_cloud(predictions):
    """Extract point cloud from Gaussian splats.

    Args:
        predictions: dict from pipe.reconstructor(), containing:
            - "splats": List[List[Gaussians]]
              Each Gaussians object has:
                .means       torch.Tensor [N, 3]   — 3D positions
                .harmonics   torch.Tensor [N, K*3]  — SH coefficients
                .opacities   torch.Tensor [N]       — opacity values
                .confidences torch.Tensor [N]       — confidence scores
                .timestamp   int                    — frame index

    Returns:
        points:        np.ndarray [M, 3]  float32
        colors:        np.ndarray [M, 3]  uint8
        frame_indices: np.ndarray [M]     int
    """
    all_points = []
    all_colors = []
    all_frame_indices = []

    splats_batch = predictions["splats"][0]  # first batch
    for gs in splats_batch:
        means = gs.means.detach().cpu().float()          # [N, 3]
        harmonics = gs.harmonics.detach().cpu().float()  # [N, K*3]
        opacities = gs.opacities.detach().cpu().float()  # [N]

        # Filter by opacity
        mask = opacities > 0.05
        if gs.confidences is not None:
            conf = gs.confidences.detach().cpu().float()
            mask = mask & (conf > 0.0)

        means = means[mask]
        harmonics = harmonics[mask]

        if means.shape[0] == 0:
            continue

        # SH DC → RGB: harmonics is [N, 3, K], DC component is [:, :, 0]
        C0 = 0.28209479177387814
        dc = harmonics[:, 0] if harmonics.ndim == 3 else harmonics[:, :3]
        rgb = (dc * C0 + 0.5).clamp(0.0, 1.0) * 255.0
        rgb = rgb.numpy().astype(np.uint8)

        all_points.append(means.numpy())
        all_colors.append(rgb)
        all_frame_indices.append(np.full(means.shape[0], gs.timestamp if gs.timestamp >= 0 else 0, dtype=np.int32))

    if len(all_points) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.int32)

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.uint8)
    frame_indices = np.concatenate(all_frame_indices, axis=0)
    return points, colors, frame_indices


def build_scene_glb(points, colors, frame_indices, cam2world,
                    selected_idx=None, vis_frame_num=11, camera_size=1.0):
    """Build a trimesh.Scene with point cloud and camera frustums.

    Returns the trimesh.Scene object (caller is responsible for exporting).
    """
    scene = trimesh.Scene()
    S = cam2world.shape[0]

    # --- Point cloud ---
    if points.shape[0] > 0:
        if vis_frame_num and vis_frame_num > 0:
            unique_frames = np.unique(frame_indices)
            chosen = unique_frames[np.linspace(0, len(unique_frames) - 1, vis_frame_num, dtype=int)]
            mask = np.isin(frame_indices, chosen)
            pts, cols = points[mask], colors[mask]
        else:
            pts, cols = points, colors

        if pts.shape[0] > 0:
            # Subsample if too many points for viewer performance
            max_pts = 500_000
            if pts.shape[0] > max_pts:
                indices = np.random.choice(pts.shape[0], max_pts, replace=False)
                pts = pts[indices]
                cols = cols[indices]

            rgba = np.hstack([cols, np.full((cols.shape[0], 1), 255, dtype=np.uint8)])
            pc = trimesh.PointCloud(vertices=pts, colors=rgba)
            scene.add_geometry(pc)

    # --- Camera frustums ---
    cmap = colormaps.get_cmap("gist_rainbow")
    for i in range(S):
        c2w = cam2world[i]
        cam_color = tuple(int(255 * x) for x in cmap(i / S)[:3])

        if selected_idx is not None and i == selected_idx:
            cur_cam_size = camera_size * 2
        else:
            cur_cam_size = camera_size

        integrate_camera_into_scene(scene, c2w, cam_color, cur_cam_size)

    # OpenCV→OpenGL flip: Y and Z
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    scene_transformation = np.linalg.inv(cam2world[0]) @ flip
    scene.apply_transform(scene_transformation)

    return scene
