import torch


def homo_matrix_inverse(homo_matrix):
    """
    Computes the inverse of a batch of 4x4 (or 3x4) homogeneous transformation matrices.
    """
    assert homo_matrix.shape[-2:] == (4, 4) or homo_matrix.shape[-2:] == (3, 4), "Input must be a batch of 4x4 or 3x4 matrices"

    R, T = homo_matrix[..., :3, :3].reshape(-1, 3, 3), homo_matrix[..., :3, 3:4].reshape(-1, 3, 1)

    with torch.cuda.amp.autocast(enabled=False):
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
    depth_map_padded = torch.nn.functional.pad(depth_map, (padding, padding, padding, padding), mode='replicate')

    # Apply convolution
    smoothed_depth = torch.nn.functional.conv2d(depth_map_padded, kernel, padding=0)

    return smoothed_depth.squeeze(0).squeeze(0)


def fast_perceptual_color_distance(color1, color2):
    """
    Fast RGB perceptual color distance approximation.
    Based on the formula you provided which accounts for human visual sensitivity.

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
