# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import PIL
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC


class ImageList:
    """Convenience class to apply the same operation to a whole set of images."""

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch("resize", *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch("crop", *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def rescale_image_depthmap(
    image, depthmap, camera_intrinsics, output_resolution, force=True
):
    """Jointly rescale a (image, depthmap)
    so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(
        output_resolution, resample=lanczos if scale_final < 1 else bicubic
    )
    if depthmap is not None:
        depthmap = cv2.resize(
            depthmap,
            output_resolution,
            fx=scale_final,
            fy=scale_final,
            interpolation=cv2.INTER_NEAREST,
        )

    # no offset here; simple rescaling
    if camera_intrinsics is not None:
        camera_intrinsics = camera_matrix_of_crop(
            camera_intrinsics, input_resolution, output_resolution, scaling=scale_final
        )

    return image.to_pil(), depthmap, camera_intrinsics


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def camera_matrix_of_crop(
    input_camera_matrix,
    input_resolution,
    output_resolution,
    scaling=1,
    offset_factor=0.5,
    offset=None,
):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix


def crop_image_depthmap(image, depthmap, camera_intrinsics, crop_bbox):
    """
    Return a crop of the input view.
    """
    image = ImageList(image)
    l, t, r, b = crop_bbox

    image = image.crop((l, t, r, b))

    if depthmap is not None:
        depthmap = depthmap[t:b, l:r]

    if camera_intrinsics is not None:
        camera_intrinsics = camera_intrinsics.copy()
        camera_intrinsics[0, 2] -= l
        camera_intrinsics[1, 2] -= t

    return image.to_pil(), depthmap, camera_intrinsics


def bbox_from_intrinsics_in_out(
    input_camera_matrix, output_camera_matrix, output_resolution
):
    out_width, out_height = output_resolution
    l, t = np.int32(np.round(input_camera_matrix[:2, 2] - output_camera_matrix[:2, 2]))
    crop_bbox = (l, t, l + out_width, t + out_height)
    return crop_bbox


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """Open an image or a depthmap with opencv-python."""
    if path.endswith((".exr", "EXR")):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f"Could not load image={path} with {options=}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def homo_matrix_inverse(homo_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of a batch of 4x4 (or 3x4) homogeneous transformation matrices.
    returning a batch of 4x4 inverses.
    """
    # last two dims must be 4x4 or 3x4
    assert homo_matrix.shape[-2:] in ((4, 4), (3, 4)), "Input must be a batch of 4x4 or 3x4 matrices"

    R, T = homo_matrix[..., :3, :3].reshape(-1, 3, 3), homo_matrix[..., :3, 3:4].reshape(-1, 3, 1)

    # invert R and T
    R_inv = np.swapaxes(R, -1, -2)            # (B,3,3)
    T_inv = -np.matmul(R_inv, T)              # (B,3,1)

    inv_mats = np.tile(
        np.eye(4, dtype=homo_matrix.dtype)[None], (R_inv.shape[0], 1, 1)
    )                                         # (B,4,4)
    inv_mats[:, :3, :3] = R_inv
    inv_mats[:, :3, 3:4] = T_inv

    # reshape back to original batch dims, with 4x4 at the end
    inv_mats = inv_mats.reshape(*homo_matrix.shape[:-2], 4, 4)
    return inv_mats


def homo_matrix_multiply(homo_matrix1: np.ndarray, homo_matrix2: np.ndarray) -> np.ndarray:
    """
    Computes the multiplication of two batches of 4x4 (or 3x4) homogeneous transformation matrices,
    returning a batch of 4x4 homogeneous transformation matrices.
    """
    assert homo_matrix1.shape[-2:] in ((4, 4), (3, 4)), "Input must be a batch of 4x4 or 3x4 matrices"
    assert homo_matrix2.shape[-2:] in ((4, 4), (3, 4)), "Input must be a batch of 4x4 or 3x4 matrices"
    assert homo_matrix1.shape[:-2] == homo_matrix2.shape[:-2], "Input batches must have the same batch size"

    R1, T1 = homo_matrix1[..., :3, :3].reshape(-1, 3, 3), homo_matrix1[..., :3, 3:4].reshape(-1, 3, 1)
    R2, T2 = homo_matrix2[..., :3, :3].reshape(-1, 3, 3), homo_matrix2[..., :3, 3:4].reshape(-1, 3, 1)

    R_out = np.matmul(R1, R2)
    T_out = np.matmul(R1, T2) + T1

    homo_out = np.tile(
        np.eye(4, dtype=homo_matrix1.dtype)[None], (R_out.shape[0], 1, 1)
    )
    homo_out[:, :3, :3] = R_out
    homo_out[:, :3, 3:4] = T_out

    # reshape back to original batch dims, with 4x4 at the end
    homo_out = homo_out.reshape(*homo_matrix1.shape[:-2], 4, 4)
    return homo_out


def homo_matrix_multiply_points(homo_matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 (or 3x4) homogeneous transformation matrix to a batch of 3D points.

    Args:
        homo_matrix (np.ndarray): A 4x4 (or 3x4) homogeneous transformation matrix.
        points (np.ndarray): A batch of 3D points with shape (B, 3) or (B, 4).

    Returns:
        np.ndarray: The transformed points with shape (B, 3).
    """
    assert homo_matrix.shape in ((4, 4), (3, 4)), "Input must be a batch of 4x4 or 3x4 matrices"
    assert points.shape[-1] in (3, 4), "Points must be 3D or homogeneous 4D"

    temp_points = points[:, :3].T
    R, T = homo_matrix[:3, :3], homo_matrix[:3, 3:4]
    out = np.matmul(R, temp_points) + T
    return out.T


def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    """
    Convert 4-dimensional quaternions to 3x3 rotation matrices.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    """

    # Order changed to match scipy format: (i, j, k, r)
    i, j, k, r = quaternions[..., 0], quaternions[..., 1], quaternions[..., 2], quaternions[..., 3]
    two_s = 2 / ((quaternions * quaternions).sum(axis=-1) + eps)

    # Construct rotation matrix elements using quaternion algebra
    o = np.stack(
        (
            1 - two_s * (j * j + k * k),  # R[0,0]
            two_s * (i * j - k * r),  # R[0,1]
            two_s * (i * k + j * r),  # R[0,2]
            two_s * (i * j + k * r),  # R[1,0]
            1 - two_s * (i * i + k * k),  # R[1,1]
            two_s * (j * k - i * r),  # R[1,2]
            two_s * (i * k - j * r),  # R[2,0]
            two_s * (j * k + i * r),  # R[2,1]
            1 - two_s * (i * i + j * j),  # R[2,2]
        ),
        axis=-1,
    )
    return o.reshape(*quaternions.shape[:-1], 3, 3)
