import logging
import os
from collections import namedtuple
from itertools import accumulate
from typing import Optional, Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

DEFAULT_TRANSITIONS = (15, 6, 4, 11, 13, 6)

logger = logging.getLogger("STORM")
turbo_cmap = cm.get_cmap("turbo")

depth_visualizer = lambda frame, opacity: visualize_depth(
    frame,
    opacity,
    lo=4.0,
    hi=120,
    depth_curve_fn=lambda x: -np.log(x + 1e-6),
)

flow_visualizer = (
    lambda frame: scene_flow_to_rgb(
        frame,
        background="bright",
        flow_max_radius=1.0,
    )
    .cpu()
    .numpy()
)


def to8b(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def sinebow(h):
    """A cyclic and uniform colormap, see http://basecase.org/env/on-rainbows."""
    f = lambda x: np.sin(np.pi * x) ** 2
    return np.stack([f(3 / 6 - h), f(5 / 6 - h), f(7 / 6 - h)], -1)


def matte(vis, acc, dark=0.8, light=1.0, width=8):
    """Set non-accumulated pixels to a Photoshop-esque checker pattern."""
    bg_mask = np.logical_xor(
        (np.arange(acc.shape[0]) % (2 * width) // width)[:, None],
        (np.arange(acc.shape[1]) % (2 * width) // width)[None, :],
    )
    bg = np.where(bg_mask, light, dark)
    return vis * acc[:, :, None] + (bg * (1 - acc))[:, :, None]


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def visualize_cmap(
    value,
    weight,
    colormap,
    lo=None,
    hi=None,
    percentile=99.0,
    curve_fn=lambda x: x,
    modulus=None,
    matte_background=True,
):
    """Visualize a 1D image and a 1D weighting according to some colormap.
    from mipnerf

    Args:
      value: A 1D image.
      weight: A weight map, in [0, 1].
      colormap: A colormap function.
      lo: The lower bound to use when rendering, if None then use a percentile.
      hi: The upper bound to use when rendering, if None then use a percentile.
      percentile: What percentile of the value map to crop to when automatically
        generating `lo` and `hi`. Depends on `weight` as well as `value'.
      curve_fn: A curve function that gets applied to `value`, `lo`, and `hi`
        before the rest of visualization. Good choices: x, 1/(x+eps), log(x+eps).
      modulus: If not None, mod the normalized value by `modulus`. Use (0, 1]. If
        `modulus` is not None, `lo`, `hi` and `percentile` will have no effect.
      matte_background: If True, matte the image over a checkerboard.

    Returns:
      A colormap rendering.
    """
    # Identify the values that bound the middle of `value' according to `weight`.
    if lo is None or hi is None:
        lo_auto, hi_auto = weighted_percentile(
            value, weight, [50 - percentile / 2, 50 + percentile / 2]
        )
        # If `lo` or `hi` are None, use the automatically-computed bounds above.
        eps = np.finfo(np.float32).eps
        lo = lo or (lo_auto - eps)
        hi = hi or (hi_auto + eps)

    # Curve all values.
    value, lo, hi = [curve_fn(x) for x in [value, lo, hi]]

    # Wrap the values around if requested.
    if modulus:
        value = np.mod(value, modulus) / modulus
    else:
        # Otherwise, just scale to [0, 1].
        value = np.nan_to_num(np.clip((value - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    if weight is not None:
        value *= weight
    else:
        weight = np.ones_like(value)
    if colormap:
        colorized = colormap(value)[..., :3]
    else:
        assert len(value.shape) == 3 and value.shape[-1] == 3
        colorized = value

    return matte(colorized, weight) if matte_background else colorized


def visualize_depth(x, acc=None, lo=None, hi=None, depth_curve_fn=lambda x: -np.log(x + 1e-6)):
    """Visualizes depth maps."""
    return visualize_cmap(
        x,
        acc,
        cm.get_cmap("turbo"),
        curve_fn=depth_curve_fn,
        lo=lo,
        hi=hi,
        matte_background=False,
    )


def _make_colorwheel(
    transitions: tuple = DEFAULT_TRANSITIONS, backend="torch"
) -> Union[np.ndarray, torch.Tensor]:
    """Creates a colorwheel (borrowed/modified from flowpy).
    A colorwheel defines the transitions between the six primary hues:
    Red(255, 0, 0), Yellow(255, 255, 0), Green(0, 255, 0), Cyan(0, 255, 255), Blue(0, 0, 255) and Magenta(255, 0, 255).
    Args:
        transitions: Contains the length of the six transitions, based on human color perception.
    Returns:
        colorwheel: The RGB values of the transitions in the color space.
    Notes:
        For more information, see:
        https://web.archive.org/web/20051107102013/http://members.shaw.ca/quadibloc/other/colint.htm
        http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    """
    colorwheel_length = sum(transitions)
    # The red hue is repeated to make the colorwheel cyclic
    base_hues = map(
        np.array,
        (
            [255, 0, 0],
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255],
            [0, 0, 255],
            [255, 0, 255],
            [255, 0, 0],
        ),
    )
    colorwheel = np.zeros((colorwheel_length, 3), dtype="uint8")
    hue_from = next(base_hues)
    start_index = 0
    for hue_to, end_index in zip(base_hues, accumulate(transitions)):
        transition_length = end_index - start_index
        colorwheel[start_index:end_index] = np.linspace(
            hue_from, hue_to, transition_length, endpoint=False
        )
        hue_from = hue_to
        start_index = end_index
    if backend == "torch":
        return torch.FloatTensor(colorwheel)
    else:
        return colorwheel


WHEEL = _make_colorwheel()
N_COLS = len(WHEEL)
WHEEL = torch.vstack((WHEEL, WHEEL[0]))  # Make the wheel cyclic for interpolation


def scene_flow_to_rgb(
    flow: torch.Tensor,
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "bright",
) -> Union[torch.Tensor, np.ndarray]:
    """Creates a RGB representation of an optical flow (borrowed/modified from flowpy).
    Adapted from https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior/blob/main/visualize.py
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity, useful for comparing different flows.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow should look 'bright' or 'dark'.
    Returns: An array of RGB colors.
    """
    valid_backgrounds = ("bright", "dark")
    if background not in valid_backgrounds:
        raise ValueError(
            f"background should be one the following: {valid_backgrounds}, not {background}."
        )
    if isinstance(flow, np.ndarray):
        backend = "np"
        op = np
    else:
        backend = "torch"
        op = torch

    # For scene flow, it's reasonable to assume displacements in x and y directions only for visualization pursposes.
    complex_flow = flow[..., 0] + 1j * flow[..., 1]
    radius, angle = op.abs(complex_flow), op.angle(complex_flow)
    if flow_max_radius is None:
        # flow_max_radius = torch.max(radius)
        flow_max_radius = op.quantile(radius, 0.99)
    if flow_max_radius > 0:
        radius /= flow_max_radius
    # Map the angles from (-pi, pi] to [0, 2pi) to [0, ncols - 1)
    wheel = _make_colorwheel(backend=backend)
    n_cols = len(wheel)
    wheel = op.vstack((wheel, wheel[0]))
    angle[angle < 0] += 2 * np.pi
    angle = angle * ((n_cols - 1) / (2 * np.pi))

    # Interpolate the hues
    angle_fractional, angle_floor, angle_ceil = (
        op.fmod(angle, 1),
        op.trunc(angle),
        op.ceil(angle),
    )
    angle_fractional = angle_fractional[..., None]
    if backend == "torch":
        _wheel = wheel.to(angle_floor.device)
        float_hue = (
            _wheel[angle_floor.long()] * (1 - angle_fractional)
            + _wheel[angle_ceil.long()] * angle_fractional
        )
    else:
        float_hue = (
            wheel[angle_floor.astype(op.int64)] * (1 - angle_fractional)
            + wheel[angle_ceil.astype(op.int64)] * angle_fractional
        )
    ColorizationArgs = namedtuple(
        "ColorizationArgs",
        ["move_hue_valid_radius", "move_hue_oversized_radius", "invalid_color"],
    )

    def move_hue_on_V_axis(hues, factors):
        return hues * factors[..., None]

    def move_hue_on_S_axis(hues, factors):
        return 255.0 - factors[..., None] * (255.0 - hues)

    if background == "dark":
        parameters = ColorizationArgs(
            move_hue_on_V_axis, move_hue_on_S_axis, op.array([255.0, 255.0, 255.0])
        )
    else:
        parameters = ColorizationArgs(move_hue_on_S_axis, move_hue_on_V_axis, op.zeros(3))
    colors = parameters.move_hue_valid_radius(float_hue, radius)
    oversized_radius_mask = radius > 1
    colors[oversized_radius_mask] = parameters.move_hue_oversized_radius(
        float_hue[oversized_radius_mask], 1 / radius[oversized_radius_mask]
    )
    colors = colors / 255.0
    return colors


def get_robust_pca(features: Tensor, m: float = 2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)


def get_pca_map(
    feature_map: torch.Tensor,
    img_size,
    interpolation="nearest",
    return_pca_stats=False,
    pca_stats=None,
):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if len(feature_map.shape) != 4:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    batch_size = feature_map.shape[0]
    if pca_stats is None:
        pca_stats = []
        for i in range(batch_size):
            reduct_mat, color_min, color_max = get_robust_pca(
                feature_map[i].reshape(-1, feature_map.shape[-1])
            )
            pca_stats.append((reduct_mat, color_min, color_max))
    pca_colors = []
    for i in range(batch_size):
        reduct_mat, color_min, color_max = pca_stats[i]
        pca_color = feature_map[i] @ reduct_mat
        pca_color = (pca_color - color_min) / (color_max - color_min)
        pca_color = pca_color.clamp(0, 1)
        pca_color = F.interpolate(
            pca_color.permute(2, 0, 1)[None],
            size=img_size,
            mode=interpolation,
        )
        pca_color = pca_color.permute(0, 2, 3, 1)
        pca_colors.append(pca_color)
    pca_color = torch.cat(pca_colors, dim=0)
    if return_pca_stats:
        return pca_color.cpu(), pca_stats
    return pca_color.cpu()


def get_scale_map(
    scalar_map: torch.Tensor,
    img_size,
    interpolation="nearest",
):
    """
    scalar_map: (1, h, w, C) is the feature map of a single image.
    """
    if len(scalar_map.shape) != 1:
        scalar_map = scalar_map[None]
    scalar_map = (scalar_map - scalar_map.min(dim=-1).values) / (
        scalar_map.max(dim=-1).values - scalar_map.min(dim=-1).values + 1e-6
    )
    scalar_map = F.interpolate(
        scalar_map.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    # cmap = plt.get_cmap("viridis")
    # scalar_map = cmap(scalar_map)[..., :3]
    # make it 3 channels
    scalar_map = torch.cat([scalar_map] * 3, dim=-1)
    return scalar_map.cpu()


def get_similarity_map(features: Tensor, img_size=(224, 224)):
    """
    compute the similarity map of the central patch to the rest of the image
    """
    assert len(features.shape) == 4, "features should be (1, C, H, W)"
    H, W, C = features.shape[1:]
    center_patch_feature = features[0, H // 2, W // 2, :]
    center_patch_feature_normalized = center_patch_feature / center_patch_feature.norm()
    center_patch_feature_normalized = center_patch_feature_normalized.unsqueeze(1)
    # Reshape and normalize the entire feature tensor
    features_flat = features.view(-1, C)
    features_normalized = features_flat / features_flat.norm(dim=1, keepdim=True)

    similarity_map_flat = features_normalized @ center_patch_feature_normalized
    # Reshape the flat similarity map back to the spatial dimensions (H, W)
    similarity_map = similarity_map_flat.view(H, W)

    # Normalize the similarity map to be in the range [0, 1] for visualization
    similarity_map = (similarity_map - similarity_map.min()) / (
        similarity_map.max() - similarity_map.min()
    )
    # we don't want the center patch to be the most similar
    similarity_map[H // 2, W // 2] = -1.0
    similarity_map = (
        F.interpolate(
            similarity_map.unsqueeze(0).unsqueeze(0),
            size=img_size,
            mode="bilinear",
        )
        .squeeze(0)
        .squeeze(0)
    )

    similarity_map_np = similarity_map.cpu().numpy()
    negative_mask = similarity_map_np < 0

    colormap = plt.get_cmap("turbo")

    # Apply the colormap directly to the normalized similarity map and multiply by 255 to get RGB values
    similarity_map_rgb = colormap(similarity_map_np)[..., :3]
    similarity_map_rgb[negative_mask] = [1.0, 0.0, 0.0]
    return similarity_map_rgb.cpu()