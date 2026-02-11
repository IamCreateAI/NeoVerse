from typing import Dict, Tuple, Optional
from jaxtyping import Float
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_sum
from einops import rearrange

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from ..utils.frustum import calculate_unprojected_mask
from ..utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3, project_world_points_to_image
from ..utils import sh_utils, act_gs


class Gaussians:
    def __init__(
        self,
        means: Float[Tensor, "*batch 3"],
        harmonics: Float[Tensor, "*batch 3 _"],
        opacities: Float[Tensor, " *batch"],
        scales: Float[Tensor, "*batch 3"],
        rotations: Float[Tensor, "*batch 4"],
        confidences: Optional[Float[Tensor, "*batch"]] = None,
        timestamp: int = 0,
        life_span: Float[Tensor, "*batch"] | float = 1.0,
        life_span_gamma: float = 0.0,
        forward_timestamp: Optional[int] = None,
        forward_vel: Optional[Float[Tensor, "*batch 3"]] = None,
        forward_scales: Optional[Float[Tensor, "*batch 3"]] = None,
        forward_rotations: Optional[Float[Tensor, "*batch 3"]] = None,
        backward_timestamp: Optional[int] = None,
        backward_vel: Optional[Float[Tensor, "*batch 3"]] = None,
        backward_scales: Optional[Float[Tensor, "*batch 3"]] = None,
        backward_rotations: Optional[Float[Tensor, "*batch 3"]] = None,
    ):
        self.means = means
        self.harmonics = harmonics
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations
        self.confidences = confidences
        self.timestamp = timestamp
        self.life_span = life_span
        self.life_span_gamma = life_span_gamma

        if forward_timestamp is not None:
            assert forward_timestamp >= timestamp, "Forward timestamp must be greater than or equal to current timestamp."
            self.forward_timestamp = forward_timestamp
        else:
            self.forward_timestamp = None

        if forward_vel is not None:
            assert forward_timestamp is not None, "Forward velocity must be provided with forward timestamp."
            self.forward_vel = forward_vel
        else:
            self.forward_vel = None

        if forward_scales is not None:
            assert forward_timestamp is not None, "Forward scales must be provided with forward timestamp."
            self.forward_scales = forward_scales
        else:
            self.forward_scales = None

        if forward_rotations is not None:
            assert forward_timestamp is not None, "Forward rotations must be provided with forward timestamp."
            self.forward_rotations = forward_rotations
        else:
            self.forward_rotations = None

        if backward_timestamp is not None:
            assert backward_timestamp <= timestamp, "Backward timestamp must be less than or equal to current timestamp."
            self.backward_timestamp = backward_timestamp
        else:
            self.backward_timestamp = None

        if backward_vel is not None:
            assert backward_timestamp is not None, "Backward velocity must be provided with backward timestamp."
            self.backward_vel = backward_vel
        else:
            self.backward_vel = None

        if backward_scales is not None:
            assert backward_timestamp is not None, "Backward scales must be provided with backward timestamp."
            self.backward_scales = backward_scales
        else:
            self.backward_scales = None

        if backward_rotations is not None:
            assert backward_timestamp is not None, "Backward rotations must be provided with backward timestamp."
            self.backward_rotations = backward_rotations
        else:
            self.backward_rotations = None

    def keep_indices(self, indices):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value[indices])

    def to(self, dtype):
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(dtype=dtype))

    def transition(self, target_timestamp, mask=None):
        if mask is None:
            mask = torch.ones(len(self.means), dtype=torch.bool, device=self.means.device)
        transitioned_means = self.transition_means(target_timestamp, mask)
        transitioned_harmonics = self.transition_harmonics(target_timestamp, mask)
        transitioned_opacities = self.transition_opacities(target_timestamp, mask)
        transitioned_scales = self.transition_scales(target_timestamp, mask)
        transitioned_rotations = self.transition_rotations(target_timestamp, mask)
        return Gaussians(
            means=transitioned_means,
            harmonics=transitioned_harmonics,
            opacities=transitioned_opacities,
            scales=transitioned_scales,
            rotations=transitioned_rotations,
        )

    def transition_means(self, target_timestamp, mask):
        means = self.means[mask]
        if self.timestamp == -1 or target_timestamp == self.timestamp:
            delta_means = torch.zeros_like(means)
            # we still consider forward and backward velocities for gradient flow
            if self.forward_vel is not None:
                delta_means = delta_means + 0 * self.forward_vel[mask]
            if self.backward_vel is not None:
                delta_means = delta_means + 0 * self.backward_vel[mask]
        elif target_timestamp > self.timestamp and target_timestamp < self.forward_timestamp:
            delta_time = (target_timestamp - self.timestamp) / (self.forward_timestamp - self.timestamp)
            delta_means = self.forward_vel[mask] * delta_time
        elif target_timestamp < self.timestamp and target_timestamp > self.backward_timestamp:
            delta_time = (self.timestamp - target_timestamp) / (self.timestamp - self.backward_timestamp)
            delta_means = self.backward_vel[mask] * delta_time
        else:
            means = means[[]]
            delta_means = torch.zeros_like(means)
            if self.forward_vel is not None:
                delta_means = delta_means + 0 * self.forward_vel[[]]
            if self.backward_vel is not None:
                delta_means = delta_means + 0 * self.backward_vel[[]]
        transitioned_means = means + delta_means
        return transitioned_means

    def transition_harmonics(self, target_timestamp, mask):
        if self.timestamp == -1 or target_timestamp == self.timestamp:
            transitioned_harmonics = self.harmonics[mask]
        elif target_timestamp > self.timestamp and target_timestamp < self.forward_timestamp:
            transitioned_harmonics = self.harmonics[mask]
        elif target_timestamp < self.timestamp and target_timestamp > self.backward_timestamp:
            transitioned_harmonics = self.harmonics[mask]
        else:
            transitioned_harmonics = self.harmonics[[]]
        return transitioned_harmonics

    def transition_opacities(self, target_timestamp, mask):
        opacities = self.opacities[mask]
        if isinstance(self.life_span, float):
            life_span = torch.ones_like(opacities) * self.life_span
        else:
            life_span = self.life_span[mask]

        if self.timestamp == -1 or target_timestamp == self.timestamp:
            delta_time = 0
        elif target_timestamp > self.timestamp and target_timestamp < self.forward_timestamp:
            delta_time = (target_timestamp - self.timestamp) / (self.forward_timestamp - self.timestamp)
        elif target_timestamp < self.timestamp and target_timestamp > self.backward_timestamp:
            delta_time = (self.timestamp - target_timestamp) / (self.timestamp - self.backward_timestamp)
        else:
            opacities = opacities[[]]
            life_span = life_span[[]]
            delta_time = 0

        # Power decay function: o(x) = o * exp(-γ * x^(1/(1-T+ε)))
        # - T (life_span) controls decay rate: T=0 → fast decay, T→1 → slow decay
        # - γ (life_span_gamma) controls overall decay strength
        # - x (delta_time) is normalized time difference in [0, 1]
        power = 1.0 / (1.0 - life_span + 1e-6)
        transitioned_opacities = opacities * torch.exp(
            -self.life_span_gamma * (delta_time ** power)
        )
        return transitioned_opacities

    def transition_scales(self, target_timestamp, mask):
        scales = self.scales[mask]
        if self.timestamp == -1 or target_timestamp == self.timestamp:
            delta_scales = torch.ones_like(scales)
            if self.forward_scales is not None:
                # we still consider forward and backward for gradient flow
                delta_scales = delta_scales * torch.pow(self.forward_scales[mask], 0)
            if self.backward_scales is not None:
                delta_scales = delta_scales * torch.pow(self.backward_scales[mask], 0)
        elif target_timestamp > self.timestamp and target_timestamp < self.forward_timestamp:
            if self.forward_scales is None:
                delta_scales = torch.ones_like(scales)
            else:
                delta_time = (target_timestamp - self.timestamp) / (self.forward_timestamp - self.timestamp)
                delta_scales = torch.pow(self.forward_scales[mask], delta_time)
        elif target_timestamp < self.timestamp and target_timestamp > self.backward_timestamp:
            if self.backward_scales is None:
                delta_scales = torch.ones_like(scales)
            else:
                delta_time = (self.timestamp - target_timestamp) / (self.timestamp - self.backward_timestamp)
                delta_scales = torch.pow(self.backward_scales[mask], delta_time)
        else:
            # only consider constant gaussians
            scales = scales[[]]
            delta_scales = torch.ones_like(scales)
            if self.forward_scales is not None:
                delta_scales = delta_scales * torch.pow(self.forward_scales[[]], 0)
            if self.backward_scales is not None:
                delta_scales = delta_scales * torch.pow(self.backward_scales[[]], 0)
        transitioned_scales = scales * delta_scales
        return transitioned_scales

    def transition_rotations(self, target_timestamp, mask):
        rotations = self.rotations[mask]
        if self.timestamp == -1 or target_timestamp == self.timestamp:
            delta_rotations = torch.zeros_like(rotations[:, :3])
            # we still consider forward and backward for gradient flow
            if self.forward_rotations is not None:
                delta_rotations = delta_rotations + 0 * self.forward_rotations[mask]
            if self.backward_rotations is not None:
                delta_rotations = delta_rotations + 0 * self.backward_rotations[mask]
        elif target_timestamp > self.timestamp and target_timestamp < self.forward_timestamp:
            if self.forward_rotations is None:
                delta_rotations = torch.zeros_like(rotations[:, :3])
            else:
                delta_time = (target_timestamp - self.timestamp) / (self.forward_timestamp - self.timestamp)
                delta_rotations = self.forward_rotations[mask] * delta_time
        elif target_timestamp < self.timestamp and target_timestamp > self.backward_timestamp:
            if self.backward_rotations is None:
                delta_rotations = torch.zeros_like(rotations[:, :3])
            else:
                delta_time = (self.timestamp - target_timestamp) / (self.timestamp - self.backward_timestamp)
                delta_rotations = self.backward_rotations[mask] * delta_time
        else:
            # only consider constant gaussians
            rotations = rotations[[]]
            delta_rotations = torch.zeros_like(rotations[:, :3])
            if self.forward_rotations is not None:
                delta_rotations = delta_rotations + 0 * self.forward_rotations[[]]
            if self.backward_rotations is not None:
                delta_rotations = delta_rotations + 0 * self.backward_rotations[[]]
        transitioned_rotations = self.rotate_quaternion(rotations, delta_rotations)
        return transitioned_rotations

    def rotate_quaternion(self, quaternion, rot_velocity):
        angle = torch.norm(rot_velocity, p=2, dim=-1, keepdim=True)
        axis = F.normalize(rot_velocity, p=2, dim=-1, eps=1e-8)
        half_angle = angle / 2
        cos_half_angle = torch.cos(half_angle)
        sin_half_angle = torch.sin(half_angle)
        delta_quat = torch.cat(
            [cos_half_angle, axis * sin_half_angle], dim=-1
        )
        transitioned_rotations = self.quaternion_multiply(quaternion, delta_quat)
        return transitioned_rotations

    def quaternion_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )


class Rasterizer:
    def __init__(self, rasterization_mode="classic", packed=False, abs_grad=True, with_eval3d=False,
                 camera_model="pinhole", sparse_grad=False, distributed=False, grad_strategy=DefaultStrategy,
                 confidence_prune_threshold=-1, opacity_prune_threshold=-1, bidirection=True, backgrounds="black"):
        self.rasterization_mode = rasterization_mode
        self.packed = packed
        self.abs_grad = abs_grad
        self.camera_model = camera_model
        self.sparse_grad = sparse_grad
        self.grad_strategy = grad_strategy
        self.distributed = distributed
        self.with_eval3d = with_eval3d
        self.confidence_prune_threshold = confidence_prune_threshold
        self.opacity_prune_threshold = opacity_prune_threshold
        self.bidirection = bidirection
        self.backgrounds = backgrounds

    def forward(self, render_splats, render_viewmats, render_Ks, render_timestamps, sh_degree, width, height):
        assert len(render_splats) == len(render_viewmats) == len(render_Ks) == len(render_timestamps), \
            "Number of batches in gaussians must match the batch size in render_viewmats and render_Ks."
        # Prevent OOM by using chunked rendering
        rendered_colors_list, rendered_depths_list, rendered_alphas_list = [], [], []
        for b_idx in range(len(render_splats)):
            batch_colors_list, batch_depths_list, batch_alphas_list = [], [], []
            batch_splats = render_splats[b_idx]
            batch_viewmats = render_viewmats[b_idx]
            batch_Ks = render_Ks[b_idx]
            batch_timestamps = render_timestamps[b_idx]
            assert len(batch_viewmats) == len(batch_Ks) == len(batch_timestamps)
            assert len(batch_splats) > 0, "At least one Gaussian must be present in the batch."
            for s_idx in range(len(batch_viewmats)):
                viewmats_i = batch_viewmats[s_idx]
                Ks_i = batch_Ks[s_idx]
                timestamp_i = batch_timestamps[s_idx]
                transitioned_splats = []
                for splats in batch_splats:
                    if splats.timestamp == -1 or splats.timestamp == timestamp_i:
                        render_flag = True
                    elif timestamp_i > splats.timestamp and splats.forward_timestamp is not None and timestamp_i < splats.forward_timestamp:
                        if self.bidirection:
                            render_flag = True
                        elif abs(timestamp_i - splats.timestamp) <= abs(timestamp_i - splats.forward_timestamp):
                            render_flag = True
                        else:
                            render_flag = False
                    elif timestamp_i < splats.timestamp and splats.backward_timestamp is not None and timestamp_i > splats.backward_timestamp:
                        if self.bidirection:
                            render_flag = True
                        elif abs(timestamp_i - splats.timestamp) < abs(timestamp_i - splats.backward_timestamp):
                            render_flag = True
                        else:
                            render_flag = False
                    else:
                        render_flag = False
                    if render_flag:
                        mask = torch.ones_like(splats.opacities, dtype=torch.bool)
                        if self.opacity_prune_threshold >= 0:
                            mask = mask & (splats.opacities >= self.opacity_prune_threshold)
                        if self.confidence_prune_threshold >= 0 and splats.confidences is not None:
                            mask = mask & (splats.confidences >= self.confidence_prune_threshold)
                        transitioned_splats.append(
                            splats.transition(timestamp_i, mask=mask)
                        )
                rendered_colors, rendered_depths, rendered_alphas = self.rasterize_splats(
                    transitioned_splats, viewmats_i[None], Ks_i[None],
                    width=width, height=height, sh_degree=sh_degree,
                )
                batch_colors_list.append(rendered_colors)
                batch_depths_list.append(rendered_depths)
                batch_alphas_list.append(rendered_alphas)
            rendered_colors_list.append(torch.cat(batch_colors_list, dim=0))  # V H W 3
            rendered_depths_list.append(torch.cat(batch_depths_list, dim=0))  # V H W 1
            rendered_alphas_list.append(torch.cat(batch_alphas_list, dim=0))  # V H W 1
        rendered_colors = torch.stack(rendered_colors_list, dim=0)
        rendered_depths = torch.stack(rendered_depths_list, dim=0)
        rendered_alphas = torch.stack(rendered_alphas_list, dim=0)
        return rendered_colors, rendered_depths, rendered_alphas

    def rasterize_splats(
        self,
        splats,
        viewmats: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        if len(splats) > 0:
            means = torch.cat([splat.means for splat in splats], dim=0)
            quats = torch.cat([splat.rotations for splat in splats], dim=0)
            scales = torch.cat([splat.scales for splat in splats], dim=0)
            opacities = torch.cat([splat.opacities for splat in splats], dim=0)
            colors = torch.cat([splat.harmonics for splat in splats], dim=0)

        if len(splats) == 0 or means.shape[0] == 0:
            return (
                torch.zeros((1, height, width, 3), dtype=torch.float32, device=viewmats.device),
                torch.zeros((1, height, width, 1), dtype=torch.float32, device=viewmats.device),
                torch.zeros((1, height, width, 1), dtype=torch.float32, device=viewmats.device),
            )

        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.packed,
            absgrad=(
                self.abs_grad
                if isinstance(self.grad_strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.sparse_grad,
            rasterize_mode=self.rasterization_mode,
            distributed=self.distributed,
            camera_model=self.camera_model,
            with_eval3d=self.with_eval3d,
            render_mode="RGB+ED",
            backgrounds=means.new_ones((1, 3)) if self.backgrounds == "white" else None,
            **kwargs,
        )
        return render_colors[..., :3].clamp(0.0, 1.0), render_colors[..., 3:4], render_alphas


class GaussianSplatRenderer(nn.Module):
    def __init__(
        self,
        feature_dim: int = 256,       # Output channels of gs_feat_head
        sh_degree: int = 0,
        enable_prune: bool = True,
        voxel_size: float = 0.002,    # Default voxel size for prune_gs
        enable_conf_filter: bool = False,  # Enable confidence filtering
        conf_threshold_percent: float = 30.0,  # Confidence threshold percentage
        max_gaussians: int = 5000000,  # Maximum number of Gaussians
        is_4dgs: bool = False,
        life_span_gamma: float = 10.0,  # Gamma for life span decay
        dynamic_threshold: float = 0,  # Threshold for classifying dynamic gaussians
        global_motion_tracking: bool = False,  # Whether to use global motion tracking
        dynamic_threshold2: float = 0.0,  # Second dynamic threshold after global motion tracking
        occlusion_threshold: float = 0.05,  # Depth threshold for occlusion checking
        bidirection: bool = True,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.nums_sh = (sh_degree + 1) ** 2
        self.voxel_size = voxel_size
        self.enable_prune = enable_prune
        self.enable_conf_filter = enable_conf_filter
        self.conf_threshold_percent = conf_threshold_percent
        self.max_gaussians = max_gaussians
        self.is_4dgs = is_4dgs
        self.life_span_gamma = life_span_gamma
        self.dynamic_threshold = dynamic_threshold
        self.global_motion_tracking = global_motion_tracking
        self.dynamic_threshold2 = dynamic_threshold2
        self.occlusion_threshold = occlusion_threshold

        # Predict Gaussian parameters from GS features (quaternions/scales/opacities/SH/weights)
        splits_and_inits = [
            (4, 1.0, 0.0),                # quats
            (3, 0.00003, -7.0),           # scales
            (1, 1.0, -2.0),               # opacities
            (3 * self.nums_sh, 1.0, 0.0), # residual_sh
            (1, 1.0, -2.0),               # weights for 3DGS or life span for 4DGS
        ]
        gaussian_raw_channels = 4 + 3 + 1 + self.nums_sh * 3 + 1

        self.gs_head = nn.Sequential(
            nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(feature_dim, gaussian_raw_channels, kernel_size=1),
        )
        # Initialize weights and biases of the final layer by segments
        final_conv_layer = self.gs_head[-1]
        start_channels = 0
        for out_channel, s, b in splits_and_inits:
            nn.init.xavier_uniform_(final_conv_layer.weight[start_channels:start_channels+out_channel], s)
            nn.init.constant_(final_conv_layer.bias[start_channels:start_channels+out_channel], b)
            start_channels += out_channel

        if self.is_4dgs:
            self.gs_head_dynamic = nn.Sequential(
                nn.Conv2d(feature_dim // 2, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(feature_dim, gaussian_raw_channels, kernel_size=1),
            )
            final_conv_layer = self.gs_head_dynamic[-1]
            start_channels = 0
            for out_channel, s, b in splits_and_inits:
                nn.init.xavier_uniform_(final_conv_layer.weight[start_channels:start_channels+out_channel], s)
                nn.init.constant_(final_conv_layer.bias[start_channels:start_channels+out_channel], b)
                start_channels += out_channel

        # Rasterizer
        self.rasterizer = Rasterizer(bidirection=bidirection)

    # ======== Main entry point: Complete GS rendering and fill results back to predictions ========
    def render(
        self,
        gs_feats: torch.Tensor,                    # [B, S, 3, H, W]
        images: torch.Tensor,                      # [B, S+V, 3, H, W]
        predictions: Dict[str, torch.Tensor],      # From WorldMirror: pose/depth/pts3d etc
        views: Dict[str, torch.Tensor],
        context_predictions: Dict[str, torch.Tensor],
        is_inference: bool=True,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns predictions with the following fields filled:
        - rendered_colors / rendered_depths / (rendered_alphas during training)
        - gt_colors / gt_depths / valid_masks
        - splats / rendered_extrinsics / rendered_intrinsics
        """
        B, _, _, H, W = images.shape
        S = context_predictions.get("imgs", images).shape[1] # context view nums
        V = images.shape[1] - S                              # target view nums

        # 1) Predict GS features from tokens, then convert to Gaussian parameters
        gs_feats_reshape = rearrange(gs_feats, "b s c h w -> (b s) c h w")
        gs_params_static = self.gs_head(gs_feats_reshape)
        if self.is_4dgs:
            gs_params_dynamic = self.gs_head_dynamic(gs_feats_reshape)
            is_static = views["is_static"][:, :S].reshape(-1)
            gs_params = torch.where(
                is_static[:, None, None, None],
                gs_params_static,
                gs_params_dynamic
            )
        else:
            gs_params = gs_params_static
        gt_colors = images.permute(0, 1, 3, 4, 2)

        # 2) Select rendering cameras
        if self.training:
            # Using all gt cameras
            render_viewmats, render_Ks = self.prepare_cameras(views, S + V)
            gt_valid_masks_src = views["valid_mask"][:, :S]      # [B, S, H, W]
            gt_valid_masks_tgt = views["valid_mask"][:, S:]     # [B, V, H, W]
            unproject_masks = calculate_unprojected_mask(views, S)     # [B, V, H, W]
            valid_masks = torch.cat([gt_valid_masks_src, (gt_valid_masks_tgt & unproject_masks)], dim=1)
            render_timestamps = views["timestamp"]
        else:
            # Re-predict the camera for novel views and perform translation scale alignment
            pred_all_extrinsic, pred_all_intrinsic = self.prepare_cameras(predictions, S + V)
            scale_factor = 1.0
            if "camera_poses" in context_predictions:
                pred_context_extrinsic, _ = self.prepare_cameras(context_predictions, S)
                scale_factor = pred_context_extrinsic[:, :, :3, 3].norm(dim=-1).mean(dim=1, keepdim=True) / (
                    pred_all_extrinsic[:, :S, :3, 3].norm(dim=-1).mean(dim=1, keepdim=True) + 1e-6
                )
                scale_factor = scale_factor.unsqueeze(-1)

            pred_all_extrinsic[..., :3, 3] = pred_all_extrinsic[..., :3, 3] * scale_factor
            render_viewmats, render_Ks = pred_all_extrinsic, pred_all_intrinsic
            valid_masks = views.get("valid_mask", torch.ones(B, S + V, H, W, dtype=bool, device=images.device))
            render_timestamps = views["timestamp"]
        render_world2cam = closed_form_inverse_se3(render_viewmats.reshape(-1, 4, 4)).reshape(B, -1, 4, 4)

        # 3) Generate splats from gs_params + predictions, and perform voxel merging
        if self.training:
            splats = self.prepare_splats(views, predictions, images, gs_params, S, position_from="gsdepth+gtcamera")
        elif not is_inference:
            splats = self.prepare_splats(views, predictions, images, gs_params, S, context_predictions, position_from="gsdepth+predcamera")
        else:
            splats = self.prepare_splats(views, predictions, images, gs_params, S, position_from="gsdepth+predcamera")

        predictions["splats"] = splats
        # # 4) Rasterization rendering (training: chunked rendering + novel view valid mask correction; evaluation: view-by-view)
        # rendered_colors, rendered_depths, rendered_alphas = self.rasterizer.forward(
        #     splats,
        #     render_viewmats=render_world2cam.detach(),
        #     render_Ks=render_Ks.detach(),
        #     render_timestamps=render_timestamps,
        #     sh_degree=min(self.sh_degree, 0),
        #     width=W, height=H,
        # )

        # if self.training and V > 0:
        #     nvs_rendered_mask = rendered_alphas[:, S:, ..., 0].detach() > 0.1
        #     valid_masks[:, S:] = nvs_rendered_mask & valid_masks[:, S:]

        # # 5) return predictions
        # predictions["rendered_colors"] = rendered_colors
        # predictions["rendered_depths"] = rendered_depths
        # predictions["rendered_alphas"] = rendered_alphas
        # predictions["gt_colors"] = gt_colors
        # predictions["gt_depths"] = views.get("depthmap")
        # predictions["valid_masks"] = valid_masks.bool()
        predictions["rendered_extrinsics"] = render_viewmats
        predictions["rendered_intrinsics"] = render_Ks
        predictions["rendered_timestamps"] = render_timestamps
        return predictions

    def apply_confidence_filter(self, splats, conf):
        """
        Apply confidence filtering to Gaussian splats before pruning.
        Discard bottom p% confidence points, keep top (100-p)%.

        Args:
            splats: Dictionary containing Gaussian parameters
            gs_depth_conf: Confidence tensor [B, S, H, W]

        Returns:
            Filtered splats dictionary
        """
        if not self.enable_conf_filter or conf is None:
            return splats

        N = splats["means"].shape[0]

        # Mask invalid/very small values
        conf = conf.masked_fill(conf <= 1e-5, float("-inf"))

        # Keep top (100-p)% points, discard bottom p%
        if self.conf_threshold_percent > 0:
            keep_from_percent = int(np.ceil(N * (100.0 - self.conf_threshold_percent) / 100.0))
        else:
            keep_from_percent = N
        K = max(1, min(self.max_gaussians, keep_from_percent))

        # Select top-K indices for each batch (deterministic, no randomness)
        topk_idx = torch.topk(conf, K, dim=0, largest=True, sorted=False).indices  # [K]

        filtered = {}
        mask_keys = ["means", "quats", "scales", "opacities", "sh", "conf", "weights"]

        for key in splats.keys():
            if key in mask_keys and key in splats:
                x = splats[key]
                if x.ndim == 1:  # [B, N]
                    filtered[key] = torch.gather(x, 0, topk_idx)
                else:
                    # Expand indices to match tensor dimensions
                    expand_idx = topk_idx.clone()
                    for i in range(x.ndim - 1):
                        expand_idx = expand_idx.unsqueeze(-1)
                    expand_idx = expand_idx.expand(-1, *x.shape[1:])
                    filtered[key] = torch.gather(x, 0, expand_idx)
            else:
                filtered[key] = splats[key]

        return filtered

    def prune_gs(self, splats, voxel_size=0.002):
        """
        Prune Gaussian splats by merging those in the same voxel.

        Args:
            splats: Dictionary containing Gaussian parameters
            voxel_size: Size of voxels for spatial grouping

        Returns:
            Dictionary with pruned splats
        """
        # Compute voxel indices
        coords = splats["means"]
        voxel_indices = (coords / voxel_size).floor().long()
        unique_voxels, inverse_indices = torch.unique(
            voxel_indices, dim=0, return_inverse=True
        )
        splat_weights = splats["weights"]
        voxel_weights = scatter_sum(splat_weights, inverse_indices, dim=0)
        weights = splat_weights / torch.clamp(voxel_weights[inverse_indices], min=1e-8)

        merged = {}
        for key, data in splats.items():
            if data.ndim == 1:
                merged[key] = scatter_sum(data * weights, inverse_indices, dim=0)
            elif data.ndim == 2:
                merged[key] = scatter_sum(data * weights.unsqueeze(-1), inverse_indices, dim=0)
            else:
                merged[key] = scatter_sum(data * weights.unsqueeze(-1).unsqueeze(-1), inverse_indices, dim=0)
        merged["quats"] = F.normalize(merged["quats"], p=2, dim=-1, eps=1e-8)
        return merged

    def prepare_splats(self, views, predictions, images, gs_params, context_nums,
                       context_predictions={}, position_from="gsdepth+gtcamera"):
        """
        Prepare Gaussian splats from model predictions and input data.

        Args:
            views: Dictionary containing view data (camera poses, intrinsics, etc.)
            predictions: Model predictions including depth, pose_enc, etc.
            images: Input images [B, S_all, 3, H, W]
            gs_params: Gaussian splatting parameters from model
            context_predictions: Optional context predictions for camera poses
            position_from: Method to compute 3D positions ("pts3d", "gsdepth+gtcamera", "gsdepth+predcamera)
            debug: Whether to use debug mode with ground truth data

        Returns:
            splats: Dictionary containing prepared Gaussian splat parameters
        """
        B, _, _, H, W = images.shape
        S = context_nums
        splats = {}

        # Only take parameters from source view branch
        gs_params = rearrange(gs_params, "(b s) c h w -> b s h w c", b=B)
        splats["gs_feats"] = gs_params.reshape(B, S, H * W, -1)

        # Split Gaussian parameters
        quats, scales, opacities, residual_sh, weights = torch.split(
            gs_params, [4, 3, 1, self.nums_sh * 3, 1], dim=-1
        )

        # Apply activation functions to Gaussian parameters
        splats["quats"] = act_gs.reg_dense_rotation(quats.reshape(B, S, H * W, 4))
        splats["scales"] = act_gs.reg_dense_scales(scales.reshape(B, S, H * W, 3)).clamp_max(0.3)
        splats["opacities"] = act_gs.reg_dense_opacities(opacities.reshape(B, S, H * W))

        # Handle spherical harmonics (SH) coefficients
        residual_sh = act_gs.reg_dense_sh(residual_sh.reshape(B, S, H * W, self.nums_sh * 3))
        new_sh = torch.zeros_like(residual_sh)
        new_sh[..., 0, :] = sh_utils.RGB2SH(
            images[:, :S].permute(0, 1, 3, 4, 2).reshape(B, S, H * W, 3)
        )
        splats["sh"] = new_sh + residual_sh

        splats["weights"] = act_gs.reg_dense_weights(weights.reshape(B, S, H * W))

        # Compute 3D positions based on specified method
        depth_from, camera_from = position_from.split("+")
        if depth_from == "depth":
            depth = context_predictions.get("depth", predictions["depth"][:, :S]).reshape(B * S, H, W)
            conf = context_predictions.get("depth_conf", predictions["depth_conf"][:, :S])
        elif depth_from == "gsdepth":
            depth = predictions["gs_depth"][:, :S].reshape(B * S, H, W)
            conf = predictions["gs_depth_conf"][:, :S]
        else:
            raise ValueError(f"Invalid depth_from={depth_from}")

        if camera_from == "gtcamera":
            pose4x4 = views["camera_poses"][:, :S].reshape(B * S, 4, 4)
            intrinsic = views["camera_intrs"][:, :S].reshape(B * S, 3, 3)
        elif camera_from == "predcamera":
            pose4x4 = context_predictions.get("camera_poses", predictions["camera_poses"])[:, :S].reshape(B * S, 4, 4).detach()
            intrinsic = context_predictions.get("camera_intrs", predictions["camera_intrs"])[:, :S].reshape(B * S, 3, 3).detach()
        else:
            raise ValueError(f"Invalid camera_from={camera_from}")

        pts3d, _, _ = depth_to_world_coords_points(depth, pose4x4, intrinsic)
        pts3d = pts3d.reshape(B, S, H * W, 3)
        splats["means"] = pts3d
        splats["conf"] = conf.reshape(B, S, H * W)

        splats["timestamp"] = views["timestamp"][:, :S]
        if "velocity_fwd" in predictions:
            camera2world = pose4x4.reshape(B, S, 4, 4)
            world_velocity_fwd = torch.einsum(
                "bsij, bshwj -> bshwi", camera2world[:, :S-1, :3, :3], predictions["velocity_fwd"]
            )
            splats["world_velocity_fwd"] = world_velocity_fwd.reshape(B, S-1, H * W, 3)

            world_velocity_bwd = torch.einsum(
                "bsij, bshwj -> bshwi", camera2world[:, 1:, :3, :3], predictions["velocity_bwd"]
            )
            splats["world_velocity_bwd"] = world_velocity_bwd.reshape(B, S-1, H * W, 3)

            context_vel_mag_fwd = torch.cat([world_velocity_fwd.norm(dim=-1), torch.zeros_like(world_velocity_fwd[:, :1, ..., 0])], dim=1)
            context_vel_mag_bwd = torch.cat([torch.zeros_like(world_velocity_bwd[:, :1, ..., 0]), world_velocity_bwd.norm(dim=-1)], dim=1)
            context_vel_mag = torch.max(context_vel_mag_fwd, context_vel_mag_bwd)
        else:
            context_vel_mag = None

        if "gs_fwd_attr" in predictions:
            splats["angular_velocity_fwd"] = predictions["gs_fwd_attr"].reshape(B, S-1, H * W, -1)
            splats["angular_velocity_bwd"] = predictions["gs_bwd_attr"].reshape(B, S-1, H * W, -1)

        gaussians = self.separate_splats(
            splats,
            context_extrs=closed_form_inverse_se3(pose4x4).reshape(B, S, 4, 4),
            context_intrs=intrinsic.reshape(B, S, 3, 3),
            context_depth=depth.reshape(B, S, H, W),
            context_vel_mag=context_vel_mag,
            static_flag=views["is_static"][:, 0],
        )
        return gaussians

    def prepare_cameras(self, views, nums):
        viewmats = views["camera_poses"][:, :nums]
        Ks = views["camera_intrs"][:, :nums]
        return viewmats, Ks

    def separate_splats(self, splats, context_extrs=None, context_intrs=None, context_depth=None, context_vel_mag=None, static_flag=None):
        B = splats["means"].shape[0]
        gaussian_list = []
        for b in range(B):
            # Classify dynamic/constant based on velocity and static_flag
            dynamic_indices, constant_indices = self._classify_gaussians(
                means=splats["means"][b],
                static_flag=static_flag[b] if static_flag is not None else False,
                context_extrs=context_extrs[b] if context_extrs is not None else None,
                context_intrs=context_intrs[b] if context_intrs is not None else None,
                context_depth=context_depth[b] if context_depth is not None else None,
                context_vel_mag=context_vel_mag[b] if context_vel_mag is not None else None,
            )

            # Generate constant fused gaussians
            constant_gaussians = self._create_constant_gaussians(
                splats, b, constant_indices, static_flag[b] if static_flag is not None else False
            )

            # Compute attributes for dynamic gaussians
            dynamic_gaussians = self._create_dynamic_gaussians(splats, b, dynamic_indices)

            # Combine constant gaussians with dynamic/static gaussians
            final_gaussian_list = dynamic_gaussians
            if constant_gaussians is not None:
                final_gaussian_list.append(constant_gaussians)
            gaussian_list.append(final_gaussian_list)

        return gaussian_list

    def _create_constant_gaussians(self, splats, batch_idx, constant_indices, static_flag=True):
        if len(constant_indices) == 0:
            return None

        s_idx, n_idx = constant_indices[:, 0], constant_indices[:, 1]
        constant_splats = {}
        for key in ["means", "quats", "scales", "opacities", "sh", "conf", "weights"]:
            constant_splats[key] = splats[key][batch_idx][s_idx, n_idx]

        # Apply confidence filtering before pruning
        if self.enable_conf_filter:
            constant_splats = self.apply_confidence_filter(constant_splats, constant_splats["conf"])

        # Only apply pruning for static scenes
        if static_flag and self.enable_prune:
            constant_splats = self.prune_gs(constant_splats, voxel_size=self.voxel_size)

        gaussians = Gaussians(
            means=constant_splats["means"],
            harmonics=constant_splats["sh"],
            opacities=constant_splats["opacities"],
            scales=constant_splats["scales"],
            rotations=constant_splats["quats"],
            confidences=constant_splats["conf"],
            timestamp=-1,
        )
        return gaussians

    def _create_dynamic_gaussians(self, splats, batch_idx, dynamic_indices):
        S, N, _ = splats["means"][batch_idx].shape

        # Create dynamic mask matrix for more efficient indexing
        dynamic_mask_all = torch.zeros((S, N), dtype=torch.bool, device=splats["means"].device)
        if len(dynamic_indices) > 0:
            s_idx, n_idx = dynamic_indices[:, 0], dynamic_indices[:, 1]
            dynamic_mask_all[s_idx, n_idx] = True

        gaussian_list = []
        for s in range(S):
            dynamic_mask = dynamic_mask_all[s]
            if dynamic_mask.any():
                gs = Gaussians(
                    means=splats["means"][batch_idx, s][dynamic_mask],
                    harmonics=splats["sh"][batch_idx, s][dynamic_mask],
                    opacities=splats["opacities"][batch_idx, s][dynamic_mask],
                    scales=splats["scales"][batch_idx, s][dynamic_mask],
                    rotations=splats["quats"][batch_idx, s][dynamic_mask],
                    confidences=splats["conf"][batch_idx, s][dynamic_mask],
                    timestamp=splats["timestamp"][batch_idx, s].item(),
                    life_span=splats["weights"][batch_idx, s][dynamic_mask],
                    life_span_gamma=self.life_span_gamma,
                    forward_timestamp=splats["timestamp"][batch_idx, s + 1].item() if s < (S - 1) else None,
                    forward_vel=splats["world_velocity_fwd"][batch_idx, s][dynamic_mask] if "world_velocity_fwd" in splats and s < (S - 1) else None,
                    forward_rotations=splats["angular_velocity_fwd"][batch_idx, s][dynamic_mask] if "angular_velocity_fwd" in splats and s < (S - 1) else None,
                    backward_timestamp=splats["timestamp"][batch_idx, s - 1].item() if s > 0 else None,
                    backward_vel=splats["world_velocity_bwd"][batch_idx, s - 1][dynamic_mask] if "world_velocity_bwd" in splats and s > 0 else None,
                    backward_rotations=splats["angular_velocity_bwd"][batch_idx, s - 1][dynamic_mask] if "angular_velocity_bwd" in splats and s > 0 else None,
                )
                gaussian_list.append(gs)
        return gaussian_list

    def _classify_gaussians(self, means, static_flag=False,
                            context_extrs=None, context_intrs=None,
                            context_depth=None, context_vel_mag=None):
        """
        Classify gaussians into dynamic and constant categories.
        Returns masks for dynamic gaussians and fusion data for constant gaussians.
        """
        S, N, _ = means.shape
        if static_flag:
            constant_mask = torch.ones((S, N), dtype=torch.bool, device=means.device)
        else:
            constant_mask = torch.zeros((S, N), dtype=torch.bool, device=means.device)
            if context_vel_mag is not None:
                for s in range(S):
                    is_static = context_vel_mag[s].flatten() < self.dynamic_threshold
                    # Further classify static into constant and non-constant through global motion tracking
                    if is_static.sum() > 0:
                        static_indices = torch.where(is_static)[0]
                        if self.global_motion_tracking:
                            static_means = means[s, static_indices]
                            constant_classification = self._global_motion_tracking(
                                static_means, context_extrs, context_intrs, context_depth, context_vel_mag, s
                            )
                            constant_mask[s, static_indices] = constant_classification
                        else:
                            constant_mask[s, static_indices] = True
        constant_indices = torch.nonzero(constant_mask, as_tuple=False)
        dynamic_indices = torch.nonzero(~constant_mask, as_tuple=False)
        return dynamic_indices, constant_indices

    def _global_motion_tracking(self, static_means, context_extrs, context_intrs, context_depth, context_vel_mag, reference_idx):
        """
        Args:
            static_means: [N, 3] 3D coordinates of static Gaussians
            context_extrs: [S, 4, 4] context frame extrinsic matrices (world to camera)
            context_intrs: [S, 3, 3] context frame intrinsic matrices
            context_depth: [S, H, W, 1] context frame depth maps
            context_vel_mag: [S, H, W] context frame velocity magnitude maps
            reference_idx: int, the reference frame index for comparison

        Returns:
            is_constant: [N] boolean mask indicating which static Gaussians are constant
        """
        if len(static_means) == 0:
            return torch.zeros_like(static_means[:, 0], dtype=torch.bool)

        # If context data is not available, classify all static gaussians as constant
        if (context_extrs is None or context_intrs is None or
            context_depth is None or context_vel_mag is None):
            return torch.ones_like(static_means[:, 0], dtype=torch.bool)

        S = context_extrs.shape[0]
        # Replicate static means for all frames: [S, N, 3]
        static_means_expanded = static_means.unsqueeze(0).expand(S, -1, -1)

        # Project all points to all frames in one call
        projected_coords, projected_depths = project_world_points_to_image(
            static_means_expanded, context_extrs, context_intrs
        )  # [S, N, 2], [S, N, 1]

        # Remove depth dimension: [S, N]
        projected_depths = projected_depths.squeeze(-1)

        # Sample velocity magnitudes
        sampled_velocities, visibility_mask = self._sample_velocities_at_coords(
            context_vel_mag, projected_coords, projected_depths, context_depth, reference_idx
        )  # [S, N]

        sampled_velocities = sampled_velocities * visibility_mask.float()
        is_constant = sampled_velocities.max(dim=0).values < self.dynamic_threshold2
        return is_constant

    def _sample_velocities_at_coords(self, vel_mag_map, projected_coords, projected_depths, depth_maps, reference_idx):
        """Sample velocities at projected coordinates using nearest neighbor."""
        S, N = projected_coords.shape[:2]
        H, W = vel_mag_map.shape[-2], vel_mag_map.shape[-1]

        # First check if coordinates are within image bounds
        in_bounds = ((projected_coords[..., 0] >= 0) & (projected_coords[..., 0] < W) &
                     (projected_coords[..., 1] >= 0) & (projected_coords[..., 1] < H))
        valid_depth = projected_depths > 1e-8

        # Only process points that are within bounds
        valid_mask = in_bounds & valid_depth

        # Convert to integer coordinates for sampling
        coords_int = projected_coords.round().long()
        coords_int[..., 0] = coords_int[..., 0].clamp(0, W-1)
        coords_int[..., 1] = coords_int[..., 1].clamp(0, H-1)

        # Sample depth values from depth maps
        seq_indices = torch.arange(S, device=projected_depths.device)[:, None].expand(S, N)
        sampled_depths = depth_maps[seq_indices, coords_int[..., 1], coords_int[..., 0]]
        sampled_vels = vel_mag_map[seq_indices, coords_int[..., 1], coords_int[..., 0]]

        # Check visibility: point is visible if its projected depth is not significantly larger than depth map
        depth_check = projected_depths <= (sampled_depths + self.occlusion_threshold)

        # Only mark as visible if both valid and depth check passes
        is_visible = valid_mask & depth_check
        is_visible[reference_idx] = True

        return sampled_vels, is_visible

if __name__ == "__main__":
    device = "cuda:0"
    means = torch.randn((100, 3), device=device)
    quats = torch.randn((100, 4), device=device)
    scales = torch.rand((100, 3), device=device) * 0.1
    opacities = torch.rand((100,), device=device)
    colors = torch.rand((100, 3), device=device)

    viewmats = torch.eye(4, device=device)[None, :, :].repeat(10, 1, 1)
    Ks = torch.tensor([
    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :].repeat(10, 1, 1)
    width, height = 300, 200

    rasterizer = Rasterizer()
    splats = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
    }
    colors, alphas, _ = rasterizer.rasterize_splats(splats, viewmats, Ks, width, height)
