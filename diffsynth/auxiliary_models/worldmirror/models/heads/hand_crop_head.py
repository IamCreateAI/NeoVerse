"""
Hand prediction head with ROI-Align feature cropping.

Instead of processing the full-image feature map and spatially averaging
(which dilutes hand signal with background), this head:

  1. Projects backbone tokens to multi-scale 2D feature maps (same as DPTHead)
  2. Adds positional embeddings *before* ROI Align (preserving absolute spatial info)
  3. Applies ROI Align at each scale using hand bounding boxes
  4. Fuses cropped multi-scale features via DPT-style refinement blocks
  5. Predicts per-hand MANO parameters via a lightweight conv head

The bounding boxes are expected in normalized [0, 1] coordinates
(x1, y1, x2, y2) and can come from any source: GT projection, a hand
detector, or a learned proposal network.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .dense_head import _make_scratch, _make_fusion_block, custom_interpolate
from ..utils.grid import create_uv_grid, position_grid_to_embed


class HandCropHead(nn.Module):
    """
    Hand prediction head with ROI Align on backbone feature maps.

    Parameters
    ----------
    dim_in : int
        Input feature dimension from backbone (typically ``2 * embed_dim``).
    patch_size : int
        Backbone patch size (default 14 for DINOv2-L).
    hand_param_dim : int
        Output dimension **per hand** (32 for MANO: 3 pos + 4 quat + 15 pose + 10 betas).
    num_hands : int
        Number of hands to predict per frame (default 2).
    crop_size : int
        ROI Align output size at the *native patch-grid* scale.
        Other DPT scales are derived proportionally (4x, 2x, 1x, 0.5x).
        With ``crop_size=8`` and ``patch_size=14``: the finest crop is 32x32,
        covering 8*14=112 pixels in the original 224x224 image.
    features : int
        Channel width of the DPT fusion blocks.
    out_channels : list[int]
        Per-layer projection channel sizes (must have 4 entries).
    pos_embed : bool
        If True, add sinusoidal positional embeddings to the projected
        feature maps *before* ROI Align so that absolute image position
        is preserved in the cropped features.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        hand_param_dim: int = 32,
        num_hands: int = 2,
        crop_size: int = 8,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.hand_param_dim = hand_param_dim
        self.num_hands = num_hands
        self.crop_size = crop_size
        self.pos_embed = pos_embed

        # --- Multi-scale projection (mirrors DPTHead) -----------------------
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0],
                               kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1],
                               kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3],
                      kernel_size=3, stride=2, padding=1),
        ])

        # --- Feature fusion (mirrors DPTHead) --------------------------------
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(
            features, features // 2, kernel_size=3, stride=1, padding=1,
        )

        # --- Per-hand prediction head ----------------------------------------
        # Conv layers + pool produce a feature vector per crop.
        # We concatenate 4 crop-geometry values (principal-point offset +
        # crop size) so the predictor can map from crop-local features back
        # to global camera coordinates.
        self.hand_feat_extractor = nn.Sequential(
            nn.Conv2d(features // 2, features // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 4, features // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),                          # → [N*nh, features//4]
        )
        # +4 for crop geometry: (offset_x, offset_y, crop_w, crop_h)
        self.hand_fc = nn.Linear(features // 4 + 4, hand_param_dim)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        hand_bboxes: torch.Tensor,
        hand_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        token_list : list[Tensor]
            4 tensors of shape ``[B, S, N_tokens, C]`` from backbone
            intermediate layers.
        images : Tensor
            Input images ``[B, S, 3, H, W]`` (used only for shape).
        patch_start_idx : int
            Index where patch tokens start in the token sequence.
        hand_bboxes : Tensor
            ``[B, S, num_hands, 4]`` bounding boxes in **normalized [0, 1]**
            image coordinates, format ``(x1, y1, x2, y2)``.
        hand_valid : Tensor, optional
            ``[B, S, num_hands]`` boolean mask.  Invalid hands get zeroed
            predictions.

        Returns
        -------
        hand_params : Tensor
            ``[B, S, num_hands * hand_param_dim]`` — concatenated MANO
            parameters for all hands.
        """
        B, S, _, H, W = images.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        N = B * S  # total number of frames

        # ---- Phase 1: project tokens → multi-scale feature maps -------------
        feats = []
        for proj, resize, tokens in zip(
            self.projects, self.resize_layers, token_list
        ):
            patch_tokens = tokens[:, :, patch_start_idx:]          # [B, S, P, C]
            patch_tokens = patch_tokens.reshape(N, -1, tokens.shape[-1])
            patch_tokens = self.norm(patch_tokens)

            feat = (patch_tokens
                    .permute(0, 2, 1)
                    .reshape(N, -1, ph, pw))                       # [N, C, ph, pw]
            feat = proj(feat)

            # Positional embeddings encode *absolute* image position so that
            # the cropped features know where in the full frame they come from.
            if self.pos_embed:
                feat = self._apply_pos_embed(feat, W, H)

            feat = resize(feat)
            feats.append(feat)

        # Shapes (224×224 input, patch=14 → ph=pw=16):
        #   feats[0]: [N, 256,  64, 64]   (4× upsampled)
        #   feats[1]: [N, 512,  32, 32]   (2× upsampled)
        #   feats[2]: [N, 1024, 16, 16]   (native patch grid)
        #   feats[3]: [N, 1024,  8,  8]   (2× downsampled)

        # ---- Phase 2: ROI Align at each scale --------------------------------
        rois = self._prepare_rois(hand_bboxes, B, S, H, W)
        # rois: [N * num_hands, 5]

        cs = self.crop_size
        roi_output_sizes = [cs * 4, cs * 2, cs, max(cs // 2, 2)]

        cropped_feats = []
        for feat, out_size in zip(feats, roi_output_sizes):
            spatial_scale = feat.shape[-1] / float(W)
            cropped = roi_align(
                feat, rois,
                output_size=(out_size, out_size),
                spatial_scale=spatial_scale,
                aligned=True,
            )
            cropped_feats.append(cropped)

        # ---- Phase 3: fuse cropped features (DPT-style) ---------------------
        layer_1_rn = self.scratch.layer1_rn(cropped_feats[0])
        layer_2_rn = self.scratch.layer2_rn(cropped_feats[1])
        layer_3_rn = self.scratch.layer3_rn(cropped_feats[2])
        layer_4_rn = self.scratch.layer4_rn(cropped_feats[3])

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        out = self.scratch.refinenet1(out, layer_1_rn)
        out = self.scratch.output_conv1(out)
        # out: [N * num_hands, features//2, crop_size*8, crop_size*8]

        # ---- Phase 4: predict per-hand MANO params --------------------------
        # Extract visual features from cropped region
        vis_feat = self.hand_feat_extractor(out)     # [N*nh, features//4]

        # Compute crop geometry relative to the principal point (image centre).
        # This tells the predictor *where* in the full frame each crop sits,
        # which is essential for recovering the global wrist translation.
        #   offset_x = crop_centre_x − 0.5   (0.5 = normalised principal point)
        #   offset_y = crop_centre_y − 0.5
        #   crop_w   = x2 − x1               (normalised width)
        #   crop_h   = y2 − y1               (normalised height)
        crop_geom = self._compute_crop_geometry(hand_bboxes, B, S)  # [N*nh, 4]

        hand_params = self.hand_fc(torch.cat([vis_feat, crop_geom], dim=-1))
        # hand_params: [N * num_hands, hand_param_dim]

        hand_params = hand_params.reshape(B, S, self.num_hands, self.hand_param_dim)

        # Zero-out predictions for absent / invalid hands
        if hand_valid is not None:
            mask = hand_valid.reshape(B, S, self.num_hands, 1).float()
            hand_params = hand_params * mask

        # Flatten hands dim → [B, S, num_hands * hand_param_dim]
        hand_params = hand_params.reshape(B, S, self.num_hands * self.hand_param_dim)
        return hand_params

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_rois(
        self,
        hand_bboxes: torch.Tensor,
        B: int, S: int, H: int, W: int,
    ) -> torch.Tensor:
        """Convert normalised bboxes to ``roi_align`` format.

        Returns ``[N*num_hands, 5]`` tensor where each row is
        ``[batch_idx, x1, y1, x2, y2]`` in *pixel* coordinates and
        ``batch_idx`` indexes into the ``B*S`` (= N) dim of the feature map.
        """
        N = B * S
        bboxes = hand_bboxes.reshape(N, self.num_hands, 4)

        bboxes_pixel = bboxes.clone()
        bboxes_pixel[..., [0, 2]] *= W   # x coords
        bboxes_pixel[..., [1, 3]] *= H   # y coords

        batch_idx = torch.arange(N, device=bboxes.device, dtype=bboxes.dtype)
        batch_idx = batch_idx[:, None].expand(-1, self.num_hands)  # [N, nh]

        rois = torch.cat([
            batch_idx.reshape(-1, 1),
            bboxes_pixel.reshape(-1, 4),
        ], dim=1)
        return rois

    def _compute_crop_geometry(
        self,
        hand_bboxes: torch.Tensor,
        B: int, S: int,
    ) -> torch.Tensor:
        """Principal-point adjustment: encode each crop's position and size.

        Returns ``[N*num_hands, 4]`` with columns::

            offset_x = crop_centre_x − cx   (cx = 0.5 in normalised coords)
            offset_y = crop_centre_y − cy   (cy = 0.5)
            crop_w   = x2 − x1
            crop_h   = y2 − y1

        The offset tells the predictor how far the crop centre is from
        the camera principal point so it can correctly recover the
        global wrist translation  t_x = (u − cx) · Z / f.
        """
        N = B * S
        bboxes = hand_bboxes.reshape(N, self.num_hands, 4)  # normalised

        cx = (bboxes[..., 0] + bboxes[..., 2]) / 2.0  # crop centre x
        cy = (bboxes[..., 1] + bboxes[..., 3]) / 2.0  # crop centre y
        cw = bboxes[..., 2] - bboxes[..., 0]           # crop width
        ch = bboxes[..., 3] - bboxes[..., 1]           # crop height

        # Offset from principal point (image centre = 0.5 in normalised coords)
        geom = torch.stack([cx - 0.5, cy - 0.5, cw, ch], dim=-1)
        return geom.reshape(N * self.num_hands, 4)

    def _apply_pos_embed(
        self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1,
    ) -> torch.Tensor:
        """Sinusoidal positional embedding (identical to DPTHead)."""
        patch_w, patch_h = x.shape[-1], x.shape[-2]
        pos = create_uv_grid(
            patch_w, patch_h,
            aspect_ratio=W / H,
            dtype=x.dtype,
            device=x.device,
        )
        pos = position_grid_to_embed(pos, x.shape[1])
        pos = (pos * ratio).permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos
