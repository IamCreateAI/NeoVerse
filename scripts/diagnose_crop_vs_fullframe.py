"""
Diagnostic script: run one batch through both crop and full-frame paths
and print intermediate feature statistics to identify where information is lost.

Usage:
    python -m scripts.diagnose_crop_vs_fullframe --config configs/train_hand_head.yaml
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TVF

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror
from diffsynth.auxiliary_models.worldmirror.models.heads.hamer_head import HamerManoHead
from diffsynth.utils.auxiliary import load_video
from scripts.train_hand_head import (
    HOT3DHandDataset, discover_sequences, build_views,
    HAND_PARAM_DIM, NUM_HANDS,
)


def print_tensor_stats(name, t):
    """Print min/max/mean/std/nan/inf for a tensor."""
    if t is None:
        print(f"  {name}: None")
        return
    t_float = t.float()
    nans = t_float.isnan().sum().item()
    infs = t_float.isinf().sum().item()
    print(f"  {name}: shape={list(t.shape)} "
          f"min={t_float.min().item():.5f} max={t_float.max().item():.5f} "
          f"mean={t_float.mean().item():.5f} std={t_float.std().item():.5f} "
          f"nan={nans} inf={infs}")


def diagnose_bbox_quality(hand_bboxes, hand_valid):
    """Analyze bounding box statistics across the batch."""
    B, S = hand_bboxes.shape[:2]
    N = B * S
    bboxes = hand_bboxes.reshape(N, 2, 4)
    valid = hand_valid.reshape(N, 2)

    total_hands = N * 2
    n_valid = valid.sum().item()
    n_invalid = total_hands - n_valid

    print(f"\n{'='*60}")
    print(f"BBOX QUALITY ANALYSIS")
    print(f"{'='*60}")
    print(f"  Total hand slots: {total_hands}")
    print(f"  Valid hands: {n_valid} ({100*n_valid/total_hands:.1f}%)")
    print(f"  Invalid hands: {n_invalid} ({100*n_invalid/total_hands:.1f}%)")

    if n_valid > 0:
        valid_bboxes = bboxes[valid]  # [n_valid, 4]
        widths = valid_bboxes[:, 2] - valid_bboxes[:, 0]
        heights = valid_bboxes[:, 3] - valid_bboxes[:, 1]
        areas = widths * heights
        print(f"  Valid bbox widths:  min={widths.min():.4f} max={widths.max():.4f} mean={widths.mean():.4f}")
        print(f"  Valid bbox heights: min={heights.min():.4f} max={heights.max():.4f} mean={heights.mean():.4f}")
        print(f"  Valid bbox areas:   min={areas.min():.4f} max={areas.max():.4f} mean={areas.mean():.4f}")

        # Check for degenerate bboxes
        degenerate = (widths < 0.02) | (heights < 0.02)
        print(f"  Degenerate (w<0.02 or h<0.02): {degenerate.sum().item()}")

        # Check for bboxes near image boundary (clipped)
        near_edge = ((valid_bboxes[:, 0] < 0.01) | (valid_bboxes[:, 1] < 0.01) |
                     (valid_bboxes[:, 2] > 0.99) | (valid_bboxes[:, 3] > 0.99))
        print(f"  Near-edge (clipped): {near_edge.sum().item()}")

    if n_invalid > 0:
        invalid_bboxes = bboxes[~valid]  # [n_invalid, 4]
        print(f"  Fallback bboxes (first 3): {invalid_bboxes[:3].tolist()}")


def hooked_forward_crop(head: HamerManoHead, token_list, images, patch_start_idx,
                         hand_bboxes, hand_valid):
    """Run crop path with intermediate stat printing."""
    B, S = images.shape[:2]
    N = B * S

    tokens = token_list[-1][:, :, patch_start_idx:]
    tokens = tokens.reshape(N, -1, tokens.shape[-1])

    print(f"\n{'='*60}")
    print(f"CROP PATH — Intermediate Features")
    print(f"{'='*60}")
    print_tensor_stats("backbone_tokens (input)", tokens)

    H, W = images.shape[3], images.shape[4]
    ph, pw = H // head.patch_size, W // head.patch_size
    print(f"  Image size: {H}x{W}, Patch grid: {ph}x{pw}")

    feat_map = tokens.permute(0, 2, 1).reshape(N, -1, ph, pw)
    print_tensor_stats("feat_map (2D)", feat_map)

    rois = head._prepare_rois(hand_bboxes, B, S, H, W)
    print_tensor_stats("rois", rois)
    print(f"  spatial_scale = {pw / float(W):.6f}")

    from torchvision.ops import roi_align
    cropped = roi_align(feat_map, rois,
                        output_size=(head.crop_size, head.crop_size),
                        spatial_scale=pw / float(W), aligned=True)
    print_tensor_stats("roi_align output", cropped)

    # Check if valid vs invalid crops differ
    valid_flat = hand_valid.reshape(N * 2)
    if valid_flat.any():
        valid_crops = cropped[valid_flat]
        print_tensor_stats("  valid crops", valid_crops)
    if (~valid_flat).any():
        invalid_crops = cropped[~valid_flat]
        print_tensor_stats("  invalid (fallback) crops", invalid_crops)

    cropped_tokens = cropped.flatten(2).permute(0, 2, 1)
    crop_ctx = head.context_proj(head.context_norm(cropped_tokens))
    print_tensor_stats("crop_ctx (after proj)", crop_ctx)

    global_ctx = head.context_proj(head.context_norm(tokens))
    global_ctx = global_ctx.repeat_interleave(2, dim=0)
    print_tensor_stats("global_ctx (replicated)", global_ctx)

    context = head.crop_to_global(crop_ctx, context=global_ctx)
    print_tensor_stats("fused context (crop_to_global output)", context)

    left_q = head.query_tokens[:, 0:1, :].expand(N, -1, -1)
    right_q = head.query_tokens[:, 1:2, :].expand(N, -1, -1)
    queries = torch.stack([left_q, right_q], dim=1).reshape(N * 2, 1, -1)

    bboxes_norm = hand_bboxes.reshape(N * 2, 4)
    cx = (bboxes_norm[:, 0] + bboxes_norm[:, 2]) * 0.5
    cy = (bboxes_norm[:, 1] + bboxes_norm[:, 3]) * 0.5
    bw = bboxes_norm[:, 2] - bboxes_norm[:, 0]
    bh = bboxes_norm[:, 3] - bboxes_norm[:, 1]
    geom = torch.stack([cx, cy, bw, bh], dim=-1)
    geom_emb = head.geom_proj(geom).unsqueeze(1)
    print_tensor_stats("geom_emb", geom_emb)

    queries = queries + geom_emb
    print_tensor_stats("queries (with geom)", queries)

    out = head.transformer(queries, context=context)
    out = head.output_norm(out)
    print_tensor_stats("transformer output", out)

    out = out.reshape(N, 2, -1)
    left_params = head._decode_hand(out[:, 0, :])
    right_params = head._decode_hand(out[:, 1, :])
    hand_params = torch.cat([left_params, right_params], dim=-1)
    print_tensor_stats("decoded hand_params (before valid mask)", hand_params)

    if hand_valid is not None:
        valid = hand_valid.reshape(N, 2, 1).expand(-1, -1, head.HAND_PARAM_DIM).reshape(N, -1).float()
        hand_params = hand_params * valid
        print_tensor_stats("decoded hand_params (after valid mask)", hand_params)

    return hand_params.reshape(B, S, -1)


def hooked_forward_fullframe(head: HamerManoHead, token_list, images, patch_start_idx):
    """Run full-frame path with intermediate stat printing."""
    B, S = images.shape[:2]
    N = B * S

    tokens = token_list[-1][:, :, patch_start_idx:]
    tokens = tokens.reshape(N, -1, tokens.shape[-1])

    print(f"\n{'='*60}")
    print(f"FULL-FRAME PATH — Intermediate Features")
    print(f"{'='*60}")
    print_tensor_stats("backbone_tokens (input)", tokens)

    context = head.context_proj(head.context_norm(tokens))
    print_tensor_stats("context (after proj)", context)

    query = head.query_tokens.expand(N, -1, -1)
    print_tensor_stats("queries (2 tokens)", query)

    out = head.transformer(query, context=context)
    out = head.output_norm(out)
    print_tensor_stats("transformer output", out)

    left_params = head._decode_hand(out[:, 0, :])
    right_params = head._decode_hand(out[:, 1, :])
    hand_params = torch.cat([left_params, right_params], dim=-1)
    print_tensor_stats("decoded hand_params", hand_params)

    return hand_params.reshape(B, S, -1)


def analyze_loss_masking(preds_crop, preds_full, gt_crop, gt_full, hand_valid):
    """Compare loss with and without hand_valid masking."""
    print(f"\n{'='*60}")
    print(f"LOSS ANALYSIS")
    print(f"{'='*60}")

    # Crop path losses
    loss_crop_unmasked = F.mse_loss(preds_crop, gt_crop).item()

    B, S = hand_valid.shape[:2]
    valid_expanded = hand_valid.unsqueeze(-1).expand(-1, -1, -1, HAND_PARAM_DIM)
    valid_flat = valid_expanded.reshape(B, S, -1).float()
    n_valid_params = valid_flat.sum().item()
    n_total_params = valid_flat.numel()

    if n_valid_params > 0:
        diff_sq = (preds_crop - gt_crop) ** 2
        masked_loss = (diff_sq * valid_flat).sum() / n_valid_params
        print(f"  Crop loss (unmasked MSE, current): {loss_crop_unmasked:.6f}")
        print(f"  Crop loss (masked MSE, valid only): {masked_loss.item():.6f}")
        print(f"  Dilution factor: {n_valid_params/n_total_params:.3f} "
              f"({int(n_valid_params)}/{int(n_total_params)} params valid)")
        print(f"  Effective loss boost from masking: {masked_loss.item()/max(loss_crop_unmasked, 1e-10):.2f}x")
    else:
        print(f"  Crop loss (unmasked): {loss_crop_unmasked:.6f}")
        print(f"  WARNING: No valid hands in batch!")

    # Full-frame path loss
    loss_full = F.mse_loss(preds_full, gt_full).item()
    print(f"  Full-frame loss: {loss_full:.6f}")
    print(f"  NOTE: Full-frame GT is in WORLD space, crop GT is in CAMERA space — not directly comparable!")


def analyze_gt_stats(gt_crop, gt_full, hand_valid):
    """Compare GT statistics between crop (camera-space) and full-frame (world-space)."""
    print(f"\n{'='*60}")
    print(f"GT STATISTICS")
    print(f"{'='*60}")

    for name, gt in [("full-frame (world-space)", gt_full), ("crop (camera-space)", gt_crop)]:
        print(f"\n  {name}:")
        for hand_idx, hand_name in enumerate(("left", "right")):
            off = hand_idx * HAND_PARAM_DIM
            t = gt[..., off:off + 3]
            q = gt[..., off + 3:off + 7]
            nz = t.abs().sum(dim=-1) > 1e-6
            if nz.any():
                tv = t[nz]
                print(f"    {hand_name} t: min={tv.min():.4f} max={tv.max():.4f} "
                      f"mean={tv.mean():.4f} std={tv.std():.4f} (N={nz.sum()})")
                qv = q[nz]
                print(f"    {hand_name} q: min={qv.min():.4f} max={qv.max():.4f} "
                      f"mean={qv.mean():.4f} std={qv.std():.4f}")
            else:
                print(f"    {hand_name}: all-zero (absent)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hand_head.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build CROP model ---
    crop_cfg = {**model_cfg, "use_hand_crop": True}
    crop_cfg.pop("checkpoint", None)
    model_crop = WorldMirror(**crop_cfg)

    # --- Build FULL-FRAME model (share backbone weights) ---
    full_cfg = {**model_cfg, "use_hand_crop": False}
    full_cfg.pop("checkpoint", None)
    model_full = WorldMirror(**full_cfg)

    # Load checkpoint into both
    ckpt = torch.load(model_cfg["checkpoint"], map_location=device)
    sd = ckpt.get("state_dict", ckpt.get("reconstructor", ckpt))
    model_crop.load_state_dict(sd, strict=False)
    model_full.load_state_dict(sd, strict=False)
    model_crop.to(device).eval()
    model_full.to(device).eval()

    # --- Load one batch with hand crops ---
    all_seqs = discover_sequences(data_cfg["data_root"])
    random.seed(42)
    random.shuffle(all_seqs)

    num_frames = data_cfg["num_frames"]
    res = tuple(data_cfg["resolution"])
    rescale_factor = cfg.get("hand_crop", {}).get("rescale_factor", 2.0)

    # Crop dataset (camera-space GT)
    ds_crop = HOT3DHandDataset(
        all_seqs[:3], num_frames=num_frames, res=res,
        use_hand_crop=True, rescale_factor=rescale_factor,
    )
    # Full-frame dataset (world-space GT)
    ds_full = HOT3DHandDataset(
        all_seqs[:3], num_frames=num_frames, res=res,
        use_hand_crop=False,
    )

    if len(ds_crop) == 0 or len(ds_full) == 0:
        print("ERROR: No clips found. Check data_root and sequence paths.")
        return

    loader_crop = DataLoader(ds_crop, batch_size=2, shuffle=False)
    loader_full = DataLoader(ds_full, batch_size=2, shuffle=False)

    batch_crop = next(iter(loader_crop))
    batch_full = next(iter(loader_full))

    imgs = batch_crop["img"].to(device)
    gt_crop = batch_crop["gt"].to(device)
    gt_full = batch_full["gt"].to(device)
    hand_bboxes = batch_crop["hand_bboxes"].to(device)
    hand_valid = batch_crop["hand_valid"].to(device)

    print(f"Batch: B={imgs.shape[0]}, S={imgs.shape[1]}, H={imgs.shape[3]}, W={imgs.shape[4]}")

    # --- Bbox analysis ---
    diagnose_bbox_quality(hand_bboxes, hand_valid)

    # --- GT analysis ---
    analyze_gt_stats(gt_crop, gt_full, hand_valid)

    # --- Run backbone (shared) ---
    with torch.no_grad():
        views = build_views(imgs, num_frames, device, hand_bboxes, hand_valid)
        token_list, patch_start_idx, _, _ = model_crop.visual_geometry_transformer(
            imgs, use_motion=False
        )

        print(f"\nBackbone output: {len(token_list)} levels, "
              f"patch_start_idx={patch_start_idx}")
        for i, tl in enumerate(token_list):
            print_tensor_stats(f"  token_list[{i}]", tl)

        # --- Crop path ---
        preds_crop = hooked_forward_crop(
            model_crop.hand_head, token_list, imgs, patch_start_idx,
            hand_bboxes, hand_valid,
        )

        # --- Full-frame path ---
        preds_full = hooked_forward_fullframe(
            model_full.hand_head, token_list, imgs, patch_start_idx,
        )

        # --- Loss comparison ---
        analyze_loss_masking(preds_crop, preds_full, gt_crop, gt_full, hand_valid)

    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*60}")
    print("""
Key things to check in the output above:
1. BBOX QUALITY: What fraction of hands are valid? Are valid bboxes reasonable sizes?
2. GT STATS: Compare world-space vs camera-space ranges — are they on similar scales?
3. CROP vs FULL-FRAME features: Do roi_align outputs look reasonable? Any NaN/Inf?
4. LOSS DILUTION: How much does masking change the effective loss?
5. VALID vs INVALID crops: Do fallback-bbox crops have wildly different statistics?
""")


if __name__ == "__main__":
    main()
