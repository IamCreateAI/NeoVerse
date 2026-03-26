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

import torch
import torch.nn.functional as F
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

HAND_PARAM_DIM = 32  # per hand: pos(3) + rot(4) + pose(15) + betas(10)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class HOT3DHandDataset(Dataset):
    """Sliding-window clips over a list of sequences."""

    def __init__(self, seq_dirs, num_frames=16, res=(224, 224), clip_stride=None):
        self.num_frames = num_frames
        self.res = res
        self.clips = []

        if clip_stride is None:
            clip_stride = num_frames

        for seq_path in seq_dirs:
            video_path = os.path.join(seq_path, "video_main_rgb.mp4")
            jsonl_path = os.path.join(seq_path, "hand_data/mano_hand_pose_trajectory.jsonl")

            if not os.path.exists(video_path) or not os.path.exists(jsonl_path):
                print(f"Skipping {seq_path} because it doesn't have a video or jsonl file")
                continue

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

            for start in range(0, n_video - num_frames + 1, clip_stride):
                self.clips.append({
                    "video_path":   video_path,
                    "gt_frames":    gt_per_frame[start : start + num_frames],
                    "frame_offset": start,
                    "seq_path":     seq_path,
                })

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
        gt = torch.stack(clip["gt_frames"])
        return {"img": imgs, "gt": gt}


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

def build_views(imgs, num_frames, device):
    B, _, _, H, W = imgs.shape
    return {
        "img":          imgs,
        "is_target":    torch.zeros((B, num_frames), dtype=torch.bool, device=device),
        "timestamp":    torch.arange(num_frames, device=device).unsqueeze(0).expand(B, -1),
        "is_static":    torch.zeros((B, num_frames), dtype=torch.bool, device=device),
        "valid_mask":   torch.ones((B, num_frames, H, W), dtype=torch.bool, device=device),
        "camera_poses": torch.eye(4, device=device).view(1, 1, 4, 4).expand(B, num_frames, 4, 4),
        "camera_intrs": torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, num_frames, 3, 3),
        "depthmap":     torch.ones((B, num_frames, H, W), device=device),
    }


# ------------------------------------------------------------------
# Visualization helpers
# ------------------------------------------------------------------

def setup_vis_items(dataset, num_vis_frames, seq_cache, mano_model, preload=False):
    """Set up visualization entries for a dataset.

    Args:
        dataset: HOT3DHandDataset
        preload: if True, also load img/gt tensors (for train vis)

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
            "ctx": {**ctx, "frame_offset": clip["frame_offset"]},
        }
        if preload:
            data = dataset[clip_idx]
            entry["img"] = data["img"]
            entry["gt"] = data["gt"]
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

def run_validation(model, val_loader, num_frames, device, vis_clip_indices=None):
    """Run validation and optionally capture gt/pred at specific clip indices."""
    model.eval()
    val_loss = 0.0
    captured = {}
    batch_size = val_loader.batch_size
    with torch.no_grad():
        for batch_idx, vbatch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
            imgs = vbatch["img"].to(device)
            gt = vbatch["gt"].to(device)
            preds = model(build_views(imgs, num_frames, device), is_inference=False, use_motion=False)
            val_loss += F.mse_loss(preds["hand_joints"], gt).item()

            if vis_clip_indices:
                for item_idx in range(imgs.shape[0]):
                    clip_idx = batch_idx * batch_size + item_idx
                    if clip_idx in vis_clip_indices:
                        captured[clip_idx] = {
                            "gt": gt[item_idx, 0].cpu(),
                            "pred": preds["hand_joints"][item_idx, 0].cpu(),
                        }

    return val_loss / max(len(val_loader), 1), captured


def render_train_vis(model, train_vis_items, num_frames, device, render_fn):
    """Forward-pass fixed train clips and render visualizations."""
    model.eval()
    with torch.no_grad():
        imgs = torch.stack([it["img"] for it in train_vis_items]).to(device)
        preds = model(build_views(imgs, num_frames, device), is_inference=False, use_motion=False)
        pairs = [
            (item["gt"][0], preds["hand_joints"][i, 0].cpu())
            for i, item in enumerate(train_vis_items)
        ]
    return render_vis_list(train_vis_items, pairs, render_fn)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_hand_head.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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

    hand_params = list(model.hand_head.parameters())
    print(f"Hand head parameters: {sum(p.numel() for p in hand_params):,}")

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

    if debug_cfg.get("single_frame", False):
        # Overfit on a single clip from the middle of the first sequence
        single_set = HOT3DHandDataset(all_seqs[:1], num_frames=num_frames, res=res, clip_stride=clip_stride)
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
        train_set = HOT3DHandDataset(train_seqs, num_frames=num_frames, res=res, clip_stride=clip_stride)
        val_set = HOT3DHandDataset(val_seqs, num_frames=num_frames, res=res, clip_stride=clip_stride) if val_seqs else None

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
    vis_cfg = cfg.get("visualization", {})
    mano_folder = vis_cfg.get("mano_model_folder")
    num_vis_frames = vis_cfg.get("num_vis_frames", 4)
    render_fn = None
    val_vis_items = []
    train_vis_items = []

    has_val_clips = val_set is not None and len(val_set.clips) > 0
    if mano_folder and (has_val_clips or len(train_set.clips) > 0):
        from scripts.hand_vis_utils import MANOModel, render_hand_comparison
        render_fn = render_hand_comparison
        mano_model = MANOModel(mano_folder)
        seq_cache = {}

        if has_val_clips:
            val_vis_items = setup_vis_items(val_set, num_vis_frames, seq_cache, mano_model)
        train_vis_items = setup_vis_items(train_set, num_vis_frames, seq_cache, mano_model, preload=True)

        if val_vis_items or train_vis_items:
            print(f"[VIS] {len(val_vis_items)} val + {len(train_vis_items)} train frames across {len(seq_cache)} sequences")

    val_vis_clip_indices = {it["clip_idx"] for it in val_vis_items} or None

    # --- Optimizer & scheduler ---
    epochs     = training_cfg["epochs"]
    optimizer  = Adam(hand_params, lr=float(training_cfg["lr"]))
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training_cfg.get("min_lr", 1e-6)))

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
            config={**data_cfg, **model_cfg, **training_cfg},
        )

    best_val_loss = float("inf")
    global_step = 0

    # --- Training loop ---
    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}", leave=False)):
            imgs = batch["img"].to(device)
            gt = batch["gt"].to(device)
            preds = model(build_views(imgs, num_frames, device), is_inference=False, use_motion=False)
            loss = F.mse_loss(preds["hand_joints"], gt)
            (loss / grad_accum_steps).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # --- Train logging ---
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "lr": scheduler.get_last_lr()[0]}, step=global_step)

                if global_step % log_every == 0 or global_step == 1:
                    lr = scheduler.get_last_lr()[0]
                    tqdm.write(f"  step {global_step} | train_loss={loss.item():.6f} | lr={lr:.2e}")
                    if use_wandb and train_vis_items:
                        train_images = render_train_vis(model, train_vis_items, num_frames, device, render_fn)
                        if train_images:
                            wandb.log({"train/hand_overlay": train_images}, step=global_step)
                        model.train()

                # --- Validation ---
                if val_loader and (global_step % val_every == 0 or global_step == 1):
                    val_loss, captured = run_validation(model, val_loader, num_frames, device, val_vis_clip_indices)
                    tqdm.write(f"  step {global_step} | val_loss={val_loss:.6f}")

                    if use_wandb:
                        log_dict = {"val/loss": val_loss}
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
                        torch.save(model.hand_head.state_dict(), os.path.join(output_dir, "best_val_loss.pt"))
                        tqdm.write("  -> New best. Saved.")
                    model.train()

                if global_step % save_every == 0:
                    torch.save(model.hand_head.state_dict(), os.path.join(output_dir, f"checkpoint_{global_step}.pt"))

        scheduler.step()

    # --- Save final ---
    torch.save(model.hand_head.state_dict(), os.path.join(output_dir, "hand_head_final.pt"))
    print(f"Final weights saved to: {os.path.join(output_dir, 'hand_head_final.pt')}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
