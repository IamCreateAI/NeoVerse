import os
import json
import random
import yaml
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader
from diffsynth.utils.auxiliary import load_video
from torchvision.transforms import functional as TVF
import argparse
from tqdm import tqdm
import wandb

from diffsynth.auxiliary_models.worldmirror.models.models.worldmirror import WorldMirror


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------
class HOT3DHandDataset(Dataset):
    """Sliding-window clips over a list of sequences."""

    def __init__(self, seq_dirs, num_frames=16, res=(224, 224), clip_stride=None):
        self.num_frames = num_frames
        self.res = res
        self.clips = []  # list of {video_path, gt_frames, frame_offset}

        if clip_stride is None:
            clip_stride = num_frames

        for seq_path in seq_dirs:
            video_path = os.path.join(seq_path, "video_main_rgb.mp4")
            jsonl_path = os.path.join(seq_path, "hand_data/mano_hand_pose_trajectory.jsonl")

            if not os.path.exists(video_path) or not os.path.exists(jsonl_path):
                print(f"Skipping {seq_path} because it doesn't have a video or jsonl file")
                continue

            with open(jsonl_path) as f:
                lines = list(f)

            n_video = len(VideoReader(video_path))
            total = min(len(lines), n_video)
            if total < num_frames:
                continue

            gt_per_frame = []
            for line in lines[:total]:
                data = json.loads(line)
                vecs = []
                for hand_id in ["0", "1"]:
                    hand = data["hand_poses"].get(hand_id, {})
                    if hand:
                        pos   = torch.tensor(hand["wrist_xform"]["t_xyz"],  dtype=torch.float32)  # 3
                        rot   = torch.tensor(hand["wrist_xform"]["q_wxyz"], dtype=torch.float32)  # 4
                        pose  = torch.tensor(hand["pose"],                  dtype=torch.float32)  # 15
                        betas = torch.tensor(hand["betas"],                 dtype=torch.float32)  # 10
                        vecs.append(torch.cat([pos, rot, pose, betas]))
                    else:
                        vecs.append(torch.zeros(32))
                gt_per_frame.append(torch.cat(vecs))  # 64

            for start in range(0, total - num_frames + 1, clip_stride):
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
        imgs = torch.stack([TVF.to_tensor(img) for img in pil_images])  # [S, 3, H, W]
        gt   = torch.stack(clip["gt_frames"])                            # [S, 64]
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


def run_validation(model, val_loader, num_frames, device, vis_clip_indices=None):
    """Run validation and optionally capture gt/pred at specific clip indices.

    Args:
        vis_clip_indices: set of clip indices to capture data for (or None)

    Returns:
        avg_loss, captured_data (dict: clip_idx -> {gt, pred}) if vis_clip_indices else avg_loss
    """
    model.eval()
    val_loss = 0.0
    captured = {}
    batch_size = val_loader.batch_size
    with torch.no_grad():
        for batch_idx, vbatch in enumerate(tqdm(val_loader, desc="Val", leave=False)):
            imgs = vbatch["img"].to(device)
            gt   = vbatch["gt"].to(device)
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

    avg_loss = val_loss / max(len(val_loader), 1)
    if vis_clip_indices is not None:
        return avg_loss, captured
    return avg_loss


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

    # --- MODEL ---
    model = WorldMirror(**{k: v for k, v in model_cfg.items() if k != "checkpoint"})
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint.get("reconstructor", checkpoint))
    missing, _ = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. New (hand head) keys: {len(missing)}")
    model.to(device)

    hand_params = list(model.hand_head.parameters())
    print(f"Hand head parameters: {sum(p.numel() for p in hand_params):,}")

    # --- DATA ---
    all_seqs = discover_sequences(data_cfg["data_root"])
    if not all_seqs:
        raise RuntimeError(f"No sequences found in {data_cfg['data_root']}")
    print(f"Found {len(all_seqs)} sequences")

    if debug_cfg.get("enabled", False):
        max_seqs = debug_cfg.get("max_sequences", 5)
        all_seqs = all_seqs[:max_seqs]
        print(f"[DEBUG] Limited to {len(all_seqs)} sequences")

    random.seed(training_cfg.get("seed", 42))
    random.shuffle(all_seqs)
    n_val      = max(1, int(len(all_seqs) * float(data_cfg.get("val_split", 0.1))))
    val_seqs   = all_seqs[:n_val]
    train_seqs = all_seqs[n_val:]

    num_frames       = data_cfg["num_frames"]
    res              = tuple(data_cfg["resolution"])
    clip_stride      = data_cfg.get("clip_stride", num_frames)
    batch_size       = training_cfg.get("batch_size", 2)
    grad_accum_steps = training_cfg.get("grad_accum_steps", 1)
    num_workers      = data_cfg.get("num_workers", 4)

    train_set = HOT3DHandDataset(train_seqs, num_frames=num_frames, res=res, clip_stride=clip_stride)
    val_set   = HOT3DHandDataset(val_seqs,   num_frames=num_frames, res=res, clip_stride=clip_stride)
    print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)

    # --- VISUALIZATION SETUP ---
    vis_entries = []       # val: list of (clip_idx, vis_context)
    train_vis_items = []   # train: list of {img, gt, ctx}
    vis_cfg = cfg.get("visualization", {})
    mano_folder = vis_cfg.get("mano_model_folder")
    num_vis_frames = vis_cfg.get("num_vis_frames", 4)
    if mano_folder and (len(val_set.clips) > 0 or len(train_set.clips) > 0):
        from scripts.hand_vis_utils import setup_vis_context, render_hand_comparison, MANOModel
        mano_model = MANOModel(mano_folder)
        seq_cache = {}

        # Val vis clips (equally spaced)
        if len(val_set.clips) > 0:
            n_clips = len(val_set.clips)
            val_vis_indices = [int(i * (n_clips - 1) / max(num_vis_frames - 1, 1)) for i in range(num_vis_frames)]
            val_vis_indices = sorted(set(val_vis_indices))
            for clip_idx in val_vis_indices:
                clip = val_set.clips[clip_idx]
                seq_path = clip["seq_path"]
                if seq_path not in seq_cache:
                    seq_cache[seq_path] = setup_vis_context(seq_path, mano_model=mano_model)
                ctx = seq_cache[seq_path]
                if ctx is not None:
                    vis_entries.append((clip_idx, {**ctx, "frame_offset": clip["frame_offset"]}))

        # Train vis clips (equally spaced, pre-loaded)
        if len(train_set.clips) > 0:
            n_clips = len(train_set.clips)
            train_vis_indices = [int(i * (n_clips - 1) / max(num_vis_frames - 1, 1)) for i in range(num_vis_frames)]
            train_vis_indices = sorted(set(train_vis_indices))
            for clip_idx in train_vis_indices:
                clip = train_set.clips[clip_idx]
                seq_path = clip["seq_path"]
                if seq_path not in seq_cache:
                    seq_cache[seq_path] = setup_vis_context(seq_path, mano_model=mano_model)
                ctx = seq_cache[seq_path]
                if ctx is not None:
                    item = train_set[clip_idx]
                    train_vis_items.append({
                        "img": item["img"],
                        "gt": item["gt"],
                        "ctx": {**ctx, "frame_offset": clip["frame_offset"]},
                    })

        if vis_entries or train_vis_items:
            print(f"[VIS] Enabled: {len(vis_entries)} val + {len(train_vis_items)} train frames across {len(seq_cache)} sequences")
    vis_clip_indices = {idx for idx, _ in vis_entries} if vis_entries else None

    # --- OPTIMIZER & SCHEDULER ---
    epochs     = training_cfg["epochs"]
    optimizer  = Adam(hand_params, lr=float(training_cfg["lr"]))
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training_cfg.get("min_lr", 1e-6)))

    log_every  = training_cfg.get("log_every", 500)   # steps
    val_every  = training_cfg.get("val_every", 2000)  # steps
    save_every = training_cfg.get("save_every", 10)   # epochs
    output_dir = training_cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training on {device} | {epochs} epochs | batch_size={batch_size} | grad_accum_steps={grad_accum_steps}")

    # --- WANDB ---
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

    def log_validation(step):
        """Run validation, render vis frames, and log to wandb."""
        if vis_entries:
            val_loss, captured = run_validation(model, val_loader, num_frames, device, vis_clip_indices=vis_clip_indices)
        else:
            val_loss = run_validation(model, val_loader, num_frames, device)
            captured = {}
        tqdm.write(f"  step {step} | val_loss={val_loss:.6f}")
        if use_wandb:
            log_dict = {"val/loss": val_loss}
            val_images = []
            for i, (clip_idx, ctx) in enumerate(vis_entries):
                if clip_idx in captured:
                    vis_img = render_hand_comparison(
                        ctx, ctx["frame_offset"],
                        captured[clip_idx]["gt"], captured[clip_idx]["pred"],
                    )
                    if vis_img is not None:
                        val_images.append(wandb.Image(vis_img, caption=f"Frame {i}: Solid=GT, Wireframe=Pred"))
            if val_images:
                log_dict["val/hand_overlay"] = val_images
            wandb.log(log_dict, step=step)
        return val_loss

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        # --- TRAIN ---
        model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}", leave=False)):
            imgs = batch["img"].to(device)
            gt   = batch["gt"].to(device)
            preds = model(build_views(imgs, num_frames, device), is_inference=False, use_motion=False)
            loss  = F.mse_loss(preds["hand_joints"], gt)
            (loss / grad_accum_steps).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0 or global_step == 1:
                    lr = scheduler.get_last_lr()[0]
                    tqdm.write(f"  step {global_step} | train_loss={loss.item():.6f} | lr={lr:.2e}")
                    if use_wandb:
                        log_dict = {"train/loss": loss.item(), "lr": lr}
                        if train_vis_items:
                            model.eval()
                            train_images = []
                            with torch.no_grad():
                                vis_imgs = torch.stack([it["img"] for it in train_vis_items]).to(device)
                                vis_preds = model(build_views(vis_imgs, num_frames, device), is_inference=False, use_motion=False)
                                for i, item in enumerate(train_vis_items):
                                    vis_img = render_hand_comparison(
                                        item["ctx"], item["ctx"]["frame_offset"],
                                        item["gt"][0], vis_preds["hand_joints"][i, 0].cpu(),
                                    )
                                    if vis_img is not None:
                                        train_images.append(wandb.Image(vis_img, caption=f"Train {i}: Solid=GT, Wireframe=Pred"))
                            if train_images:
                                log_dict["train/hand_overlay"] = train_images
                            model.train()
                        wandb.log(log_dict, step=global_step)

                if global_step % val_every == 0 or global_step == 1:
                    val_loss = log_validation(global_step)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.hand_head.state_dict(), os.path.join(output_dir, "hand_head_best.pt"))
                        tqdm.write("  -> New best. Saved.")
                    model.train()

        scheduler.step()

        if epoch % save_every == 0:
            torch.save(model.hand_head.state_dict(), os.path.join(output_dir, f"hand_head_epoch{epoch:04d}.pt"))

    # --- SAVE FINAL ---
    final = training_cfg.get("output_weights", os.path.join(output_dir, "hand_head_final.pt"))
    torch.save(model.hand_head.state_dict(), final)
    print(f"Final weights saved to: {final}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
