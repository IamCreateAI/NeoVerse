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

MANO_DIM = 44  # 2 hands * (3 t_xyz + 4 q_wxyz + 15 pose)


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
                lines = f.readlines()

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
                        pos  = torch.tensor(hand["wrist_xform"]["t_xyz"],  dtype=torch.float32)  # 3
                        rot  = torch.tensor(hand["wrist_xform"]["q_wxyz"], dtype=torch.float32)  # 4
                        pose = torch.tensor(hand["pose"],                  dtype=torch.float32)  # 15
                        vecs.append(torch.cat([pos, rot, pose]))
                    else:
                        vecs.append(torch.zeros(22))
                gt_per_frame.append(torch.cat(vecs))  # 44

            for start in range(0, total - num_frames + 1, clip_stride):
                self.clips.append({
                    "video_path":   video_path,
                    "gt_frames":    gt_per_frame[start : start + num_frames],
                    "frame_offset": start,
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
        gt   = torch.stack(clip["gt_frames"])                            # [S, 44]
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
        "camera_poses": torch.eye(4).view(1, 1, 4, 4).expand(B, num_frames, 4, 4).to(device),
        "camera_intrs": torch.eye(3).view(1, 1, 3, 3).expand(B, num_frames, 3, 3).to(device),
        "depthmap":     torch.ones((B, num_frames, H, W), device=device),
    }


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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- MODEL ---
    model = WorldMirror(enable_hand=model_cfg["enable_hand"], freeze_backbone=model_cfg["freeze_backbone"])
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

    random.seed(training_cfg.get("seed", 42))
    random.shuffle(all_seqs)
    n_val      = max(1, int(len(all_seqs) * float(data_cfg.get("val_split", 0.1))))
    val_seqs   = all_seqs[:n_val]
    train_seqs = all_seqs[n_val:]

    num_frames  = data_cfg["num_frames"]
    res         = tuple(data_cfg["resolution"])
    clip_stride = data_cfg.get("clip_stride", num_frames)
    batch_size  = training_cfg.get("batch_size", 2)
    num_workers = data_cfg.get("num_workers", 4)

    train_set = HOT3DHandDataset(train_seqs, num_frames=num_frames, res=res, clip_stride=clip_stride)
    val_set   = HOT3DHandDataset(val_seqs,   num_frames=num_frames, res=res, clip_stride=clip_stride)
    print(f"Train clips: {len(train_set)} | Val clips: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, drop_last=False)

    # --- OPTIMIZER & SCHEDULER ---
    epochs     = training_cfg["epochs"]
    optimizer  = Adam(hand_params, lr=float(training_cfg["lr"]))
    scheduler  = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(training_cfg.get("min_lr", 1e-6)))

    log_every  = training_cfg.get("log_every", 5)
    save_every = training_cfg.get("save_every", 10)
    output_dir = training_cfg.get("output_dir", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Training on {device} | {epochs} epochs | batch_size={batch_size}")

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

    epoch_bar = tqdm(range(1, epochs + 1), desc="Epochs")
    for epoch in epoch_bar:
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Train {epoch}", leave=False):
            imgs = batch["img"].to(device)
            gt   = batch["gt"].to(device)
            optimizer.zero_grad()
            preds = model(build_views(imgs, num_frames, device), is_inference=False, use_motion=False)
            loss  = F.mse_loss(preds["hand_joints"], gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        # --- VAL ---
        if epoch % log_every == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Val {epoch}", leave=False):
                    imgs = batch["img"].to(device)
                    gt   = batch["gt"].to(device)
                    preds = model(build_views(imgs, num_frames, device), is_inference=False, use_motion=False)
                    val_loss += F.mse_loss(preds["hand_joints"], gt).item()
            val_loss /= max(len(val_loader), 1)

            lr = scheduler.get_last_lr()[0]
            epoch_bar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", lr=f"{lr:.2e}")
            if use_wandb:
                wandb.log({"train/loss": train_loss, "val/loss": val_loss, "lr": lr}, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.hand_head.state_dict(), os.path.join(output_dir, "hand_head_best.pt"))
                tqdm.write("  -> New best. Saved.")
        else:
            lr = scheduler.get_last_lr()[0]
            epoch_bar.set_postfix(train=f"{train_loss:.6f}", lr=f"{lr:.2e}")
            if use_wandb:
                wandb.log({"train/loss": train_loss, "lr": lr}, step=epoch)

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
