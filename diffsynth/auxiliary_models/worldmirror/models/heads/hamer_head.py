"""
Hamer-style cross-attention transformer decoder head for MANO parameter regression.

Uses a single learned query token that cross-attends to backbone spatial features to directly regress MANO parameters.
"""

import torch
import torch.nn as nn
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, context=None):
        context = context if context is not None else x
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class TransformerCrossAttn(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, context_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout),
            ]))

    def forward(self, x, context=None):
        for sa_norm, sa, ca_norm, ca, ff_norm, ff in self.layers:
            x = sa(sa_norm(x)) + x
            x = ca(ca_norm(x), context=context) + x
            x = ff(ff_norm(x)) + x
        return x


class HamerManoHead(nn.Module):
    """Cross-attention transformer decoder for MANO parameter regression.

    Note: The original HaMeR head only predicts right hands and mirrors to predict left hand.
    We modify it to predict both hands by adding a second query token to avoid having to do two forward passes only for the hand head.
    """

    # Per-hand layout: [pos(3) + rot(4) + pose(15)] = 22, [betas(10)] = 10
    POSE_DIM = 22
    BETAS_DIM = 10
    HAND_PARAM_DIM = 32  # POSE_DIM + BETAS_DIM

    def __init__(
        self,
        context_dim=2048,
        dim=1024,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.0,
    ):
        super().__init__()
        self.context_norm = nn.LayerNorm(context_dim)
        self.context_proj = nn.Linear(context_dim, dim)

        # Two learned query tokens: one for left hand, one for right hand.
        # Note: Original HaMeR only predicts right hands and mirrors to predict left hand
        self.query_tokens = nn.Parameter(torch.randn(1, 2, dim))

        self.transformer = TransformerCrossAttn(
            dim, depth, heads, dim_head, mlp_dim, dropout=dropout, context_dim=dim,
        )
        self.output_norm = nn.LayerNorm(dim)

        # Separate projection heads for pose vs shape (shared across hands)
        self.dec_pose = nn.Linear(dim, self.POSE_DIM)
        self.dec_betas = nn.Linear(dim, self.BETAS_DIM)
        self.head_conf = nn.Linear(dim, 1)

        nn.init.xavier_uniform_(self.dec_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dec_betas.weight, gain=0.01)

    def _decode_hand(self, token):
        """Decode a single hand's MANO parameters from a transformer output token."""
        return torch.cat([self.dec_pose(token), self.dec_betas(token)], dim=-1)

    def forward(self, token_list, images, patch_start_idx):
        B, S = images.shape[:2]

        # Extract patch tokens from deepest backbone layer
        tokens = token_list[-1][:, :, patch_start_idx:]  # [B, S, N_patches, C]

        # Fold sequence into batch for per-frame processing
        tokens = tokens.reshape(B * S, -1, tokens.shape[-1])  # [B*S, N_patches, C]

        # Project context to transformer dimension
        context = self.context_proj(self.context_norm(tokens))  # [B*S, N_patches, dim]

        # Expand both query tokens for full batch
        query = self.query_tokens.expand(B * S, -1, -1)  # [B*S, 2, dim]

        # Cross-attention transformer decoder
        out = self.transformer(query, context=context)  # [B*S, 2, dim]
        out = self.output_norm(out)

        # Decode each hand from its respective query token
        left_params = self._decode_hand(out[:, 0, :])   # [B*S, 32]
        right_params = self._decode_hand(out[:, 1, :])   # [B*S, 32]
        hand_params = torch.cat([left_params, right_params], dim=-1)  # [B*S, 64]

        # Confidence from mean of both tokens
        confidence = self.head_conf(out.mean(dim=1))  # [B*S, 1]

        # Reshape back to [B, S, ...]
        hand_params = hand_params.reshape(B, S, -1)  # [B, S, 64]
        confidence = confidence.reshape(B, S, -1)  # [B, S, 1]

        return hand_params, confidence
