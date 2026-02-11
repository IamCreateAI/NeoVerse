# Modified from VACE
import math
import torch
import torch.nn.functional as F
from .wan_video_dit import DiTBlock
from .utils import hash_state_dict_keys, zero_module

class AttentionBlock(DiTBlock):
    def __init__(self, has_image_input, dim, num_heads, ffn_dim, eps=1e-6, block_id=0):
        super().__init__(has_image_input, dim, num_heads, ffn_dim, eps=eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = torch.nn.Linear(self.dim, self.dim)
        self.after_proj = torch.nn.Linear(self.dim, self.dim)

    def forward(self, c, x, context, t_mod, freqs):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)
        c = super().forward(c, context, t_mod, freqs)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class NeoVerseControlBranch(torch.nn.Module):
    def __init__(
        self,
        control_layers=(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28),
        control_in_dim=96,
        patch_size=(1, 2, 2),
        has_image_input=False,
        dim=1536,
        num_heads=12,
        ffn_dim=8960,
        eps=1e-6,
    ):
        super().__init__()
        self.control_layers = control_layers
        self.control_in_dim = control_in_dim
        self.control_layers_mapping = {i: n for n, i in enumerate(self.control_layers)}

        # control blocks
        self.control_blocks = torch.nn.ModuleList([
            AttentionBlock(has_image_input, dim, num_heads, ffn_dim, eps, block_id=i)
            for i in self.control_layers
        ])

        self.control_mask_padding = [0, 0, 0, 0, 3, 0]
        mask_cam_in_dim = 7
        mask_cam_out_dim = control_in_dim - 32
        # patch embedding for mask and camera plucker embeddings
        self.control_mask_cam_embedding = torch.nn.Sequential(
            torch.nn.Conv3d(mask_cam_in_dim, mask_cam_out_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8)),
            torch.nn.GroupNorm(mask_cam_out_dim // 8, mask_cam_out_dim),
            torch.nn.SiLU(),
        )
        self.control_patch_embedding = torch.nn.Conv3d(control_in_dim, dim, kernel_size=patch_size, stride=patch_size)

    def initialize(self, missing_keys):
        params_dict = dict(self.named_parameters())
        with torch.no_grad():
            for key in missing_keys:
                param = params_dict[key]
                if param.ndim > 1:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif "weight" in key.split(".")[-1]:
                    torch.nn.init.ones_(param)
                else:
                    torch.nn.init.zeros_(param)
            for control_block in self.control_blocks:
                control_block.after_proj = zero_module(control_block.after_proj)

    def forward(
        self,
        x,
        target_rgb_latents,     # [B, C, T, H, W]
        target_depth_latents,   # [B, C, T, H, W]
        target_cams,            # [B, T, C, H, W]
        target_masks,           # [B, T, C, H, W]
        context,
        t_mod,
        freqs,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
    ):
        target_latents = torch.cat((target_rgb_latents, target_depth_latents), dim=1)
        target_mask_cams = torch.cat((target_masks, target_cams), dim=2).permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
        target_mask_cams = F.pad(target_mask_cams, self.control_mask_padding, mode="constant", value=0)
        target_mask_cams = self.control_mask_cam_embedding(target_mask_cams)
        c = self.control_patch_embedding(torch.cat((target_latents, target_mask_cams), dim=1))
        c = c.flatten(2).transpose(1, 2)
        c = torch.cat((c, c.new_zeros(c.shape[0], x.shape[1] - c.shape[1], c.shape[2])), dim=1)

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.control_blocks:
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    c = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        c, x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                c = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    c, x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                c = block(c, x, context, t_mod, freqs)
        hints = torch.unbind(c)[:-1]
        return hints

    @staticmethod
    def state_dict_converter():
        return NeoVerseControlBranchDictConverter()


class NeoVerseControlBranchDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict_ = {name: param for name, param in state_dict.items() if name.startswith("control")}
        if hash_state_dict_keys(state_dict_) == '45cf2d04f7f77286df2bf0e723a36e03':
            config = {
                "control_layers": (0, 5, 10, 15, 20, 25, 30, 35),
                "control_in_dim": 96,
                "patch_size": (1, 2, 2),
                "has_image_input": False,
                "dim": 5120,
                "num_heads": 40,
                "ffn_dim": 13824,
                "eps": 1e-06,
            }
        return state_dict_, config
