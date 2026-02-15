import warnings
from collections import OrderedDict
import torch, warnings, os
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import repeat
from typing import Optional, Union
from einops import rearrange
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional
from gsplat.rendering import rasterization
from gsplat.cuda._torch_impl import (
    _fully_fused_projection,
    _quat_scale_to_covar_preci,
)


from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..utils.auxiliary import homo_matrix_inverse, average_filter, fast_perceptual_color_distance, pixel_to_world_coords
from ..models import ModelManager, load_state_dict
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample
from ..models.wan_video_neoverse_controller import NeoVerseControlBranch
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader, LightX2VLoRALoader
from ..data import save_video


class WanVideoNeoVersePipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None, pipeline_kwargs={}):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.control_branch: NeoVerseControlBranch = None
        self.in_iteration_models = ("dit", "control_branch")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_4DPreprocesser(**pipeline_kwargs),
            WanVideoUnit_CameraProcesser(),
            WanVideoUnit_RandomDrop(**pipeline_kwargs),
            WanVideoUnit_4DEmbedder(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_PromptEmbedder(),
        ]
        self.model_fn = model_fn_wan_video
        self.save_root = None


    def load_lora(self, module, path=None, state_dict=None, alpha=1, lora_type="diffsynth"):
        if lora_type == "diffsynth":
            loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        elif lora_type == "lightx2v":
            loader = LightX2VLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        else:
            raise ValueError(f"Unsupported lora_type {lora_type}.")
        if path is not None:
            lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        else:
            lora = state_dict
        loader.load(module, lora, alpha=alpha)


    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)
        timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=self.torch_dtype, device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)

        with torch.amp.autocast("cuda", dtype=self.torch_dtype):
            noise_pred = self.model_fn(**inputs, timestep=timestep)

        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.scheduler.training_weight(timestep)
        return loss


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.control_branch is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.control_branch,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.GroupNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )


    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default
            warnings.warn(
                "Positional args are being deprecated, use kwargs instead. Refer to "
                "https://pytorch.org/docs/master/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
                " for details.")

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        for hook in self._state_dict_pre_hooks.values():
            hook(self, prefix, keep_vars)
        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            # only get the trainable models for pipeline
            if module is not None and name in self.trainable_models:
                module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, prefix, local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination


    @staticmethod
    def from_pretrained(
        local_model_path: str = "models",
        reconstructor_path: str = "models/NeoVerse/reconstructor.ckpt",
        pipeline_kwargs: dict = {},
        lora_path: Optional[str] = None,
        lora_alpha: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        # Initialize pipeline
        pipe = WanVideoNeoVersePipeline(device=device, torch_dtype=torch_dtype, pipeline_kwargs=pipeline_kwargs)

        # Load models
        model_configs = [
            ModelConfig(path=reconstructor_path),
            ModelConfig(local_model_path=local_model_path, model_id="NeoVerse", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(local_model_path=local_model_path, model_id="NeoVerse", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(local_model_path=local_model_path, model_id="NeoVerse", origin_file_pattern="Wan2.1_VAE.pth"),
        ]
        tokenizer_config = ModelConfig(local_model_path=local_model_path, model_id="NeoVerse", origin_file_pattern="google/*")
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary()
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )

        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        pipe.dit = model_manager.fetch_model("wan_video_dit")
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.control_branch = model_manager.fetch_model("wan_video_neoverse_controller")
        pipe.reconstructor = model_manager.fetch_model("reconstructor")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary()
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        # Load Distilled LoRA for faster inference
        if lora_path is not None:
            assert os.path.exists(lora_path), f"LoRA path {lora_path} does not exist."
            pipe.load_lora(pipe.dit, lora_path, alpha=lora_alpha, lora_type="lightx2v")
            print(f"Loaded LoRA from {lora_path}")
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Control
        control_scale: Optional[float] = 1.0,
        source_views: Optional[dict] = None,
        target_rgb: Optional[torch.Tensor] = None,
        target_depth: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        target_poses: Optional[torch.Tensor] = None,
        target_intrs: Optional[torch.Tensor] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        # Inputs
        inputs_posi = {
            "prompt": prompt if isinstance(prompt, list) else [prompt],
        }
        inputs_nega = {
            "negative_prompt": negative_prompt if isinstance(negative_prompt, list) else [negative_prompt],
        }
        inputs_shared = {
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_scale": control_scale, "source_views": source_views,
            "target_rgb": target_rgb, "target_depth": target_depth, "target_mask": target_mask,
            "target_poses": target_poses, "target_intrs": target_intrs,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "sigma_shift": sigma_shift,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])

        # Decode
        self.load_models_to_device(["vae"])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])
        return video


class WanVideoUnit_4DPreprocesser(PipelineUnit):
    def __init__(
        self,
        novel_view_sampling_trans=[0.1, 0.5],
        novel_view_sampling_max_rot=0.0,
        culling_prob=0.3,
        kernel_size_range=[11, 101],
        occlusion_thresh=0.1,
        alpha_thresh=0.5,
        color_thresh=[50, 100],
        **kwargs,
    ):
        super().__init__(
            input_params=("source_views", "target_rgb", "target_depth", "target_mask", "target_poses", "target_intrs"),
            onload_model_names=("reconstructor",)
        )
        self.novel_view_sampling_trans = novel_view_sampling_trans
        self.novel_view_sampling_max_rot = novel_view_sampling_max_rot
        self.culling_prob = culling_prob
        self.kernel_size_range = kernel_size_range
        self.occlusion_thresh = occlusion_thresh
        self.alpha_thresh = alpha_thresh
        self.color_thresh = color_thresh

    def process(self, pipe: WanVideoNeoVersePipeline, source_views, target_rgb, target_depth, target_mask, target_poses, target_intrs):
        if source_views is None:
            return {}

        if isinstance(source_views, list):
            source_views = self.compose_batches_from_list(source_views)
        if pipe.is_training:
            input_video = source_views["img"].clone()
            assert len(input_video) == 1, "During training, only batch size 1 is supported."
            for b_idx in range(len(source_views["img"])):
                order_indices = torch.argsort(source_views["timestamp"][b_idx])
                input_video[b_idx] = input_video[b_idx][order_indices]
        else:
            input_video = None

        if target_rgb is not None and target_depth is not None and target_mask is not None and target_poses is not None and target_intrs is not None:
            return {
                "input_video": input_video,
                "target_rgb": target_rgb,
                "target_depth": target_depth,
                "target_mask": target_mask,
                "target_poses": target_poses,
                "target_intrs": target_intrs,
            }

        pipe.load_models_to_device(self.onload_model_names)
        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            recon_output = pipe.reconstructor(source_views, is_inference=False)
        context_num = (~source_views["is_target"]).sum()
        novel_context_poses = self.novel_view_sampling(
            recon_output["rendered_extrinsics"][:, :context_num],
            recon_output["gs_depth"].squeeze(-1),
        )
        if np.random.rand() < self.culling_prob:
            kernel_size = 0
        else:
            kernel_size = np.random.randint(self.kernel_size_range[0], self.kernel_size_range[1]+1)
        H, W = input_video.shape[-2:]
        splats = self.degradation_simulation(
            recon_output["splats"],
            novel_context_poses,
            recon_output["rendered_intrinsics"][:, :context_num],
            (H, W),
            kernel_size=kernel_size,
            occlusion_thresh=self.occlusion_thresh,
        )
        target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
            splats,
            render_viewmats=homo_matrix_inverse(recon_output["rendered_extrinsics"]),   # c2w -> w2c
            render_Ks=recon_output["rendered_intrinsics"],
            render_timestamps=recon_output["rendered_timestamps"],
            sh_degree=0, width=W, height=H,
        )
        target_mask = target_alpha > self.alpha_thresh

        # Sort renderings with timestamps: convert from "context + non-context" format to temporally ordered frames
        target_poses = recon_output["rendered_extrinsics"]
        target_intrs = recon_output["rendered_intrinsics"]
        for b_idx in range(len(target_rgb)):
            # mask out all non-context frames
            target_mask[b_idx][context_num:] = False
            order_indices = torch.argsort(recon_output["rendered_timestamps"][b_idx])
            target_rgb[b_idx] = target_rgb[b_idx][order_indices]
            target_depth[b_idx] = target_depth[b_idx][order_indices]
            target_mask[b_idx] = target_mask[b_idx][order_indices]
            target_poses[b_idx] = target_poses[b_idx][order_indices]
            target_intrs[b_idx] = target_intrs[b_idx][order_indices]

        if input_video is not None:
            color_mask = fast_perceptual_color_distance(
                rearrange(input_video, "B T C H W -> B T H W C"),
                target_rgb,
            ) < np.random.uniform(self.color_thresh[0], self.color_thresh[1])
            target_mask = target_mask & color_mask.unsqueeze(-1)
        target_mask = target_mask.float()

        if pipe.save_root is not None:
            for_save = rearrange(input_video, "B T C H W -> (B T) H W C")
            for_save = (for_save * 255).clip(0, 255)
            video = [Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy()) for image in for_save]
            output_path = pipe.save_root + "/"
            if "dataset" in source_views:
                output_path += source_views["dataset"][0][0] + "/"
            if "video_name" in source_views:
                output_path += source_views["video_name"][0][0] + "/"
            output_path += "gt.mp4"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            save_video(video, output_path, fps=15)
        return {
            "source_views": source_views,
            "input_video": input_video,
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_mask": target_mask,
            "target_poses": target_poses,
            "target_intrs": target_intrs,
        }

    def compose_batches_from_list(self, batch):
        batched_inputs = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                batched_inputs[key] = torch.stack([b[key] for b in batch], dim=1)
            elif isinstance(batch[0][key], np.ndarray):
                batched_inputs[key] = np.stack([b[key] for b in batch], axis=1)
            elif isinstance(batch[0][key], (int, float, str, bool, list)):
                batched_inputs[key] = [b[key] for b in batch]
            else:
                continue
        return batched_inputs

    def novel_view_sampling(self, poses, depths):
        """
        Generate novel view trajectory by applying random transformations to context frames.
        """
        batch_size = len(poses)
        novel_context_poses_list = []
        for batch_idx in range(batch_size):
            novel_context_poses = self._generate_novel_pose(
                poses[batch_idx], depths[batch_idx]
            )
            novel_context_poses_list.append(novel_context_poses)
        novel_context_poses = torch.stack(novel_context_poses_list, dim=0)  # [B, S_context, 4, 4]
        return novel_context_poses

    def _generate_novel_pose(self, original_poses, depth_maps):
        """
        Generate novel camera poses by applying random transformations to original poses.

        This function creates new viewpoints by translating cameras perpendicular to the
        global trajectory and adding small random rotations while maintaining focus on
        the scene center.
        """
        # Extract original camera components
        N_context = original_poses.shape[0]
        original_pos = original_poses[:, :3, 3]  # [3]
        original_rotation = original_poses[:, :3, :3]  # [3, 3]
        original_forward = original_rotation[..., 2]  # Camera looks along the z-axis

        # Calculate original view center in world coordinates
        valid_depth = depth_maps > 0
        scene_depth = (depth_maps * valid_depth).sum(dim=(1, 2)) / valid_depth.sum(dim=(1, 2)).clamp_min(1e-6)  # [S]
        view_center = original_pos + scene_depth[:, None] * original_forward

        # Generate translation based on perpendicular direction to global trajectory movement
        if N_context > 1:
            # Calculate global movement direction (start to end)
            global_movement_dir = original_pos[-1] - original_pos[0]  # [3]
            global_movement_dir = F.normalize(global_movement_dir.unsqueeze(0), dim=-1)  # [1, 3]
        else:
            # If only one frame, use the original forward direction
            global_movement_dir = original_forward.unsqueeze(0)  # [1, 3]

        # Randomly choose left or right perpendicular direction (consistent for all frames)
        up = torch.tensor([0.0, -1.0, 0.0], device=original_poses.device).unsqueeze(0)  # [1, 3]

        # Calculate perpendicular direction using global movement direction
        # Left: cross(up, movement_dir), Right: cross(movement_dir, up)
        use_left = torch.rand(1, device=original_poses.device).item() > 0.5
        if use_left:
            perpendicular_dir = torch.cross(up, global_movement_dir, dim=-1)  # [1, 3]
        else:
            perpendicular_dir = torch.cross(global_movement_dir, up, dim=-1)  # [1, 3]

        perpendicular_dir = F.normalize(perpendicular_dir, dim=-1)  # [1, 3]

        # Use the same perpendicular direction for all frames
        random_direction = perpendicular_dir.expand(N_context, -1)  # [N_context, 3]
        translation_distance = torch.rand((1, 1), device=original_poses.device) *\
            (self.novel_view_sampling_trans[1] - self.novel_view_sampling_trans[0]) + self.novel_view_sampling_trans[0]
        new_pos = original_pos + random_direction * translation_distance

        # Calculate new forward direction pointing toward view center
        new_forward = F.normalize(view_center - new_pos, dim=-1)

        # Add small random rotation to the forward direction
        angle_rad = torch.deg2rad(torch.tensor(self.novel_view_sampling_max_rot, device=original_poses.device))
        random_rotation_angle = (torch.rand((N_context, 1), device=original_poses.device) - 0.5) * 2 * angle_rad

        # Create random rotation axis perpendicular to forward direction
        random_axis = torch.randn((N_context, 3), device=original_poses.device)
        random_axis = random_axis - torch.sum(random_axis * new_forward, dim=-1, keepdim=True) * new_forward
        random_axis = F.normalize(random_axis, dim=-1)

        # Apply small rotation to forward direction
        cos_angle = torch.cos(random_rotation_angle)
        sin_angle = torch.sin(random_rotation_angle)
        new_forward = (cos_angle * new_forward +
                      sin_angle * torch.cross(random_axis, new_forward, dim=-1))
        new_forward = F.normalize(new_forward, dim=-1)

        # Construct new rotation matrix (assuming up vector is [0,1,0])
        up = torch.tensor([0.0, 1.0, 0.0], device=original_poses.device)[None].repeat(N_context, 1)
        right = torch.cross(up, new_forward, dim=-1)
        right = F.normalize(right, dim=-1)
        new_up = torch.cross(new_forward, right, dim=-1)

        new_rotation = torch.stack([right, new_up, new_forward], dim=-1)

        # Construct new pose matrix
        new_pose = torch.zeros_like(original_poses)
        new_pose[:, :3, :3] = new_rotation
        new_pose[:, :3, 3] = new_pos
        new_pose[:, 3, 3] = 1.0
        return new_pose

    def degradation_simulation(self, gaussians, novel_context_poses, context_intrinsics, image_size_hw, kernel_size, occlusion_thresh=0.1):
        """
        Degradation simulation via visibility-based Gaussian culling and average geometry filter.
        More details can be found in Section 3.2 of [NeoVerse](https://arxiv.org/abs/2601.00393).
        """
        batch_size = len(gaussians)
        novel_context_world2cam = homo_matrix_inverse(novel_context_poses)
        h, w = image_size_hw
        for b_idx in range(batch_size):
            for s_idx in range(len(novel_context_poses[b_idx])):
                cur_gaussian = gaussians[b_idx][s_idx]
                cur_extrinsic = novel_context_world2cam[b_idx][s_idx]
                cur_intrinsic = context_intrinsics[b_idx][s_idx]
                if cur_gaussian.means.shape[0] == 0:
                    continue

                # Project Gaussians to novel views
                covars, _ = _quat_scale_to_covar_preci(
                    cur_gaussian.rotations,
                    cur_gaussian.scales,
                    True,
                    False,
                    triu=False
                )
                radii, means2d, depths, conics, compensations = _fully_fused_projection(
                    cur_gaussian.means,
                    covars,
                    cur_extrinsic[None],
                    cur_intrinsic[None],
                    w, h,
                )
                valid_gs_indices = torch.where((radii[0, :, 0] > 0) & (radii[0, :, 1] > 0))[0]
                if len(valid_gs_indices) == 0:
                    continue
                gs_x = means2d[0, valid_gs_indices, 0].round().long().clamp(0, w - 1)
                gs_y = means2d[0, valid_gs_indices, 1].round().long().clamp(0, h - 1)
                gs_depths = depths[0, valid_gs_indices]

                # Render depth map from the novel view and perform visibility check
                rendered_depths, rendered_alphas, _ = rasterization(
                    means=cur_gaussian.means,
                    quats=cur_gaussian.rotations,
                    scales=cur_gaussian.scales,
                    opacities=cur_gaussian.opacities,
                    colors=cur_gaussian.harmonics,
                    viewmats=cur_extrinsic[None],
                    Ks=cur_intrinsic[None],
                    width=w,
                    height=h,
                    sh_degree=0,
                    packed=False,
                    render_mode="ED",
                )
                rendered_depths = rendered_depths[0, ..., 0]
                visible_mask = (gs_depths < (rendered_depths[gs_y, gs_x] + occlusion_thresh))
                visible_indices = valid_gs_indices[visible_mask]
                if kernel_size == 0:
                    # Visibility-based Gaussian Culling
                    cur_gaussian.keep_indices(visible_indices)
                else:
                    # Average Geometry Filter
                    smoothed_depths = average_filter(rendered_depths, kernel_size=kernel_size)
                    smoothed_gs_depths = smoothed_depths[gs_y[visible_mask], gs_x[visible_mask]]
                    visible_gs_x = gs_x[visible_mask]
                    visible_gs_y = gs_y[visible_mask]

                    # Convert pixel coordinates to world coordinates
                    world_coords = pixel_to_world_coords(
                        visible_gs_x, visible_gs_y, smoothed_gs_depths,
                        cur_intrinsic, cur_extrinsic
                    )
                    cur_gaussian.means[visible_indices] = world_coords
                    cur_gaussian.keep_indices(visible_indices)
        return gaussians


class WanVideoUnit_CameraProcesser(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("target_poses", "target_intrs", "height", "width"),
        )

    def process(self, pipe: WanVideoNeoVersePipeline, target_poses, target_intrs, height, width):
        if target_poses is None:
            return {}
        camera_maps = self.convert_plucker_map(target_poses, target_intrs, height, width)
        return {
            "target_camera_embed": camera_maps,
        }

    def convert_plucker_map(self, poses, intrinsics, height, width):
        batch_size, num_frames = poses.shape[:2]
        device = poses.device
        dtype = poses.dtype

        poses = poses.reshape(batch_size * num_frames, 4, 4)
        intrinsics = intrinsics.reshape(batch_size * num_frames, 3, 3)

        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij"
        )
        # Add a homogeneous coordinate (z=1)
        # Shape: [3, H*W]
        pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=0).reshape(3, -1)
        # Shape: [B*F, 3, H*W]
        pixel_coords = pixel_coords[None].expand(batch_size * num_frames, -1, -1)

        # Transform pixel coordinates to camera space to get ray directions
        # Apply the transformation: K_inv * [x, y, 1]^T
        # (B*F, 3, 3) @ (B*F, 3, H*W) -> (B*F, 3, H*W)
        rays_d_cam = intrinsics.inverse() @ pixel_coords

        # Transform ray directions from camera space to world space
        # Apply rotation: R * d_cam
        # (B*F, 3, 3) @ (B*F, 3, H*W) -> (B*F, 3, H*W)
        rotation_matrices = poses[:, :3, :3] # Shape: [B*F, 3, 3]
        rays_d_world = rotation_matrices @ rays_d_cam
        # Normalize the directions
        rays_d_world = F.normalize(rays_d_world, dim=1)
        rays_d_world = rays_d_world.reshape(batch_size * num_frames, 3, height, width)

        # Get ray origins from camera poses
        rays_o_world = poses[:, :3, 3] # Shape: [B*F, 3]
        # Expand to match the shape of the direction tensor: [B*F, 3, H, W]
        rays_o_world = rays_o_world[..., None, None].expand_as(rays_d_world)

        # Calculate Plücker coordinates (moment vector)
        moment = torch.cross(rays_o_world, rays_d_world, dim=1)

        # Concatenate direction and moment to form the final 6D Plücker embedding
        # Shape: [B*F, 6, H, W]
        plucker_embedding = torch.cat([rays_d_world, moment], dim=1)

        # Return shape: [B, F, 6, H, W]
        plucker_embedding = plucker_embedding.reshape(batch_size, num_frames, 6, height, width)
        return plucker_embedding


class WanVideoUnit_RandomDrop(PipelineUnit):
    def __init__(self, prompt_drop_prob=0.0, mask_drop_prob=0.0, condition_drop_prob=0.0, **kwargs):
        super().__init__(take_over=True)
        self.prompt_drop_prob = prompt_drop_prob
        self.mask_drop_prob = mask_drop_prob
        self.condition_drop_prob = condition_drop_prob

    def process(self, pipe: WanVideoNeoVersePipeline, inputs_shared, inputs_posi, inputs_nega):
        prompt = inputs_posi["prompt"]
        target_rgb = inputs_shared["target_rgb"]
        target_depth = inputs_shared["target_depth"]
        target_camera_embed = inputs_shared["target_camera_embed"]
        target_mask = inputs_shared["target_mask"]
        assert len(prompt) == target_rgb.shape[0] == target_depth.shape[0] == target_camera_embed.shape[0] == target_mask.shape[0], "Batch size must be the same for prompt and target conditions"

        batch_size = target_rgb.shape[0]
        for b_idx in range(batch_size):
            if np.random.rand() < self.prompt_drop_prob:
                prompt[b_idx] = ""
            if np.random.rand() < self.mask_drop_prob:
                target_mask[b_idx] *= 0
            if np.random.rand() < self.condition_drop_prob:
                target_rgb[b_idx] *= 0
                target_depth[b_idx] *= 0
                target_camera_embed[b_idx] *= 0
                target_mask[b_idx] *= 0
        inputs_posi["prompt"] = prompt
        inputs_shared["target_rgb"] = target_rgb
        inputs_shared["target_depth"] = target_depth
        inputs_shared["target_camera_embed"] = target_camera_embed
        inputs_shared["target_mask"] = target_mask
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_4DEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("source_views", "target_rgb", "target_depth", "target_camera_embed", "target_mask",
                          "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoNeoVersePipeline,
        source_views, target_rgb, target_depth, target_camera_embed, target_mask,
        tiled, tile_size, tile_stride
    ):
        if target_rgb is not None:
            batch_size = len(target_rgb)
            try:
                target_d_max = torch.quantile(target_depth.reshape(batch_size, -1), 0.98, dim=-1).clamp_min(1e-6)
            except:
                target_d_max = target_depth.reshape(batch_size, -1).max(dim=-1).values.clamp_min(1e-6)
            d_max = target_d_max[:, None, None, None, None]

            pipe.load_models_to_device(self.onload_model_names)
            target_rgb = rearrange(target_rgb, "B T H W C -> B T C H W")

            if pipe.save_root is not None:
                for_save = rearrange(target_rgb, "B T C H W -> (B T) H W C")
                for_save = (for_save * 255).clip(0, 255)
                video = [Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy()) for image in for_save]
                output_path = pipe.save_root + "/"
                if "dataset" in source_views:
                    output_path += source_views["dataset"][0][0] + "/"
                if "video_name" in source_views:
                    output_path += source_views["video_name"][0][0] + "/"
                output_path += "target_rgb.mp4"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_video(video, output_path, fps=15)

            target_rgb = pipe.preprocess_video(target_rgb)
            target_rgb_latents = pipe.vae.encode(target_rgb, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            target_depth = repeat(target_depth, "B T H W 1 -> B T C H W", C=3)

            if pipe.save_root is not None:
                for_save = rearrange(target_depth, "B T C H W -> (B T) H W C") / d_max[0]
                for_save = (for_save * 255).clip(0, 255)
                video = [Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy()) for image in for_save]
                output_path = pipe.save_root + "/"
                if "dataset" in source_views:
                    output_path += source_views["dataset"][0][0] + "/"
                if "video_name" in source_views:
                    output_path += source_views["video_name"][0][0] + "/"
                output_path += "target_depth.mp4"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_video(video, output_path, fps=15)

            target_depth = pipe.preprocess_video(target_depth, normalize=d_max).clamp(min=-1, max=1)
            target_depth_latents = pipe.vae.encode(target_depth, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)

            if pipe.save_root is not None:
                for_save = rearrange(target_mask, "B T H W C -> (B T) H W C").squeeze(-1)
                for_save = (for_save * 255).clip(0, 255)
                video = [Image.fromarray(image.to(device="cpu", dtype=torch.uint8).numpy()) for image in for_save]
                output_path = pipe.save_root + "/"
                if "dataset" in source_views:
                    output_path += source_views["dataset"][0][0] + "/"
                if "video_name" in source_views:
                    output_path += source_views["video_name"][0][0] + "/"
                output_path += "target_mask.mp4"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                save_video(video, output_path, fps=15)
            return {
                "target_rgb": target_rgb_latents,
                "target_depth": target_depth_latents,
                "target_camera_embed": target_camera_embed.to(dtype=pipe.torch_dtype, device=pipe.device),
                "target_mask": rearrange(target_mask, "B T H W C -> B T C H W").to(dtype=pipe.torch_dtype, device=pipe.device),
            }
        else:
            return {}


class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoNeoVersePipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device"))

    def process(self, pipe: WanVideoNeoVersePipeline, height, width, num_frames, seed, rand_device):
        length = (num_frames - 1) // 4 + 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        return {"noise": noise}



class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoNeoVersePipeline, input_video, noise, tiled, tile_size, tile_stride):
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoNeoVersePipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



def model_fn_wan_video(
    dit: WanModel,
    control_branch: NeoVerseControlBranch = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    control_scale = 1.0,
    target_rgb = None,
    target_depth = None,
    target_camera_embed = None,
    target_mask = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    context = dit.text_embedding(context)

    x = latents

    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    if target_rgb is not None:
        control_hints = control_branch(
            x, target_rgb, target_depth, target_camera_embed, target_mask,
            context, t_mod, freqs, use_gradient_checkpointing, use_gradient_checkpointing_offload,
        )

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

        if target_rgb is not None and block_id in control_branch.control_layers_mapping:
            current_control_hint = control_hints[control_branch.control_layers_mapping[block_id]]
            x = x + current_control_hint * control_scale

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x
