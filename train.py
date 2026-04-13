import torch, os
import argparse
from omegaconf import OmegaConf

from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline, ModelConfig
from training.utils import DiffusionTrainingModule, launch_training_task
from training.data.datasets.spatialvid import SpatialVID
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_path="models",
        reconstructor_path="models/NeoVerse/reconstructor.ckpt",
        pipeline_kwargs={},
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_exclude_modules=None, lora_rank=32,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Load models
        self.pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=model_path,
            reconstructor_path=reconstructor_path,
            pipeline_kwargs=pipeline_kwargs,
            device="cpu",
            torch_dtype=torch.bfloat16,
        )

        # Reset training scheduler
        self.pipe.scheduler.set_timesteps(1000, training=True)

        # Add LoRA to the base models
        if lora_base_model is not None:
            if lora_exclude_modules is not None:
                lora_exclude_modules = lora_exclude_modules.split(",")
                if len(lora_exclude_modules) == 1:
                    lora_exclude_modules = lora_exclude_modules[0]

            model = self.add_lora_to_model(
                getattr(self.pipe, lora_base_model),
                target_modules=lora_target_modules.split(","),
                exclude_modules=lora_exclude_modules,
                lora_rank=lora_rank
            )
            setattr(self.pipe, lora_base_model, model)

        # Freeze untrainable models
        self.pipe.freeze_except([] if trainable_models is None else trainable_models.split(","), lora_base_model)

        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary


    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data[0]["prompt"]}
        inputs_nega = {}

        # CFG-unsensitive parameters
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": None,
            "height": data[0]["img"].shape[-2],
            "width": data[0]["img"].shape[-1],
            "num_frames": len(data),
            "source_views": data,
            "control_scale": 1,
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }

        # Pipeline units will automatically process the input parameters.
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return {**inputs_shared, **inputs_posi}


    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.debug:
        config.num_workers = 0
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()
    args = config

    print(f"Preparing dataset {args.train_dataset}")
    dataset = eval(args.train_dataset)
    model = WanTrainingModule(
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        pipeline_kwargs=args.pipeline_kwargs,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_exclude_modules=args.lora_exclude_modules,
        lora_rank=args.lora_rank,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    launch_training_task(
        dataset, model, optimizer, scheduler, args
    )
