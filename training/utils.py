import os, torch, re, shutil
from omegaconf import OmegaConf
import numpy as np
import random
import time
import datetime
from collections import defaultdict, deque
from peft import LoraConfig, inject_adapter_in_model
import torch.distributed as dist
from functools import partial
import logging
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs
from accelerate.logging import get_logger
from torch.utils.tensorboard import SummaryWriter

printer = get_logger(__name__)


class DiffusionTrainingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def to(self, *args, **kwargs):
        for name, model in self.named_children():
            model.to(*args, **kwargs)
        return self


    def trainable_modules(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.parameters())
        return trainable_modules


    def trainable_param_names(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names


    def add_lora_to_model(self, model, target_modules, lora_rank, lora_alpha=None, exclude_modules=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules, exclude_modules=exclude_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model


    def export_trainable_state_dict(self, state_dict, remove_prefix=None):
        trainable_param_names = self.trainable_param_names()
        state_dict = {name: param for name, param in state_dict.items() if name in trainable_param_names}
        if remove_prefix is not None:
            state_dict_ = {}
            for name, param in state_dict.items():
                if name.startswith(remove_prefix):
                    name = name[len(remove_prefix):]
                state_dict_[name] = param
            state_dict = state_dict_
        return state_dict



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, accelerator: Accelerator):
        """Synchronize the count and total across all processes."""
        if accelerator.num_processes == 1:
            return
        t = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device=accelerator.device
        )
        accelerator.wait_for_everyone()
        accelerator.reduce(t, reduction="sum")
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.ndim > 0:
                    continue
                v = v.item()
            if isinstance(v, list):
                continue
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, accelerator):
        for meter in self.meters.values():
            meter.synchronize_between_processes(accelerator)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(
        self, iterable, print_freq, accelerator: Accelerator, header=None, max_iter=None
    ):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        len_iterable = min(len(iterable), max_iter) if max_iter else len(iterable)
        space_fmt = ":" + str(len(str(len_iterable))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for it, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if accelerator.is_main_process:
                        printer.info(
                            log_msg.format(
                                i,
                                len_iterable,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                                memory=torch.cuda.max_memory_allocated() / MB,
                            )
                        )
                else:
                    if accelerator.is_main_process:
                        printer.info(
                            log_msg.format(
                                i,
                                len_iterable,
                                eta=eta_string,
                                meters=str(self),
                                time=str(iter_time),
                                data=str(data_time),
                            )
                        )
            i += 1
            end = time.time()
            if max_iter and it >= max_iter:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if accelerator.is_main_process:
            printer.info(
                "{} Total time: {} ({:.4f} s / it)".format(
                    header, total_time_str, total_time / len_iterable
                )
            )

    def save(self, accelerator, model, epoch_id, iter_id=None):
        if iter_id is not None:
            name = f"checkpoint-epoch-{epoch_id}-iter-{iter_id}"
        else:
            name = f"checkpoint-epoch-{epoch_id}"
        checkpoint_path = os.path.join(self.output_path, name)
        # accelerator.save_state(checkpoint_path)
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            accelerator.save(state_dict, checkpoint_path + ".safetensors", safe_serialization=True)


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter

    def on_step_end(self, loss):
        pass

    def on_epoch_end(self, accelerator, model, epoch_id):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state_dict = accelerator.get_state_dict(model)
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    ignore_pattern = shutil.ignore_patterns(
        "debug*",
        ".vscode*",
        "assets*",
        "example*",
        "checkpoints*",
        "OLD*",
        "logs*",
        "out*",
        "runs*",
        "*.png",
        "*.mp4",
        "*__pycache__*",
        "*.git*",
        "*.idea*",
        "*.zip",
        "*.jpg",
    )
    for src_dir in ["training", "diffsynth"]:
        shutil.copytree(
            src_dir,
            os.path.join(dst_dir, src_dir),
            ignore=ignore_pattern,
            dirs_exist_ok=True,
        )
    return dst_dir


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.

    Returns:
        bool: True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """
    Get the rank of the current process in distributed training.

    Returns:
        int: The rank of the current process, or 0 if distributed training is not initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Get the total number of processes in distributed training.

    Returns:
        int: The world size, or 1 if distributed training is not initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def default_worker_init_fn(worker_id, num_workers, epoch, seed=0):
    """
    Default function to initialize random seeds for dataloader workers.

    Ensures that each worker across different ranks, epochs, and world sizes
    gets a unique random seed for reproducibility.

    Args:
        worker_id (int): ID of the dataloader worker.
        num_workers (int): Total number of dataloader workers.
        epoch (int): Current training epoch.
        seed (int, optional): Base seed for randomization. Defaults to 0.
    """
    rank = get_rank()
    world_size = get_world_size()

    # Use prime numbers for better distribution
    RANK_MULTIPLIER = 1
    WORKER_MULTIPLIER = 1
    WORLD_MULTIPLIER = 1
    EPOCH_MULTIPLIER = 12345

    worker_seed = (
        rank * num_workers * RANK_MULTIPLIER +
        worker_id * WORKER_MULTIPLIER +
        seed +
        world_size * WORLD_MULTIPLIER +
        epoch * EPOCH_MULTIPLIER
    )

    torch.random.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return


def get_worker_init_fn(seed, num_workers, epoch=0, worker_init_fn=None):
    """
    Get a worker initialization function for dataloaders.

    Args:
        seed (int): Base seed for randomization.
        num_workers (int): Number of dataloader workers.
        epoch (int): Current training epoch.
        worker_init_fn (callable, optional): Custom worker initialization function.
            If provided, this will be returned instead of the default one.

    Returns:
        callable: A worker initialization function to use with DataLoader.
    """
    if worker_init_fn is not None:
        return worker_init_fn

    return partial(
        default_worker_init_fn,
        num_workers=num_workers,
        epoch=epoch,
        seed=seed,
    )


def launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    args,
):
    accumu_steps = args.gradient_accumulation_steps
    accelerator = Accelerator(
        gradient_accumulation_steps=accumu_steps,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=6000)),
        ],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        if args.output_path:
            os.makedirs(args.output_path, exist_ok=True)
        dst_dir = save_current_code(outdir=args.output_path)
        OmegaConf.save(args, os.path.join(args.output_path, "config.yaml"))
        printer.info(f"Saving current code to {dst_dir}")

    seed = args.seed + accelerator.process_index
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        shuffle=True,
        worker_init_fn=get_worker_init_fn(
            seed=seed,
            num_workers=args.num_workers,
        ),
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if args.resume is not None:
        printer.info(f"Resuming from {args.resume}")
        pattern = r"checkpoint-epoch-(\d+)(?:-iter-(\d+))?"
        match = re.search(pattern, args.resume)
        if match:
            start_epoch = int(match.group(1))
            resume_step = None if match.group(2) is None else int(match.group(2))
        if accelerator.distributed_type == "DEEPSPEED":
            accelerator.load_state(args.resume, load_module_strict=False)
        else:
            accelerator.load_state(args.resume, strict=False)
    else:
        start_epoch = args.start_epoch
        resume_step = args.resume_step

    log_writer = (
        SummaryWriter(log_dir=args.output_path) if accelerator.is_main_process else None
    )

    printer.info("Start training")
    for epoch_id in range(start_epoch, args.num_epochs):
        metric_logger = MetricLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch_id)
        if hasattr(dataloader, "dataset") and hasattr(dataloader.dataset, "set_epoch"):
            dataloader.dataset.set_epoch(epoch_id)

        if epoch_id == start_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(dataloader, resume_step)
        else:
            active_dataloader = dataloader
            resume_step = 0

        for iter_step, data in enumerate(
            metric_logger.log_every(active_dataloader, args.print_freq, accelerator, header)
        ):
            data_iter_step = iter_step + resume_step
            epoch_f = epoch_id + data_iter_step / len(dataloader)
            step = int(epoch_f * len(dataloader))
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                loss_value = float(loss)
                accelerator.backward(loss)
                if args.clip_grad is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]
                metric_logger.update(epoch=epoch_f)
                metric_logger.update(lr=lr)
                metric_logger.update(step=step)
                metric_logger.update(loss=loss_value)
                scheduler.step()
                if (data_iter_step + 1) % accumu_steps == 0 and (
                    (data_iter_step + 1) % (accumu_steps * args.print_freq)
                ) == 0:
                    loss_value_reduce = accelerator.gather(
                        torch.tensor(loss_value).to(accelerator.device)
                    ).mean()  # MUST BE EXECUTED BY ALL NODES

                    if log_writer is None:
                        continue
                    """ We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    epoch_1000x = int(epoch_f * 1000)
                    log_writer.add_scalar("train_loss", loss_value_reduce, step)
                    log_writer.add_scalar("train_lr", lr, step)
                    log_writer.add_scalar("train_iter", epoch_1000x, step)
            if (
                data_iter_step % int(args.save_freq * len(dataloader)) == 0
                and iter_step != 0
                and iter_step != len(active_dataloader) - 1
            ):
                print("saving at step", data_iter_step)
                metric_logger.save(accelerator, model, epoch_id, data_iter_step)
        metric_logger.save(accelerator, model, epoch_id + 1)
