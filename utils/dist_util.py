# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

"""
Helpers for single-process and torchrun-based distributed training.
"""

import builtins
import os
import warnings
from datetime import timedelta

import torch as th
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3

used_device = 0
_rank = 0
_local_rank = 0
_world_size = 1
_is_distributed = False
_output_configured = False
_original_print = builtins.print


class _DDPWithAttributeAccess(DDP):
    """DDP wrapper that preserves transparent attribute access to the wrapped module."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _configure_non_main_output():
    """Suppress duplicate print/tqdm noise from non-zero ranks."""
    global _output_configured
    if _output_configured:
        return
    _output_configured = True
    if _rank != 0:
        # [DDP Core] Keep stderr tracebacks visible, but silence ordinary prints/progress bars.
        builtins.print = lambda *args, **kwargs: None
        os.environ.setdefault("TQDM_DISABLE", "1")


def setup_dist(device=0):
    """
    Setup device binding and, when launched with torchrun, initialize a process group.

    Plain `python train_*.py` keeps the original single-process behavior. A torchrun
    launch is detected from RANK/WORLD_SIZE/LOCAL_RANK without hardcoding ranks.
    """
    global used_device, _rank, _local_rank, _world_size, _is_distributed

    requested_device = int(device) if device is not None else 0
    _world_size = _env_int("WORLD_SIZE", 1)
    _rank = _env_int("RANK", 0)
    _local_rank = _env_int("LOCAL_RANK", requested_device)
    _is_distributed = _world_size > 1

    # [DDP Core] Bind each torchrun process to exactly one visible GPU before NCCL init.
    if th.cuda.is_available() and requested_device >= 0:
        used_device = _local_rank if _is_distributed else requested_device
        if used_device >= th.cuda.device_count():
            raise RuntimeError(
                f"Requested cuda:{used_device}, but only {th.cuda.device_count()} visible CUDA devices are available."
            )
        th.cuda.set_device(used_device)
    else:
        used_device = -1

    if _is_distributed and not dist.is_initialized():
        os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        backend = "nccl" if th.cuda.is_available() and used_device >= 0 else "gloo"
        # [DDP Core] torchrun provides MASTER_ADDR/MASTER_PORT/RANK/WORLD_SIZE for env://.
        dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(minutes=30))

    if dist.is_initialized():
        _rank = dist.get_rank()
        _world_size = dist.get_world_size()
        _is_distributed = _world_size > 1

    _configure_non_main_output()


def cleanup_dist():
    """Destroy the process group if this process owns one."""
    # [DDP Core] Explicit teardown avoids stale NCCL communicators on repeated launches.
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def dev():
    """
    Get the device to use for torch/distributed training.
    """
    if th.cuda.is_available() and used_device >= 0:
        return th.device(f"cuda:{used_device}")
    return th.device("cpu")


def get_rank():
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else _rank


def get_local_rank():
    return _local_rank


def get_world_size():
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else _world_size


def is_distributed():
    return get_world_size() > 1


def is_main_process():
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        if dev().type == "cuda":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="No device id is provided.*")
                dist.barrier(device_ids=[used_device])
        else:
            dist.barrier()


def broadcast_object(obj, src=0):
    """Broadcast a picklable Python object from src to every rank."""
    if not is_distributed():
        return obj
    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def unwrap_model(model):
    """Return the underlying nn.Module when model is wrapped by DDP."""
    while isinstance(model, DDP):
        model = model.module
    return model


def wrap_model_ddp(model, find_unused_parameters=False):
    """Wrap a model with DDP in torchrun mode; return it unchanged otherwise."""
    if not is_distributed():
        return model

    ddp_kwargs = {
        "find_unused_parameters": find_unused_parameters,
        "gradient_as_bucket_view": True,
    }
    if dev().type == "cuda":
        ddp_kwargs.update({"device_ids": [used_device], "output_device": used_device})

    # [DDP Core] find_unused_parameters defaults to False for fast reducer buckets.
    return _DDPWithAttributeAccess(model, **ddp_kwargs)


def reduce_mean_tensor(tensor):
    """Average a scalar/tensor across ranks for rank-0 logging."""
    if not is_distributed():
        return tensor.detach() if th.is_tensor(tensor) else tensor
    reduced = tensor.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= get_world_size()
    return reduced


def reduce_mean_dict(losses):
    """Average tensor values in a dict across ranks."""
    if not is_distributed():
        return losses
    return {
        key: reduce_mean_tensor(value) if th.is_tensor(value) else value
        for key, value in losses.items()
    }
