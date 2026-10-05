import os
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_gpu_ids(spec: str) -> List[int]:
    n = torch.cuda.device_count()
    if not torch.cuda.is_available() or n == 0:
        return []
    if spec in (None, "", "all"):
        return list(range(n))
    ids = [int(x) for x in str(spec).split(",") if x.strip() != ""]
    return [i for i in ids if 0 <= i < n]


def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed() -> Tuple[bool, int, int, torch.device]:
    if not ddp_enabled():
        if torch.cuda.is_available():
            return False, 0, 1, torch.device("cuda")
        return False, 0, 1, torch.device("cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, dist.get_world_size(), torch.device(f"cuda:{local_rank}")


def resolve_device(distributed: bool, local_rank: int, gpu_ids: List[int]) -> torch.device:
    if distributed:
        return torch.device(f"cuda:{local_rank}")
    if gpu_ids:
        torch.cuda.set_device(gpu_ids[0])
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


def is_main(rank: int = 0) -> bool:
    if ddp_enabled() and dist.is_initialized():
        return dist.get_rank() == 0
    return rank == 0


def wrap_model(model: nn.Module, device: torch.device, gpu_ids: List[int], distributed: bool, local_rank: int):
    model = model.to(device)
    if distributed:
        return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    if device.type == "cuda" and len(gpu_ids) > 1:
        return nn.DataParallel(model, device_ids=gpu_ids)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def state_dict_to_save(model: nn.Module) -> Dict[str, torch.Tensor]:
    return unwrap_model(model).state_dict()


def load_state_dict(model: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
    if state and next(iter(state)).startswith("module."):
        state = {k[len("module.") :]: v for k, v in state.items()}
    unwrap_model(model).load_state_dict(state, strict=strict)


def load_compatible_state_dict(model: nn.Module, state: Dict[str, torch.Tensor]):
    """Load tensors whose names and shapes match; skip the rest (arch upgrades)."""
    if state and next(iter(state)).startswith("module."):
        state = {k[len("module.") :]: v for k, v in state.items()}
    target = unwrap_model(model)
    current = target.state_dict()
    matched = {}
    skipped = []
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            matched[key] = value
        else:
            skipped.append(key)
    current.update(matched)
    target.load_state_dict(current)
    return len(matched), skipped


def effective_batch_size(per_gpu: int, gpu_ids: List[int], distributed: bool) -> int:
    if distributed:
        return per_gpu
    n = max(len(gpu_ids), 1)
    return per_gpu * n


def gather_numpy(arr):
    import numpy as np

    if not (ddp_enabled() and dist.is_initialized()):
        return arr
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, arr)
    parts = [p for p in parts if p is not None and getattr(p, "size", 0)]
    if not parts:
        return arr
    return np.concatenate(parts, axis=0)


def gather_list(items):
    if not (ddp_enabled() and dist.is_initialized()):
        return list(items)
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, list(items))
    out = []
    for part in parts:
        if part:
            out.extend(part)
    return out


def all_reduce_sum(value: float, device: torch.device) -> float:
    if not (ddp_enabled() and dist.is_initialized()):
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def merge_video_dicts(video_scores: Dict, video_labels: Dict):
    if not (ddp_enabled() and dist.is_initialized()):
        return video_scores, video_labels
    gathered_s = [None] * dist.get_world_size()
    gathered_l = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_s, dict(video_scores))
    dist.all_gather_object(gathered_l, dict(video_labels))
    merged_s, merged_l = defaultdict(list), {}
    for part in gathered_s:
        for key, vals in part.items():
            merged_s[key].extend(vals)
    for part in gathered_l:
        merged_l.update(part)
    return merged_s, merged_l


def barrier():
    if ddp_enabled() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if ddp_enabled() and dist.is_initialized():
        dist.destroy_process_group()
