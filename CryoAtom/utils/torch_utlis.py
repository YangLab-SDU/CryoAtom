from typing import Dict, List, Tuple, Union
import torch
import numpy as np
from torch import nn

def shared_cat(args, dim=0, is_torch=True) -> Union[torch.Tensor, np.ndarray]:
    if is_torch:
        return torch.cat(args, dim=dim)
    else:
        return np.concatenate(args, axis=dim)

def is_ndarray(x) -> bool:
    return isinstance(x, np.ndarray)
def get_batches_to_idx(idx_to_batches: torch.Tensor) -> List[torch.Tensor]:
    assert len(idx_to_batches.shape) == 1
    max_batch_num = idx_to_batches.max().item() + 1
    idxs = torch.arange(0, len(idx_to_batches), dtype=int, device=idx_to_batches.device)
    return [idxs[idx_to_batches == i] for i in range(max_batch_num)]
def get_module_device(module: nn.Module) -> torch.device:
    return next(module.parameters()).device
def get_batch_slices(
    num_total: int,
    batch_size: int,
) -> List[List[int]]:
    if num_total <= batch_size:
        return [list(range(num_total))]

    num_batches = num_total // batch_size
    batches = [
        list(range(i * batch_size, (i + 1) * batch_size)) for i in range(num_batches)
    ]
    if num_total % batch_size > 0:
        batches += [list(range(num_batches * batch_size, num_total))]
    return batches

def padded_sequence_softmax(
    padded_sequence_values: torch.Tensor,
    padded_mask: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-6,
) -> torch.Tensor:
    padded_softmax = torch.softmax(padded_sequence_values, dim=dim)
    padded_softmax = padded_softmax * padded_mask  # Mask out padded values
    padded_softmax = (
        padded_softmax / (padded_softmax.sum(dim=dim, keepdim=True) + eps).detach()
    )  # Renormalize
    return padded_softmax

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]