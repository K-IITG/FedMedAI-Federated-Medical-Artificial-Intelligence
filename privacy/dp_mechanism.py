"""Differential privacy helpers and simple secure aggregation masks."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import numpy as np

try:  # pragma: no cover - optional torch
    import torch
except ImportError:  # pragma: no cover - optional torch
    torch = None  # type: ignore


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):  # pragma: no cover - torch path
        return value.detach().cpu().numpy()
    return np.asarray(value)


def add_noise(array, stddev: float, mechanism: str = "gaussian"):
    arr = _to_numpy(array)
    if mechanism.lower() == "laplace":
        noise = np.random.laplace(loc=0.0, scale=stddev, size=arr.shape)
    else:
        noise = np.random.normal(loc=0.0, scale=stddev, size=arr.shape)
    return arr + noise


def clip_gradients(parameters: Iterable, max_grad_norm: float) -> None:
    if torch is None:  # pragma: no cover - torch path
        return
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.norm(torch.stack([g.detach().data.norm(2) for g in grads]), 2)
    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.detach().data.mul_(clip_coef)


def add_noise_to_grads(
    parameters: Iterable, max_grad_norm: float, noise_multiplier: float, mechanism: str
) -> None:
    if torch is None:  # pragma: no cover - torch path
        return
    stddev = noise_multiplier * max_grad_norm
    for p in parameters:
        if p.grad is None:
            continue
        noise = torch.from_numpy(add_noise(np.zeros_like(p.grad.detach().cpu().numpy()), stddev, mechanism)).to(
            p.grad.device
        )
        p.grad.data.add_(noise)


def dp_sgd_step(
    model,
    loss_fn,
    data,
    target,
    optimizer,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    mechanism: str = "gaussian",
):
    if torch is None:  # pragma: no cover - torch path
        raise ImportError("PyTorch is required for DP-SGD")

    optimizer.zero_grad()
    logits = model(data)
    loss = loss_fn(logits, target)
    loss.backward()
    clip_gradients(model.parameters(), max_grad_norm)
    add_noise_to_grads(model.parameters(), max_grad_norm, noise_multiplier, mechanism)
    optimizer.step()
    return loss.detach().item()


def apply_secure_aggregation(
    updates: Iterable[Mapping[str, np.ndarray]],
    masks: Optional[Iterable[Optional[Mapping[str, np.ndarray]]]] = None,
) -> Dict[str, np.ndarray]:
    """Apply simple additive secure aggregation masks that cancel out across clients."""

    updates = list(updates)
    if masks is None:
        masks = [None for _ in updates]
    masks = list(masks)
    if len(updates) != len(masks):
        raise ValueError("Updates and masks must align")

    aggregated: MutableMapping[str, np.ndarray] = {}
    for idx, update in enumerate(updates):
        mask = masks[idx]
        for key, value in update.items():
            masked = _to_numpy(value)
            if mask and key in mask:
                masked = masked + _to_numpy(mask[key])
            aggregated[key] = aggregated.get(key, 0) + masked
    return {k: np.asarray(v) for k, v in aggregated.items()}

