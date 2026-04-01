"""Local training loop for hospital nodes with optional DP-SGD."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from models.disease_predictor import (
    DiseasePredictor,
    apply_weights,
    get_loss_fn,
    get_optimizer,
    state_dict_to_numpy,
)
from privacy.dp_mechanism import dp_sgd_step

try:  # pragma: no cover - optional torch
    import torch
except ImportError:  # pragma: no cover - optional torch
    torch = None  # type: ignore


def _device():
    if torch is None:  # pragma: no cover - torch path
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: DiseasePredictor,
    dataloader,
    use_dp: bool = False,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    mechanism: str = "gaussian",
) -> Dict[str, float]:
    if torch is None:  # pragma: no cover - torch path
        raise ImportError("PyTorch is required for training")

    device = _device()
    model.to(device)
    loss_fn = get_loss_fn()
    optimizer = get_optimizer(model)

    total_loss = 0.0
    total = 0
    correct = 0

    model.train()
    for batch in dataloader:
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        if use_dp:
            loss_val = dp_sgd_step(
                model,
                loss_fn,
                features,
                labels,
                optimizer,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                mechanism=mechanism,
            )
            logits = model(features)
        else:
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            loss_val = loss.detach().item()

        total_loss += loss_val * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def evaluate(model: DiseasePredictor, dataloader) -> Dict[str, float]:
    if torch is None:  # pragma: no cover - torch path
        raise ImportError("PyTorch is required for evaluation")
    device = _device()
    loss_fn = get_loss_fn()
    model.to(device)
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def extract_weights(model: DiseasePredictor) -> Dict:
    return state_dict_to_numpy(model.state_dict())


def load_weights(model: DiseasePredictor, weights: Dict) -> None:
    apply_weights(model, weights)


def num_samples_from_loader(dataloader) -> int:
    try:
        return len(dataloader.dataset)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - fallback
        total = 0
        for batch in dataloader:
            total += len(batch[1])
        return total

