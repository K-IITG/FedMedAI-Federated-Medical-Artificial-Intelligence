"""Synthetic EHR data generation and loaders."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

try:  # pragma: no cover - optional torch
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional torch
    torch = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore


def generate_synthetic_ehr(
    num_samples: int = 200,
    input_dim: int = 64,
    num_classes: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(loc=0.0, scale=1.0, size=(num_samples, input_dim)).astype("float32")
    weights = rng.normal(loc=0.0, scale=1.0, size=(input_dim, num_classes)).astype("float32")
    logits = features @ weights
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = (probabilities.argmax(axis=1) + rng.integers(0, num_classes, size=num_samples)) % num_classes
    return features, labels.astype("int64")


def build_dataloader(
    num_samples: int = 200,
    batch_size: int = 32,
    seed: int = 42,
    input_dim: int = 64,
    num_classes: int = 6,
):
    features, labels = generate_synthetic_ehr(num_samples=num_samples, input_dim=input_dim, num_classes=num_classes, seed=seed)
    if torch is None:  # pragma: no cover - torch path
        # Fallback: simple list of batches
        batches: List[Tuple[np.ndarray, np.ndarray]] = []
        for start in range(0, len(features), batch_size):
            end = start + batch_size
            batches.append((features[start:end], labels[start:end]))
        return batches

    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def build_holdout(
    num_samples: int = 64,
    seed: int = 7,
    input_dim: int = 64,
    num_classes: int = 6,
):
    return build_dataloader(num_samples=num_samples, batch_size=num_samples, seed=seed, input_dim=input_dim, num_classes=num_classes)

