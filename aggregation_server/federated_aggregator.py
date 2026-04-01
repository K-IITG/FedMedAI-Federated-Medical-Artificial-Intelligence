"""Federated aggregation strategies with optional secure masks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union
from uuid import uuid4

import numpy as np

ArrayLike = Union[np.ndarray, List[float]]
Weights = Mapping[str, ArrayLike]


def _to_numpy(weights: Weights) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in weights.items()}


def _serializable(weights: Optional[Weights]) -> Optional[Dict[str, list]]:
    if weights is None:
        return None
    return {k: np.asarray(v).tolist() for k, v in weights.items()}


@dataclass
class ClientUpdate:
    num_samples: int
    weights: Dict[str, np.ndarray]
    mask: Optional[Dict[str, np.ndarray]] = None


@dataclass
class FederatedAggregator:
    strategy: str = "fedavg"
    global_weights: Optional[Dict[str, np.ndarray]] = None
    round: int = 0
    client_updates: List[ClientUpdate] = field(default_factory=list)
    metrics: List[Dict] = field(default_factory=list)

    def register_client(self) -> str:
        return str(uuid4())

    def set_global_weights(self, weights: Weights) -> None:
        self.global_weights = _to_numpy(weights)

    def get_global_weights(self) -> Optional[Dict[str, np.ndarray]]:
        return self.global_weights

    def submit_update(
        self, weights: Weights, num_samples: int, mask: Optional[Weights] = None
    ) -> None:
        self.client_updates.append(
            ClientUpdate(
                num_samples=num_samples,
                weights=_to_numpy(weights),
                mask=_to_numpy(mask) if mask else None,
            )
        )

    def aggregate(self, strategy: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        if not self.client_updates:
            return self.global_weights
        strategy = strategy or self.strategy
        if strategy not in {"fedavg", "fedmedian"}:
            raise ValueError(f"Unsupported strategy: {strategy}")

        weighted: List[Tuple[int, Dict[str, np.ndarray]]] = []
        for update in self.client_updates:
            weights = self._apply_mask(update.weights, update.mask)
            weighted.append((update.num_samples, weights))

        if strategy == "fedmedian":
            aggregated = self._fedmedian([w for _, w in weighted])
        else:
            aggregated = self._fedavg(weighted)

        self.global_weights = aggregated
        self.round += 1
        self.client_updates.clear()
        return self.global_weights

    def _apply_mask(
        self, weights: Mapping[str, np.ndarray], mask: Optional[Mapping[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if mask is None:
            return dict(weights)
        return {k: v + mask.get(k, 0) for k, v in weights.items()}

    def _fedavg(self, weighted: Iterable[Tuple[int, Mapping[str, np.ndarray]]]) -> Dict[str, np.ndarray]:
        weighted = list(weighted)
        total = float(sum(n for n, _ in weighted))
        if total == 0:
            raise ValueError("Total samples cannot be zero for FedAvg")
        keys = list(weighted[0][1].keys())
        agg: MutableMapping[str, np.ndarray] = {k: np.zeros_like(weighted[0][1][k]) for k in keys}
        for num, weights in weighted:
            for k in keys:
                agg[k] += weights[k] * (num / total)
        return {k: np.asarray(v) for k, v in agg.items()}

    def _fedmedian(self, weights: Iterable[Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        weights = list(weights)
        keys = list(weights[0].keys())
        med: MutableMapping[str, np.ndarray] = {}
        for k in keys:
            stacked = np.stack([w[k] for w in weights], axis=0)
            med[k] = np.median(stacked, axis=0)
        return {k: np.asarray(v) for k, v in med.items()}

    def log_metric(self, payload: Dict) -> None:
        self.metrics.append(payload)

    def reset(self) -> None:
        self.global_weights = None
        self.round = 0
        self.client_updates.clear()
        self.metrics.clear()

    @staticmethod
    def to_serializable(weights: Optional[Mapping[str, np.ndarray]]) -> Optional[Dict[str, list]]:
        return _serializable(weights)

