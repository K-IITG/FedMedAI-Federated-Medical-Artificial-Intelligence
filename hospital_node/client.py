"""Hospital client that coordinates with the aggregation server or in-process aggregator."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import requests

from aggregation_server.federated_aggregator import FederatedAggregator
from hospital_node.local_trainer import (
    extract_weights,
    load_weights,
    num_samples_from_loader,
    train_one_epoch,
)
from models.disease_predictor import DiseasePredictor

try:  # pragma: no cover - optional torch
    import torch
except ImportError:  # pragma: no cover - optional torch
    torch = None  # type: ignore


class HospitalClient:
    def __init__(
        self,
        client_id: str,
        model: DiseasePredictor,
        train_loader,
        aggregator_url: Optional[str] = None,
        aggregator: Optional[FederatedAggregator] = None,
        use_dp: bool = False,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        mechanism: str = "gaussian",
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.aggregator_url = aggregator_url
        self.aggregator = aggregator
        self.use_dp = use_dp
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism

    def sync_weights(self, global_weights: Optional[Dict]) -> None:
        if global_weights:
            load_weights(self.model, global_weights)

    def train_local(self) -> Dict[str, float]:
        return train_one_epoch(
            self.model,
            self.train_loader,
            use_dp=self.use_dp,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            mechanism=self.mechanism,
        )

    def produce_update(self) -> Dict:
        return extract_weights(self.model)

    def _post(self, path: str, payload: Dict) -> Dict:
        if not self.aggregator_url:
            raise ValueError("No aggregator_url provided")
        response = requests.post(f"{self.aggregator_url}{path}", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()

    def send_update(self, mask: Optional[Dict] = None) -> Dict:
        weights = self.produce_update()
        num_samples = num_samples_from_loader(self.train_loader)
        payload = {"weights": {k: v.tolist() for k, v in weights.items()}, "num_samples": num_samples}
        if mask:
            payload["mask"] = {k: np.asarray(v).tolist() for k, v in mask.items()}
        if self.aggregator_url:
            return self._post("/update", payload)
        if self.aggregator:
            self.aggregator.submit_update(weights, num_samples=num_samples, mask=mask)
            new_weights = self.aggregator.aggregate()
            return {"global_model": new_weights, "round": self.aggregator.round}
        raise ValueError("No aggregation target configured")

