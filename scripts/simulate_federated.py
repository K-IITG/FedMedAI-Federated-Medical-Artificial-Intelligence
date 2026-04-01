"""Local end-to-end simulation without Docker."""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np

from aggregation_server.federated_aggregator import FederatedAggregator
from hospital_node.client import HospitalClient
from hospital_node.data_handler import build_dataloader, build_holdout
from hospital_node.local_trainer import evaluate
from models.disease_predictor import build_model

try:  # pragma: no cover - optional torch
    import torch
except ImportError:  # pragma: no cover - optional torch
    torch = None  # type: ignore


def simulate_round(
    aggregator: FederatedAggregator,
    clients: List[HospitalClient],
) -> Dict:
    global_weights = aggregator.get_global_weights()
    for client in clients:
        client.sync_weights(global_weights)
        client.train_local()
        client.send_update()
    new_weights = aggregator.aggregate()
    return {"round": aggregator.round, "global_weights": new_weights}


def evaluate_global(model, weights, holdout_loader):
    if torch is None:  # pragma: no cover - torch path
        return {"loss": 0.0, "acc": 0.0}
    model.eval()
    model.load_state_dict({k: torch.as_tensor(v) for k, v in weights.items()})
    return evaluate(model, holdout_loader)


def run_simulation(
    rounds: int = 3,
    hospitals: int = 3,
    use_dp: bool = False,
    noise_multiplier: float = 0.8,
    max_grad_norm: float = 1.0,
    mechanism: str = "gaussian",
    input_dim: int = 64,
    num_classes: int = 6,
):
    aggregator = FederatedAggregator(strategy="fedavg")
    base_model = build_model(input_dim=input_dim, num_classes=num_classes)
    if torch is not None:  # pragma: no cover - torch path
        aggregator.set_global_weights({k: v.detach().cpu().numpy() for k, v in base_model.state_dict().items()})

    clients: List[HospitalClient] = []
    for idx in range(hospitals):
        model = build_model(input_dim=input_dim, num_classes=num_classes)
        loader = build_dataloader(seed=idx, num_samples=120)
        client = HospitalClient(
            client_id=f"hospital-{idx}",
            model=model,
            train_loader=loader,
            aggregator=aggregator,
            use_dp=use_dp,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            mechanism=mechanism,
        )
        clients.append(client)

    holdout = build_holdout()
    for _ in range(rounds):
        simulate_round(aggregator, clients)
        if torch is not None:  # pragma: no cover - torch path
            metrics = evaluate_global(base_model, aggregator.get_global_weights(), holdout)
            aggregator.log_metric({"round": aggregator.round, **metrics})
        else:
            aggregator.log_metric({"round": aggregator.round, "loss": 0.0, "acc": 0.0})

    return aggregator


def main():  # pragma: no cover - CLI entrypoint
    parser = argparse.ArgumentParser(description="Simulate federated training locally")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--hospitals", type=int, default=3)
    parser.add_argument("--dp", action="store_true", help="Enable DP-SGD")
    parser.add_argument("--mechanism", choices=["gaussian", "laplace"], default="gaussian")
    parser.add_argument("--noise-multiplier", type=float, default=0.8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()

    aggregator = run_simulation(
        rounds=args.rounds,
        hospitals=args.hospitals,
        use_dp=args.dp,
        mechanism=args.mechanism,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    print(json.dumps({"rounds": aggregator.round, "metrics": aggregator.metrics}, indent=2))


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
