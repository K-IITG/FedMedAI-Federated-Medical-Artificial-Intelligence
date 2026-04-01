"""Flask REST API for federated aggregation."""
from __future__ import annotations

import json
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, request

from aggregation_server.federated_aggregator import FederatedAggregator


app = Flask(__name__)
aggregator = FederatedAggregator()


def _parse_weights(raw: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in raw.items()}


@app.route("/register", methods=["POST"])
def register():
    client_id = aggregator.register_client()
    return jsonify({"client_id": client_id})


@app.route("/global-model", methods=["GET"])
def global_model():
    return jsonify({"weights": aggregator.to_serializable(aggregator.get_global_weights())})


@app.route("/update", methods=["POST"])
def update_model():
    payload = request.get_json(force=True)
    weights = _parse_weights(payload["weights"])
    num_samples = int(payload.get("num_samples", 1))
    mask = payload.get("mask")
    parsed_mask = _parse_weights(mask) if mask else None

    aggregator.submit_update(weights=weights, num_samples=num_samples, mask=parsed_mask)
    new_weights = aggregator.aggregate(strategy=payload.get("strategy"))
    return jsonify(
        {
            "round": aggregator.round,
            "global_model": aggregator.to_serializable(new_weights),
        }
    )


@app.route("/metrics", methods=["GET", "POST"])
def metrics():
    if request.method == "POST":
        aggregator.log_metric(request.get_json(force=True))
        return jsonify({"status": "recorded"})
    return jsonify({"metrics": aggregator.metrics, "round": aggregator.round})


@app.route("/reset", methods=["POST"])
def reset():
    aggregator.reset()
    return jsonify({"status": "reset"})


def create_app() -> Flask:
    return app


if __name__ == "__main__":  # pragma: no cover - manual launch
    app.run(host="0.0.0.0", port=5001)

