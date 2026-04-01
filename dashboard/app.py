"""Simple dashboard proxy that surfaces aggregator metrics."""
from __future__ import annotations

import os
from typing import Any, Dict

import requests
from flask import Flask, jsonify, render_template


AGGREGATOR_URL = os.getenv("AGGREGATOR_URL", "http://localhost:5001")

app = Flask(__name__, template_folder="templates")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/metrics")
def api_metrics():
    try:
        response = requests.get(f"{AGGREGATOR_URL}/metrics", timeout=5)
        response.raise_for_status()
        data: Dict[str, Any] = response.json()
    except Exception:
        data = {"metrics": [], "round": 0}
    return jsonify(data)


def create_app() -> Flask:
    return app


if __name__ == "__main__":  # pragma: no cover - manual launch
    app.run(host="0.0.0.0", port=5002)

