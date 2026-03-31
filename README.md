# FedMedAI-Federated-Medical-Artificial-Intelligence
🧬 FedMedAI — Federated Learning system for privacy-preserving healthcare diagnosis. Hospital nodes train locally with DP-SGD &amp;amp; Secure Aggregation; a central server runs FedAvg without ever seeing patient data. Built with PyTorch, Flask &amp;amp; Docker. 6-class disease prediction.

FedMedAI tackles one of healthcare AI's hardest problems: how do you train a powerful diagnostic model when patient data is legally and ethically across hospitals?

Each hospital trains locally on its own EHR data, applies calibrated differential privacy noise (DP-SGD), and submits only encrypted weight updates to a central aggregation server. The server runs FedAvg to build a continuously improving global model — without ever seeing a single patient record.

Built end-to-end as a deployable system: PyTorch disease predictor, Flask microservices, Docker Compose orchestration, real-time dashboard, and a full unit test suite. Designed for research extensibility plug in homomorphic encryption, personalized FL, or vertical federation.

| Module | Files | Purpose |
|:--|:--|:--|
| `models/` | `disease_predictor.py` | PyTorch MLP — 6-class disease prediction |
| `privacy/` | `dp_mechanism.py` | DP-SGD (Gaussian + Laplace) + Secure Aggregation |
| `aggregation_server/` | `federated_aggregator.py`, `app.py` | FedAvg/FedMedian engine + Flask REST API |
| `hospital_node/` | `local_trainer.py`, `client.py`, `data_handler.py` | Local training loop, hospital API, synthetic EHR data |
| `dashboard/` | `app.py`, `templates/index.html` | Real-time monitoring UI with Chart.js |
| `scripts/` | `simulate_federated.py` | Full local simulation (no Docker needed) |
| `tests/` | `test_fedmedai.py` | 17 unit tests — all passing |

🚀 Run It (3 options)
Option 1 — Docker (full stack):
bashtar -xzf fedmedai.tar.gz && cd fedmedai
docker-compose up --build
# Dashboard → http://localhost:5002

Option 2 — Local simulation (fastest):
bashpip install torch numpy flask requests
python scripts/simulate_federated.py --rounds 20 --hospitals 3 --dp

Option 3 — Manual services — start each Flask service separately (see README).
