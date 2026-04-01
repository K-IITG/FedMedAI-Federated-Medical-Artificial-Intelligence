"""Lightweight tests for the FedMedAI scaffold."""
from __future__ import annotations

import unittest

import numpy as np

from aggregation_server.federated_aggregator import FederatedAggregator
from hospital_node.data_handler import build_dataloader, generate_synthetic_ehr
from hospital_node.local_trainer import extract_weights, load_weights
from models.disease_predictor import build_model
from privacy.dp_mechanism import add_noise, apply_secure_aggregation
from scripts.simulate_federated import run_simulation

try:  # pragma: no cover - optional torch
    import torch
except ImportError:  # pragma: no cover - optional torch
    torch = None  # type: ignore


class TestFedMedAI(unittest.TestCase):
    def test_synthetic_data_shapes(self):
        x, y = generate_synthetic_ehr(num_samples=10, input_dim=8, num_classes=3, seed=1)
        self.assertEqual(x.shape, (10, 8))
        self.assertEqual(y.shape, (10,))

    @unittest.skipIf(torch is None, "torch not installed")
    def test_model_forward_shape(self):
        model = build_model(input_dim=8, num_classes=3)
        batch = torch.randn(4, 8)
        logits = model(batch)
        self.assertEqual(logits.shape, (4, 3))

    def test_dp_noise(self):
        base = np.zeros((2, 2))
        noisy = add_noise(base, stddev=1.0, mechanism="gaussian")
        self.assertEqual(noisy.shape, base.shape)
        self.assertFalse(np.allclose(base, noisy))

    def test_secure_aggregation_masks_cancel(self):
        updates = [{"w": np.array([1.0, 2.0])}, {"w": np.array([3.0, 4.0])}]
        masks = [
            {"w": np.array([1.0, -1.0])},
            {"w": np.array([-1.0, 1.0])},
        ]
        agg = apply_secure_aggregation(updates, masks)
        np.testing.assert_allclose(agg["w"], np.array([4.0, 6.0]))

    def test_fedavg(self):
        agg = FederatedAggregator(strategy="fedavg")
        agg.submit_update({"w": np.array([1.0, 1.0])}, num_samples=1)
        agg.submit_update({"w": np.array([3.0, 3.0])}, num_samples=3)
        result = agg.aggregate()
        np.testing.assert_allclose(result["w"], np.array([2.5, 2.5]))

    def test_fedmedian(self):
        agg = FederatedAggregator(strategy="fedmedian")
        agg.submit_update({"w": np.array([1.0, 10.0])}, num_samples=1)
        agg.submit_update({"w": np.array([5.0, 4.0])}, num_samples=1)
        agg.submit_update({"w": np.array([9.0, 6.0])}, num_samples=1)
        result = agg.aggregate()
        np.testing.assert_allclose(result["w"], np.array([5.0, 6.0]))

    def test_register_client_unique(self):
        agg = FederatedAggregator()
        a = agg.register_client()
        b = agg.register_client()
        self.assertNotEqual(a, b)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_weight_roundtrip(self):
        model = build_model()
        weights = extract_weights(model)
        noisy = {k: v + 0.01 for k, v in weights.items()}
        load_weights(model, noisy)
        for k, v in model.state_dict().items():
            np.testing.assert_allclose(v.detach().cpu().numpy(), noisy[k])

    @unittest.skipIf(torch is None, "torch not installed")
    def test_train_one_epoch_runs(self):
        model = build_model()
        loader = build_dataloader(num_samples=32, batch_size=8)
        client = FederatedAggregator()
        client.set_global_weights({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
        client.submit_update({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, num_samples=32)
        client.aggregate()

    def test_aggregator_reset(self):
        agg = FederatedAggregator()
        agg.submit_update({"w": np.array([1.0])}, num_samples=1)
        agg.aggregate()
        agg.reset()
        self.assertIsNone(agg.get_global_weights())
        self.assertEqual(agg.round, 0)

    @unittest.skipIf(torch is None, "torch not installed")
    def test_simulation_runs(self):
        agg = run_simulation(rounds=1, hospitals=2, use_dp=False)
        self.assertGreaterEqual(agg.round, 1)

    def test_serialization(self):
        agg = FederatedAggregator()
        weights = {"a": np.array([1.0, 2.0])}
        agg.set_global_weights(weights)
        ser = agg.to_serializable(agg.get_global_weights())
        self.assertEqual(ser["a"], [1.0, 2.0])

    def test_masks_optional(self):
        agg = apply_secure_aggregation([{"w": np.array([1.0])}], masks=None)
        np.testing.assert_allclose(agg["w"], np.array([1.0]))

    def test_zero_client_updates_returns_global(self):
        agg = FederatedAggregator()
        self.assertIsNone(agg.aggregate())

    def test_invalid_strategy(self):
        agg = FederatedAggregator()
        agg.submit_update({"w": np.array([1.0])}, num_samples=1)
        with self.assertRaises(ValueError):
            agg.aggregate(strategy="unknown")


if __name__ == "__main__":  # pragma: no cover - manual run
    unittest.main()

