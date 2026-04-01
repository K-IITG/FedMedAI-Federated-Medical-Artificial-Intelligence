"""PyTorch MLP for 6-class disease prediction."""
from typing import Dict, Iterable, Tuple

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - torch optional
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


DEFAULT_INPUT_DIM = 64
DEFAULT_NUM_CLASSES = 6


def _ensure_torch_available():
    if torch is None or nn is None:  # pragma: no cover - guarded by import
        raise ImportError(
            "PyTorch is required for DiseasePredictor; install torch>=2.2.0"
        ) from _TORCH_IMPORT_ERROR


class DiseasePredictor(nn.Module):
    """Simple MLP baseline used by hospital nodes."""

    def __init__(
        self,
        input_dim: int = DEFAULT_INPUT_DIM,
        hidden_dims: Tuple[int, int] = (128, 64),
        num_classes: int = DEFAULT_NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        _ensure_torch_available()
        super().__init__()

        layers = []
        last = input_dim
        for width in hidden_dims:
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last = width
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(last, num_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # type: ignore[name-defined]
        return self.classifier(self.feature_extractor(x))


def build_model(
    input_dim: int = DEFAULT_INPUT_DIM, num_classes: int = DEFAULT_NUM_CLASSES
) -> DiseasePredictor:
    """Factory to create the model with defaults."""

    return DiseasePredictor(input_dim=input_dim, num_classes=num_classes)


def get_loss_fn():
    _ensure_torch_available()
    return nn.CrossEntropyLoss()


def get_optimizer(model: DiseasePredictor, lr: float = 1e-3):
    _ensure_torch_available()
    return torch.optim.Adam(model.parameters(), lr=lr)


def state_dict_to_numpy(state_dict: Dict) -> Dict:
    """Convert a PyTorch state_dict to plain NumPy arrays for transport."""

    _ensure_torch_available()
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def numpy_to_state_dict(state: Dict, device: str = "cpu") -> Dict:
    """Convert NumPy weights back to tensors."""

    _ensure_torch_available()
    return {k: torch.as_tensor(v, device=device) for k, v in state.items()}


def apply_weights(model: DiseasePredictor, weights: Dict) -> None:
    """Load numpy or tensor weights into the model."""

    _ensure_torch_available()
    if not weights:
        return
    if isinstance(next(iter(weights.values())), torch.Tensor):
        model.load_state_dict(weights)  # type: ignore[arg-type]
    else:
        model.load_state_dict(numpy_to_state_dict(weights))  # type: ignore[arg-type]

