"""Funkcje błędu (straty).

Każda strata ma:
    __call__(a, y)  -> skalar (średnia po próbkach)
    gradient(a, y)  -> dL/da, ta sama kształtu co a

a - wyjście sieci (n, k), y - wartości oczekiwane (n, k)
dla klasyfikacji: kodowanie one-hot
"""
import numpy as np

_EPS = 1e-12


class Loss:
    name = "base"

    def __call__(self, a, y):
        raise NotImplementedError

    def gradient(self, a, y):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class MSE(Loss):
    """Błąd średniokwadratowy: średnia po wszystkich elementach."""
    name = "mse"

    def __call__(self, a, y):
        return float(np.mean((a - y) ** 2))

    def gradient(self, a, y):
        return 2.0 * (a - y) / a.size


class MAE(Loss):
    """Średni błąd bezwzględny."""
    name = "mae"

    def __call__(self, a, y):
        return float(np.mean(np.abs(a - y)))

    def gradient(self, a, y):
        return np.sign(a - y) / a.size


class CrossEntropy(Loss):
    """Entropia krzyżowa; zakłada wyjście softmax i y w kodowaniu one-hot."""
    name = "cross_entropy"

    def __call__(self, a, y):
        return float(-np.sum(y * np.log(np.clip(a, _EPS, 1.0))) / a.shape[0])

    def gradient(self, a, y):
        return -y / np.clip(a, _EPS, 1.0) / a.shape[0]


LOSSES = {"mse": MSE, "mae": MAE, "cross_entropy": CrossEntropy}

# które straty mają sens dla jakiego zadania
TASK_LOSSES = {
    "classification": ("cross_entropy", "mse"),
    "regression": ("mse", "mae"),
}


def get_loss(spec):
    if isinstance(spec, Loss):
        return spec
    if spec in LOSSES:
        return LOSSES[spec]()
    raise ValueError(f"Nieznana funkcja błędu {spec!r}. Dostępne: {sorted(LOSSES)}")
