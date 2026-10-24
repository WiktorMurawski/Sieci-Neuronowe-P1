"""Funkcje aktywacji.

Każda aktywacja ma dwie metody:
    forward(z)          -> a = f(z)
    backward(z, a, da)  -> dL/dz, mając dL/da (da)

Dzięki takiemu interfejsowi także softmax (którego pochodna to macierz
Jacobiego, a nie zwykłe mnożenie element po elemencie) pasuje do tego samego
schematu co pozostałe funkcje.
"""
import numpy as np


class Activation:
    name = "base"

    def forward(self, z):
        raise NotImplementedError

    def backward(self, z, a, da):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class Linear(Activation):
    name = "linear"

    def forward(self, z):
        return z

    def backward(self, z, a, da):
        return da


class Sigmoid(Activation):
    name = "sigmoid"

    def forward(self, z):
        # clip chroni przed przepełnieniem w exp()
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def backward(self, z, a, da):
        return da * a * (1.0 - a)


class Tanh(Activation):
    name = "tanh"

    def forward(self, z):
        return np.tanh(z)

    def backward(self, z, a, da):
        return da * (1.0 - a * a)


class ReLU(Activation):
    name = "relu"

    def forward(self, z):
        return np.maximum(0.0, z)

    def backward(self, z, a, da):
        return da * (z > 0)


class Softmax(Activation):
    """Softmax po osi 1 (każdy wiersz = jedna próbka)."""
    name = "softmax"

    def forward(self, z):
        e = np.exp(z - z.max(axis=1, keepdims=True))  # stabilność numeryczna
        return e / e.sum(axis=1, keepdims=True)

    def backward(self, z, a, da):
        # J^T da  dla  J_ij = a_i (delta_ij - a_j)
        s = np.sum(da * a, axis=1, keepdims=True)
        return a * (da - s)


ACTIVATIONS = {
    "linear": Linear,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "softmax": Softmax,
}


def get_activation(spec):
    """Zwraca obiekt aktywacji z nazwy (str) albo przepuszcza gotowy obiekt."""
    if isinstance(spec, Activation):
        return spec
    if spec in ACTIVATIONS:
        return ACTIVATIONS[spec]()
    raise ValueError(f"Nieznana aktywacja {spec!r}. Dostępne: {sorted(ACTIVATIONS)}")
