"""Optymalizatory. Aktualizują parametry w miejscu (p -= ...)."""
import numpy as np


class Optimizer:
    def step(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    """Gradient prosty (mini-batch), opcjonalnie z momentum."""

    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self._v = None

    def step(self, params, grads):
        if self.momentum > 0:
            if self._v is None:
                self._v = [np.zeros_like(p) for p in params]
            for i, (p, g) in enumerate(zip(params, grads)):
                self._v[i] = self.momentum * self._v[i] - self.lr * g
                p += self._v[i]
        else:
            for p, g in zip(params, grads):
                p -= self.lr * g


class Adam(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self._m = self._v = None
        self._t = 0

    def step(self, params, grads):
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]
        self._t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * g * g
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


OPTIMIZERS = {"sgd": SGD, "adam": Adam}


def get_optimizer(spec, lr, **params):
    """Z nazwy ("sgd", "adam") + lr + parametrów albo gotowy obiekt."""
    if isinstance(spec, Optimizer):
        return spec
    if spec in OPTIMIZERS:
        return OPTIMIZERS[spec](lr=lr, **params)
    raise ValueError(f"Nieznany optymalizator {spec!r}. Dostępne: {sorted(OPTIMIZERS)}")
