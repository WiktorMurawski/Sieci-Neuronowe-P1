"""Optymalizatory: klasa bazowa Optimizer i jej implementacje (na razie SGD).

Optymalizator dostaje listę parametrów sieci (tablice numpy) i listę ich gradientów
i aktualizuje parametry w miejscu. Żeby dodać nowy wariant (np. RMSprop), wystarczy
odziedziczyć po Optimizer i zaimplementować step().
"""
import numpy as np


class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, params, grads):
        """Aktualizuje params w miejscu na podstawie grads (te same kształty i kolejność)."""
        raise NotImplementedError


class SGD(Optimizer):
    """Mini-batch SGD. Bez momentum: p -= lr * g.  Z momentum: v = momentum * v - lr * g;  p += v."""

    def __init__(self, lr=0.01, momentum=0.0):
        super().__init__(lr)
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        for p, g, v in zip(params, grads, self.velocity):
            v *= self.momentum
            v -= self.lr * g
            p += v
