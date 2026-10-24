"""Testy poprawności: sprawdzenie gradientów (różnice skończone), powtarzalność, uczenie.

Uruchomienie:  python test_mlp.py
"""
import itertools

import numpy as np

from mlp import MLP
from optimizers import SGD


def numeric_grad_check(net, X, Y, eps=1e-5):
    """Maksymalny względny błąd między gradientem z backprop a numerycznym.

    Mianownik ma dolną granicę 1e-4, bo dla bardzo małych gradientów różnice
    skończone toną w szumie zaokrągleń (to ograniczenie testu, nie backpropu).
    """
    _, grads = net._loss_and_grads(X, Y)
    params = [p for layer in net.layers for p in layer.params()]
    worst = 0.0
    for p, g in zip(params, grads):
        for i in range(p.size):
            old = p.flat[i]
            p.flat[i] = old + eps
            lp = net.loss(net._forward(X), Y)
            p.flat[i] = old - eps
            lm = net.loss(net._forward(X), Y)
            p.flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = g.flat[i]
            worst = max(worst, abs(num - ana) / max(1e-4, abs(num) + abs(ana)))
    return worst


def test_gradients():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 3))
    hidden_acts = ["sigmoid", "tanh", "relu", "linear"]
    heads = [  # (task, aktywacja wyjściowa, strata)
        ("regression", "linear", "mse"),
        ("regression", "linear", "mae"),
        ("classification", "softmax", "cross_entropy"),
        ("classification", "softmax", "mse"),
        ("classification", "sigmoid", "mse"),
    ]
    for act, (task, out_act, loss), bias in itertools.product(hidden_acts, heads, (True, False)):
        net = MLP([3, 5, 4, 2], hidden_activation=act, output_activation=out_act,
                  task=task, loss=loss, use_bias=bias, seed=1)
        if task == "classification":
            Y = np.eye(2)[rng.integers(0, 2, size=7)]
        else:
            Y = rng.normal(size=(7, 2))
        err = numeric_grad_check(net, X, Y)
        assert err < 1e-5, f"gradient: {act}/{out_act}/{loss}/bias={bias}: błąd {err:.2e}"
    print(f"OK  gradienty ({len(hidden_acts) * len(heads) * 2} konfiguracji)")


def test_reproducibility():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)

    def run(seed):
        net = MLP([2, 8, 2], seed=seed)
        net.fit(X, y, epochs=10, batch_size=8, optimizer=SGD(lr=0.1))
        return net.get_weights()

    a, b, c = run(5), run(5), run(6)
    assert all(np.array_equal(p, q) for la, lb in zip(a, b) for p, q in zip(la, lb))
    assert not np.array_equal(a[0][0], c[0][0])
    print("OK  powtarzalność (to samo ziarno = te same wagi)")


def test_learning():
    rng = np.random.default_rng(0)
    # XOR-podobny problem klasyfikacji (nieseparowalny liniowo), etykiety 1 i 2
    X = rng.uniform(-1, 1, size=(300, 2))
    y = np.where(X[:, 0] * X[:, 1] > 0, 1, 2)
    net = MLP([2, 12, 12, 2], hidden_activation="tanh", seed=0)
    net.fit(X, y, epochs=200, batch_size=32, optimizer=SGD(lr=0.05, momentum=0.9))
    acc = net.score(X, y)
    assert acc > 0.95, acc
    assert set(net.predict(X)) <= {1, 2}
    # regresja R -> R
    x = np.linspace(-3, 3, 200)
    t = np.sin(x)
    reg = MLP([1, 16, 16, 1], hidden_activation="tanh", task="regression", seed=0)
    reg.fit(x, t, epochs=300, batch_size=32, optimizer=SGD(lr=0.02, momentum=0.9))
    r2 = reg.score(x, t)
    assert r2 > 0.98, r2
    # brak warstw ukrytych: regresja liniowa dokładnie odtwarza prostą
    lin = MLP([1, 1], task="regression", seed=0)
    lin.fit(x, 2 * x + 1, epochs=300, batch_size=None, optimizer=SGD(lr=0.1))
    assert lin.score(x, 2 * x + 1) > 0.9999
    print(f"OK  uczenie (accuracy={acc:.3f}, R2 sin={r2:.4f})")


if __name__ == "__main__":
    test_gradients()
    test_reproducibility()
    test_learning()
    print("Wszystkie testy przeszły.")
