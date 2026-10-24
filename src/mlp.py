"""Perceptron wielowarstwowy (MLP) z propagacją wsteczną - tylko numpy.

    from optimizers import SGD

    net = MLP([2, 16, 16, 3], hidden_activation="tanh", task="classification",
              loss="cross_entropy", use_bias=True, seed=42)
    history = net.fit(X_train, y_train, epochs=200, batch_size=32,
                      optimizer=SGD(lr=0.05, momentum=0.9), X_val=X_val, y_val=y_val)
    net.predict(X_test)
    net.score(X_test, y_test)
"""
from dataclasses import dataclass, field

import numpy as np

from activations import get_activation
from losses import TASK_LOSSES, get_loss
from optimizers import SGD


class Dense:
    """Warstwa w pełni połączona: a = f(x @ W + b)."""

    def __init__(self, n_in, n_out, activation, use_bias, rng):
        self.activation = get_activation(activation)
        if self.activation.name == "relu":                    # inicjalizacja He dla ReLU
            self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        else:                                                 # Xavier dla pozostałych
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W = rng.uniform(-limit, limit, size=(n_in, n_out))
        self.b = np.zeros(n_out) if use_bias else None

    def forward(self, x):
        self.x = x
        self.z = x @ self.W
        if self.b is not None:
            self.z = self.z + self.b
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, da):
        """da = dL/da tej warstwy. Zapamiętuje gradienty wag, zwraca dL/dx (da warstwy poprzedniej)."""
        dz = self.activation.backward(self.z, self.a, da)
        self.dW = self.x.T @ dz
        self.db = dz.sum(axis=0) if self.b is not None else None
        return dz @ self.W.T

    def params(self):
        return [self.W] if self.b is None else [self.W, self.b]

    def grads(self):
        return [self.dW] if self.b is None else [self.dW, self.db]


@dataclass
class History:
    """Przebieg uczenia - dane wejściowe do wizualizacji."""
    metric_name: str = ""
    batch_loss: list = field(default_factory=list)       # błąd każdej iteracji (mini-batcha)
    loss: list = field(default_factory=list)             # błąd na całym zbiorze uczącym, po każdej epoce
    val_loss: list = field(default_factory=list)
    metric: list = field(default_factory=list)           # accuracy (klasyfikacja) lub R^2 (regresja)
    val_metric: list = field(default_factory=list)
    snapshot_epochs: list = field(default_factory=list)  # 0 = wagi początkowe
    snapshots: list = field(default_factory=list)        # kopie wag: [migawka][warstwa][W lub b]


class MLP:
    """
    layer_sizes       [liczba wejść, neurony ukryte..., liczba wyjść]; bez warstw ukrytych: [n_in, n_out].
                      Klasyfikacja: liczba wyjść = liczba klas (kodowanie one-hot).
    hidden_activation "sigmoid", "tanh" lub "relu".
    output_activation domyślnie "softmax" (klasyfikacja) lub "linear" (regresja).
    task              "classification" lub "regression".
    loss              klasyfikacja: "cross_entropy" (domyślnie) albo "mse"; regresja: "mse" (domyślnie) albo "mae".
    use_bias          czy warstwy mają biasy.
    seed              ziarno generatora: wagi początkowe i kolejność danych są wtedy powtarzalne.
    """

    def __init__(self, layer_sizes, hidden_activation="tanh", output_activation=None,
                 task="classification", loss=None, use_bias=True, seed=None):
        if task not in TASK_LOSSES:
            raise ValueError("task musi być 'classification' lub 'regression'")
        self.layer_sizes = list(layer_sizes)
        self.task = task
        self.rng = np.random.default_rng(seed)
        self.loss = get_loss(loss or TASK_LOSSES[task][0])
        if self.loss.name not in TASK_LOSSES[task]:
            raise ValueError(f"Strata {self.loss.name!r} nie pasuje do zadania {task!r}; dostępne: {TASK_LOSSES[task]}")
        if output_activation is None:
            output_activation = "softmax" if task == "classification" else "linear"
        if self.loss.name == "cross_entropy" and output_activation != "softmax":
            raise ValueError("cross_entropy wymaga aktywacji wyjściowej 'softmax'")

        n_layers = len(layer_sizes) - 1
        activations = [hidden_activation] * (n_layers - 1) + [output_activation]
        self.layers = [Dense(layer_sizes[i], layer_sizes[i + 1], activations[i], use_bias, self.rng)
                       for i in range(n_layers)]

    # ------------------------------------------------------------------ propagacja
    def _forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def _loss_and_grads(self, X, Y):
        """Przejście w przód i wstecz dla jednego batcha; zwraca (błąd, gradienty wszystkich parametrów)."""
        a = self._forward(X)
        loss = self.loss(a, Y)
        grad = self.loss.gradient(a, Y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss, [g for layer in self.layers for g in layer.grads()]

    # ------------------------------------------------------------------ dane
    @staticmethod
    def _as2d(X):
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def _targets(self, y):
        """Klasyfikacja: etykiety -> one-hot (etykiety mogą być dowolne, np. 1, 2, 3). Regresja: kolumna."""
        y = np.asarray(y)
        if self.task == "regression":
            return y.astype(float).reshape(len(y), -1)
        return (y[:, None] == self.classes_).astype(float)

    def _evaluate(self, X, Y):
        """(błąd, metryka) na całym zbiorze; metryka = accuracy albo R^2."""
        a = self._forward(X)
        if self.task == "classification":
            metric = np.mean(a.argmax(axis=1) == Y.argmax(axis=1))
        else:
            metric = 1.0 - np.sum((a - Y) ** 2) / max(np.sum((Y - Y.mean(axis=0)) ** 2), 1e-12)
        return self.loss(a, Y), float(metric)

    def get_weights(self):
        """Kopie wag: lista po warstwach, w każdej [W] albo [W, b]."""
        return [[p.copy() for p in layer.params()] for layer in self.layers]

    # ------------------------------------------------------------------ uczenie
    def fit(self, X, y, epochs=100, batch_size=32, optimizer=None,
            X_val=None, y_val=None, snapshot_every=None):
        """Uczy sieć metodą propagacji wstecznej z wybranym optymalizatorem (domyślnie SGD(lr=0.01)).

        batch_size=None -> cały zbiór naraz; batch_size=1 -> klasyczny SGD.
        optimizer: obiekt z optimizers.py, np. SGD(lr=0.05, momentum=0.9). Obiekt przechowuje swój stan
        (np. prędkość w momentum), więc do nowego uczenia utwórz nowy.
        snapshot_every=k -> co k epok zapisuje kopię wag (do wykresu ewolucji wag).
        Zwraca History. Jeśli błąd przestanie być skończony (uczenie się rozbiegło), uczenie jest przerywane.
        """
        X = self._as2d(X)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) != self.layer_sizes[-1]:
                raise ValueError(f"W danych jest {len(self.classes_)} klas, a sieć ma {self.layer_sizes[-1]} wyjść")
        Y = self._targets(y)
        has_val = X_val is not None
        if has_val:
            X_val, Y_val = self._as2d(X_val), self._targets(y_val)

        n = len(X)
        batch_size = n if batch_size is None else min(batch_size, n)
        optimizer = optimizer or SGD()
        params = [p for layer in self.layers for p in layer.params()]

        history = History(metric_name="accuracy" if self.task == "classification" else "r2")
        if snapshot_every:
            history.snapshot_epochs.append(0)
            history.snapshots.append(self.get_weights())

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n)                   # nowa kolejność próbek w każdej epoce
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                loss, grads = self._loss_and_grads(X[idx], Y[idx])
                history.batch_loss.append(loss)
                optimizer.step(params, grads)

            loss, metric = self._evaluate(X, Y)
            history.loss.append(loss)
            history.metric.append(metric)
            if has_val:
                val_loss, val_metric = self._evaluate(X_val, Y_val)
                history.val_loss.append(val_loss)
                history.val_metric.append(val_metric)
            if snapshot_every and epoch % snapshot_every == 0:
                history.snapshot_epochs.append(epoch)
                history.snapshots.append(self.get_weights())
            if not np.isfinite(loss):
                break
        return history

    # ------------------------------------------------------------------ użycie
    def predict_proba(self, X):
        """Surowe wyjście sieci (klasyfikacja: prawdopodobieństwa klas z softmax)."""
        return self._forward(self._as2d(X)).copy()

    def predict(self, X):
        """Klasyfikacja: etykiety klas (w oryginalnym kodowaniu). Regresja: przewidywane wartości."""
        out = self.predict_proba(X)
        if self.task == "classification":
            return self.classes_[out.argmax(axis=1)]
        return out[:, 0] if out.shape[1] == 1 else out

    def score(self, X, y):
        """Klasyfikacja: accuracy. Regresja: R^2."""
        return self._evaluate(self._as2d(X), self._targets(y))[1]
