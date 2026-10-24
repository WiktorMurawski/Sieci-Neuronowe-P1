"""Perceptron wielowarstwowy (MLP) z propagacją wsteczną - tylko numpy.

Przykład użycia (interfejs podobny do scikit-learn):

    net = MLP([2, 16, 16, 3], hidden_activation="tanh", task="classification",
              loss="cross_entropy", use_bias=True, seed=42)
    hist = net.fit(X_train, y_train, epochs=200, batch_size=16, lr=0.05,
                   optimizer="sgd", optimizer_params={"momentum": 0.9},
                   X_val=X_test, y_val=y_test)
    y_pred = net.predict(X_test)
    acc = net.score(X_test, y_test)

fit() zwraca obiekt History z błędem każdej iteracji i epoki, metryką oraz kopiami wag co snapshot_every epok
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from activations import get_activation
from losses import TASK_LOSSES, get_loss
from optimizers import get_optimizer


class Dense:
    """Warstwa w pełni połączona: a = f(x @ W + b)."""

    def __init__(self, n_in, n_out, activation, use_bias, init, init_std, rng):
        self.activation = get_activation(activation)
        if init == "auto":
            init = "he" if self.activation.name in ("relu", "leaky_relu") else "xavier"
        if init == "xavier":
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W = rng.uniform(-limit, limit, size=(n_in, n_out))
        elif init == "he":
            self.W = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        elif init == "normal":
            self.W = rng.normal(0.0, init_std, size=(n_in, n_out))
        else:
            raise ValueError(f"Nieznana inicjalizacja {init!r} (auto/xavier/he/normal)")
        self.b = np.zeros(n_out) if use_bias else None

    def forward(self, x):
        self.x = x
        self.z = x @ self.W
        if self.b is not None:
            self.z = self.z + self.b
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, da):
        """da = dL/da tej warstwy; zwraca dL/dx (da dla warstwy poprzedniej)."""
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
    """Zapis przebiegu uczenia (dane wejściowe do wizualizacji)."""
    metric_name: str = ""
    batch_loss: list = field(default_factory=list)       # błąd każdej iteracji (mini-batcha)
    loss: list = field(default_factory=list)             # błąd na całym zb. uczącym, po epoce
    val_loss: list = field(default_factory=list)
    metric: list = field(default_factory=list)           # accuracy (klasyf.) lub R^2 (regr.)
    val_metric: list = field(default_factory=list)
    snapshot_epochs: list = field(default_factory=list)  # 0 = wagi początkowe
    snapshots: list = field(default_factory=list)        # [warstwa][W lub b] dla każdej migawki


class MLP:
    """
    Parametry
    ---------
    layer_sizes : [n_wejść, n_ukr_1, ..., n_wyjść]. Bez warstw ukrytych: [n_in, n_out].
                  Klasyfikacja: n_wyjść = liczba klas (kodowanie one-hot).
    hidden_activation : nazwa albo lista nazw (po jednej na warstwę ukrytą).
    output_activation : domyślnie "softmax" (klasyfikacja) / "linear" (regresja).
    task : "classification" albo "regression".
    loss : klasyfikacja: "cross_entropy" (domyślnie), "mse"; regresja: "mse" (domyślnie), "mae".
    use_bias : czy warstwy mają biasy.
    init : "auto" (Xavier dla sigmoid/tanh/linear/softmax, He dla ReLU), "xavier", "he", "normal".
    seed : ziarno generatora - wagi początkowe i tasowanie danych są w pełni powtarzalne.
    """

    def __init__(self, layer_sizes, hidden_activation="tanh", output_activation=None,
                 task="classification", loss=None, use_bias=True,
                 init="auto", init_std=0.1, seed=None):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes musi zawierać co najmniej wejście i wyjście")
        if task not in TASK_LOSSES:
            raise ValueError("task musi być 'classification' lub 'regression'")
        self.layer_sizes = list(layer_sizes)
        self.task = task
        self.use_bias = use_bias
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        n_layers = len(layer_sizes) - 1
        if isinstance(hidden_activation, (list, tuple)):
            if len(hidden_activation) != n_layers - 1:
                raise ValueError("hidden_activation: potrzeba po jednej aktywacji na warstwę ukrytą")
            hidden = list(hidden_activation)
        else:
            hidden = [hidden_activation] * (n_layers - 1)
        if output_activation is None:
            output_activation = "softmax" if task == "classification" else "linear"
        activations = hidden + [output_activation]

        if loss is None:
            loss = TASK_LOSSES[task][0]
        self.loss = get_loss(loss)
        if self.loss.name not in TASK_LOSSES[task]:
            raise ValueError(f"Strata {self.loss.name!r} nie pasuje do zadania {task!r}; "
                             f"dostępne: {TASK_LOSSES[task]}")

        self.layers = [Dense(layer_sizes[i], layer_sizes[i + 1], activations[i],
                             use_bias, init, init_std, self.rng)
                       for i in range(n_layers)]

        out_name = self.layers[-1].activation.name
        if self.loss.name == "cross_entropy" and out_name != "softmax":
            raise ValueError("cross_entropy wymaga aktywacji wyjściowej 'softmax'")
        if task == "regression" and out_name == "softmax":
            raise ValueError("softmax nie nadaje się do regresji")
        if task == "classification" and layer_sizes[-1] < 2:
            raise ValueError("klasyfikacja wymaga >= 2 neuronów wyjściowych (kodowanie one-hot)")

        self.classes_ = None

    # pomocnicze
    @property
    def n_parameters(self):
        return sum(p.size for layer in self.layers for p in layer.params())

    def __repr__(self):
        acts = [layer.activation.name for layer in self.layers]
        return (f"MLP(layer_sizes={self.layer_sizes}, activations={acts}, task={self.task!r}, "
                f"loss={self.loss.name!r}, use_bias={self.use_bias}, seed={self.seed}, "
                f"parametry={self.n_parameters})")

    @staticmethod
    def _as2d(X):
        X = np.asarray(X, dtype=float)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def get_weights(self):
        """Kopie wag: lista po warstwach, w każdej [W] albo [W, b]."""
        return [[p.copy() for p in layer.params()] for layer in self.layers]

    def set_weights(self, weights):
        for layer, new in zip(self.layers, weights):
            for p, w in zip(layer.params(), new):
                p[...] = w

    def _encode_targets(self, y, fit):
        """Klasyfikacja: etykiety -> one-hot (etykiety mogą być dowolne, np. 1,2,3).
        Regresja: wektor -> kolumna."""
        n_out = self.layer_sizes[-1]
        y = np.asarray(y)
        if self.task == "regression":
            y = y.astype(float)
            return y.reshape(-1, 1) if y.ndim == 1 else y
        if y.ndim == 2:  # już one-hot
            if fit:
                self.classes_ = np.arange(y.shape[1])
            return y.astype(float)
        if fit:
            self.classes_ = np.unique(y)
            if len(self.classes_) != n_out:
                raise ValueError(f"W danych jest {len(self.classes_)} klas, a sieć ma "
                                 f"{n_out} neuronów wyjściowych")
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= len(self.classes_)) or np.any(self.classes_[np.clip(idx, 0, len(self.classes_) - 1)] != y):
            raise ValueError("W y są etykiety, których nie było w zbiorze uczącym")
        Y = np.zeros((len(y), n_out))
        Y[np.arange(len(y)), idx] = 1.0
        return Y

    # propagacja
    def _forward(self, X):
        a = X
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def _loss_and_grads(self, Xb, Yb):
        """Przejście w przód i wstecz dla jednego batcha; zwraca (błąd, gradienty)."""
        a = self._forward(Xb)
        loss = self.loss(a, Yb)
        grad = self.loss.gradient(a, Yb)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return loss, [g for layer in self.layers for g in layer.grads()]

    def _evaluate(self, X, Y):
        """(błąd, metryka) na całym zbiorze; metryka = accuracy albo R^2."""
        a = self._forward(X)
        loss = self.loss(a, Y)
        if self.task == "classification":
            metric = float(np.mean(a.argmax(axis=1) == Y.argmax(axis=1)))
        else:
            ss_res = np.sum((a - Y) ** 2)
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
            metric = float(1.0 - ss_res / max(ss_tot, 1e-12))
        return loss, metric

    # uczenie
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.01,
            optimizer="sgd", optimizer_params=None, shuffle=True,
            X_val=None, y_val=None, snapshot_every=1, verbose=0):
        """Uczy sieć mini-batch SGD (lub wariantem) z propagacją wsteczną.

        batch_size=None -> cały zbiór naraz; batch_size=1 -> klasyczny SGD.
        snapshot_every: co ile epok zapisać kopię wag (None = wcale; dla dużych sieci/MNIST).
        verbose=k: wypisuj postęp co k epok.
        Zwraca History (także w self.history_).
        """
        if (X_val is None) != (y_val is None):
            raise ValueError("podaj jednocześnie X_val i y_val")
        X = self._as2d(X)
        Y = self._encode_targets(y, fit=True)
        has_val = X_val is not None
        if has_val:
            X_val = self._as2d(X_val)
            Y_val = self._encode_targets(y_val, fit=False)

        n = len(X)
        bs = n if batch_size is None else min(batch_size, n)
        opt = get_optimizer(optimizer, lr, **(optimizer_params or {}))
        params = [p for layer in self.layers for p in layer.params()]

        hist = History(metric_name="accuracy" if self.task == "classification" else "r2")
        if snapshot_every:
            hist.snapshot_epochs.append(0)
            hist.snapshots.append(self.get_weights())

        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(n) if shuffle else np.arange(n)
            for start in range(0, n, bs):
                idx = order[start:start + bs]
                loss, grads = self._loss_and_grads(X[idx], Y[idx])
                hist.batch_loss.append(loss)
                opt.step(params, grads)

            tr_loss, tr_metric = self._evaluate(X, Y)
            hist.loss.append(tr_loss)
            hist.metric.append(tr_metric)
            if has_val:
                v_loss, v_metric = self._evaluate(X_val, Y_val)
                hist.val_loss.append(v_loss)
                hist.val_metric.append(v_metric)
            if snapshot_every and epoch % snapshot_every == 0:
                hist.snapshot_epochs.append(epoch)
                hist.snapshots.append(self.get_weights())

            if verbose and (epoch % verbose == 0 or epoch == 1):
                msg = f"epoka {epoch:4d}  błąd={tr_loss:.5f}  {hist.metric_name}={tr_metric:.4f}"
                if has_val:
                    msg += f"  | val błąd={v_loss:.5f}  val {hist.metric_name}={v_metric:.4f}"
                print(msg)

            if not np.isfinite(tr_loss):
                warnings.warn(f"Błąd przestał być skończony w epoce {epoch} "
                              "(uczenie rozbiegło się - zmniejsz lr lub przeskaluj dane).")
                break

        self.history_ = hist
        return hist

    # użycie
    def predict_proba(self, X):
        """Surowe wyjście sieci (dla klasyfikacji: prawdopodobieństwa klas z softmax)."""
        return self._forward(self._as2d(X)).copy()

    def predict(self, X):
        """Klasyfikacja: etykiety klas (w oryginalnym kodowaniu). Regresja: wartości."""
        out = self.predict_proba(X)
        if self.task == "classification":
            return self.classes_[out.argmax(axis=1)]
        return out[:, 0] if out.shape[1] == 1 else out

    def score(self, X, y):
        """Klasyfikacja: accuracy. Regresja: R^2."""
        return self._evaluate(self._as2d(X), self._encode_targets(y, fit=False))[1]
