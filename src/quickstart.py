"""Szybki przykład użycia MLP na czterech przykładowych plikach CSV.

Uruchomienie:  python quickstart.py   (pliki data.*.csv w tym samym katalogu)
"""
import numpy as np

from data import Standardizer, load_csv, train_test_split
from mlp import MLP


def classification(path, seed=0):
    X, y = load_csv(path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, seed=seed, stratify=True)
    n_classes = len(np.unique(y))

    net = MLP([2, 16, 16, n_classes], hidden_activation="tanh", task="classification",
              loss="cross_entropy", use_bias=True, seed=seed)
    net.fit(X_tr, y_tr, epochs=200, batch_size=16, lr=0.05,
            optimizer="sgd", optimizer_params={"momentum": 0.9})
    print(f"{path:32s} train acc={net.score(X_tr, y_tr):.3f}  test acc={net.score(X_te, y_te):.3f}")


def regression(path, seed=0):
    X, y = load_csv(path)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=seed)

    # skalowanie wejścia i celu (cel sięga setek/tysięcy -> bez tego uczenie się rozjeżdża)
    sx, sy = Standardizer().fit(X_tr), Standardizer().fit(y_tr.reshape(-1, 1))
    net = MLP([1, 32, 32, 1], hidden_activation="tanh", task="regression",
              loss="mse", use_bias=True, seed=seed)
    net.fit(sx.transform(X_tr), sy.transform(y_tr.reshape(-1, 1)), epochs=300, batch_size=64,
            lr=0.01, optimizer="adam")

    pred = sy.inverse_transform(net.predict(sx.transform(X_te)).reshape(-1, 1))[:, 0]
    rmse = np.sqrt(np.mean((pred - y_te) ** 2))
    r2 = 1 - np.sum((pred - y_te) ** 2) / np.sum((y_te - y_te.mean()) ** 2)
    print(f"{path:32s} test R2={r2:.4f}  RMSE={rmse:.3f} (w jednostkach y)")


if __name__ == "__main__":
    for f in ("data.simple.test.100.csv", "data.three_gauss.test.100.csv"):
        classification(f)
    for f in ("data.activation.test.100.csv", "data.cube.test.100.csv"):
        regression(f)
