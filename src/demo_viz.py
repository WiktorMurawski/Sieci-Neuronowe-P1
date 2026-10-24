"""Generuje komplet wykresów dla czterech przykładowych plików CSV do katalogu figures/.

Uruchomienie:  python demo_viz.py
"""
import matplotlib

matplotlib.use("Agg")  # zapis do plików, bez okna

import numpy as np

import viz
from data import Standardizer, load_csv, train_test_split
from mlp import MLP


def classification(name, seed=0):
    X, y = load_csv(f"data.{name}.test.100.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, seed=seed, stratify=True)
    net = MLP([2, 16, 16, len(np.unique(y))], hidden_activation="tanh", task="classification",
              loss="cross_entropy", seed=seed)
    hist = net.fit(X_tr, y_tr, epochs=200, batch_size=16, lr=0.05, optimizer="sgd",
                   optimizer_params={"momentum": 0.9}, X_val=X_te, y_val=y_te, snapshot_every=1)
    out = f"figures/{name}"
    viz.save_figure(viz.plot_classification(net, X_tr, y_tr, X_te, y_te), f"{out}_klasyfikacja.png")
    viz.save_figure(viz.plot_loss(hist), f"{out}_blad.png")
    viz.save_figure(viz.plot_weight_trajectories(hist), f"{out}_wagi_trajektorie.png")
    viz.save_figure(viz.plot_weight_heatmaps(hist, epochs=[0, 5, 50, 200]), f"{out}_wagi_heatmapy.png")


def regression(name, seed=0):
    X, y = load_csv(f"data.{name}.test.100.csv")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, seed=seed)
    sx, sy = Standardizer().fit(X_tr), Standardizer().fit(y_tr.reshape(-1, 1))
    net = MLP([1, 32, 32, 1], hidden_activation="tanh", task="regression", loss="mse", seed=seed)
    hist = net.fit(sx.transform(X_tr), sy.transform(y_tr.reshape(-1, 1)), epochs=300,
                   batch_size=64, lr=0.01, optimizer="adam", snapshot_every=1,
                   X_val=sx.transform(X_te), y_val=sy.transform(y_te.reshape(-1, 1)))
    out = f"figures/{name}"
    viz.save_figure(viz.plot_regression(net, X_tr, y_tr, X_te, y_te, x_scaler=sx, y_scaler=sy),
                    f"{out}_regresja.png")
    viz.save_figure(viz.plot_loss(hist), f"{out}_blad.png")
    viz.save_figure(viz.plot_weight_trajectories(hist), f"{out}_wagi_trajektorie.png")
    viz.save_figure(viz.plot_weight_heatmaps(hist, epochs=[0, 10, 100, 300]), f"{out}_wagi_heatmapy.png")


if __name__ == "__main__":
    for n in ("simple", "three_gauss"):
        classification(n)
    for n in ("activation", "cube"):
        regression(n)
    print("Zapisano wykresy w katalogu figures/")
