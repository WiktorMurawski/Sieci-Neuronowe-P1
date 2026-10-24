"""Wizualizacje MLP (matplotlib). Funkcje zwracają Figure; zapis: save_figure(fig, ścieżka).

    plot_regression(net, X, y, ...)       dane i krzywa nauczonej sieci (R -> R)
    plot_classification(net, X, y, ...)   dane i obszary decyzyjne sieci (R^2)
    plot_loss(history)                    błąd po iteracjach i metryka po epokach
    plot_weight_trajectories(history)     wartości wag w kolejnych epokach

Jeśli sieć była uczona na danych przeskalowanych (Standardizer), podaj x_scaler / y_scaler,
a wykresy będą rysowane w oryginalnych jednostkach.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]   # kształt jako drugi kanał kodowania klasy


def save_figure(fig, path, dpi=150):
    """Zapisuje wykres (tworząc katalog) i zamyka go."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_regression(net, X, y, X_test=None, y_test=None, x_scaler=None, y_scaler=None, title=None):
    """Regresja R -> R: punkty zbioru i krzywa wyuczona przez sieć."""
    X = np.asarray(X, float).reshape(len(X), -1)
    all_x = X if X_test is None else np.vstack([X, np.asarray(X_test, float).reshape(len(X_test), -1)])
    grid = np.linspace(all_x.min(), all_x.max(), 500).reshape(-1, 1)
    scaled = grid if x_scaler is None else x_scaler.transform(grid)
    curve = net.predict(scaled).reshape(-1, 1)
    if y_scaler is not None:
        curve = y_scaler.inverse_transform(curve)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(X[:, 0], y, s=12, color="gray", alpha=0.6, label="zbiór uczący")
    if X_test is not None:
        ax.scatter(np.asarray(X_test, float).reshape(len(X_test), -1)[:, 0], y_test, s=22, marker="s",
                   facecolors="none", edgecolors=COLORS[1], label="zbiór testowy")
    ax.plot(grid[:, 0], curve[:, 0], color=COLORS[0], lw=2, label="sieć")
    ax.set(xlabel="x", ylabel="y", title=title or "Regresja")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_classification(net, X, y, X_test=None, y_test=None, x_scaler=None, title=None):
    """Klasyfikacja w R^2: obszary decyzyjne sieci, punkty uczące (pełne) i testowe (puste)."""
    X = np.asarray(X, float)
    all_x = X if X_test is None else np.vstack([X, X_test])
    lo, hi = all_x.min(axis=0), all_x.max(axis=0)
    pad = 0.08 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 300),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], 300))
    points = np.c_[gx.ravel(), gy.ravel()]
    if x_scaler is not None:
        points = x_scaler.transform(points)
    zone = net.predict_proba(points).argmax(axis=1).reshape(gx.shape)   # numer klasy przewidzianej w każdym punkcie

    n_classes = len(net.classes_)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.contourf(gx, gy, zone, levels=np.arange(n_classes + 1) - 0.5, colors=COLORS[:n_classes], alpha=0.2)
    ax.contour(gx, gy, zone, levels=np.arange(n_classes - 1) + 0.5, colors="k", linewidths=0.8)
    for k, label in enumerate(net.classes_):
        in_class = np.asarray(y) == label
        ax.scatter(X[in_class, 0], X[in_class, 1], s=30, color=COLORS[k], marker=MARKERS[k],
                   edgecolors="white", linewidths=0.5, label=f"klasa {label}")
        if X_test is not None:
            in_class = np.asarray(y_test) == label
            ax.scatter(X_test[in_class, 0], X_test[in_class, 1], s=40, marker=MARKERS[k],
                       facecolors="none", edgecolors=COLORS[k], linewidths=1.5)
    if X_test is not None:
        ax.scatter([], [], facecolors="none", edgecolors="gray", label="zbiór testowy (puste)")
    ax.set(xlabel="x₁", ylabel="x₂", title=title or "Klasyfikacja", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    return fig


def plot_loss(history):
    """Lewy wykres: błąd po iteracjach (aktualizacjach wag) - błąd mini-batcha oraz błąd na całym zbiorze
    po każdej epoce. Prawy: metryka (accuracy lub R^2) po epokach."""
    epochs = np.arange(1, len(history.loss) + 1)
    iterations_per_epoch = len(history.batch_loss) / len(history.loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(np.arange(1, len(history.batch_loss) + 1), history.batch_loss, color=COLORS[0], alpha=0.25, lw=0.8,
             label="mini-batch")
    ax1.plot(epochs * iterations_per_epoch, history.loss, color=COLORS[0], label="uczący (cały zbiór, po epoce)")
    ax2.plot(epochs, history.metric, color=COLORS[0], label="uczący")
    if history.val_loss:
        ax1.plot(epochs * iterations_per_epoch, history.val_loss, color=COLORS[1], label="walidacyjny")
        ax2.plot(epochs, history.val_metric, color=COLORS[1], label="walidacyjny")
    ax1.set(xlabel="iteracja (aktualizacja wag)", ylabel="błąd", yscale="log", title="Błąd w kolejnych iteracjach")
    ax2.set(xlabel="epoka", ylabel=history.metric_name, title=f"{history.metric_name} w kolejnych epokach")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_weight_trajectories(history, max_lines=40):
    """Wartości wag (niebieskie) i biasów (pomarańczowe, przerywane) w kolejnych epokach; panel na warstwę.
    Wymaga fit(..., snapshot_every=k). Przy dużej liczbie wag rysowana jest równomierna próbka."""
    snapshots, epochs = history.snapshots, history.snapshot_epochs
    n_layers = len(snapshots[0])
    n_cols = min(n_layers, 3)
    fig, axes = plt.subplots(-(-n_layers // n_cols), n_cols, figsize=(4.5 * n_cols, 3.2 * -(-n_layers // n_cols)),
                             squeeze=False)
    for layer in range(n_layers):
        ax = axes.flat[layer]
        weights = np.array([snap[layer][0].ravel() for snap in snapshots])        # (migawki, liczba wag)
        pick = np.linspace(0, weights.shape[1] - 1, min(weights.shape[1], max_lines)).astype(int)
        ax.plot(epochs, weights[:, pick], color=COLORS[0], alpha=0.5, lw=1)
        if len(snapshots[0][layer]) > 1:                                          # są biasy
            biases = np.array([snap[layer][1] for snap in snapshots])
            ax.plot(epochs, biases[:, :max_lines], color=COLORS[1], alpha=0.8, lw=1, ls="--")
        ax.set(title=f"warstwa {layer + 1}", xlabel="epoka", ylabel="wartość")
        ax.grid(alpha=0.3)
    for ax in axes.flat[n_layers:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
