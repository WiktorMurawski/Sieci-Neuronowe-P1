"""Wizualizacje MLP (matplotlib). Funkcje zwracają Figure; zapis: save_figure(fig, ścieżka).

    plot_regression(net, X, y, ...)       dane i krzywa nauczonej sieci (R -> R)
    plot_classification(net, X, y, ...)   dane i obszary decyzyjne sieci (R^2)
    plot_loss(history)                    błąd po iteracjach i metryka po epokach
    plot_weight_trajectories(history)     wartości wag w kolejnych epokach
"""
import os

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 8, "lines.linewidth": 1.2})


def save_figure(fig, path, dpi=300):
    """Zapisuje wykres jako PNG (wysoka rozdzielczość) i PDF (wektorowy, do LaTeX-a); tworzy katalog."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axvspan(X.min(), X.max(), color="gray", alpha=0.08, label="zakres danych uczących")
    if X_test is not None:
        X_test = np.asarray(X_test, float).reshape(len(X_test), -1)
        ax.scatter(X_test[:, 0], y_test, s=5, color=COLORS[1], alpha=0.6, linewidths=0,
                   label=f"zbiór testowy ({len(X_test)} pkt)", zorder=1)
    ax.plot(grid[:, 0], curve[:, 0], color=COLORS[0], lw=1.2, label="sieć", zorder=3)
    ax.scatter(X[:, 0], y, s=10, color="black", edgecolors="white", linewidths=0.3,
               label=f"zbiór uczący ({len(X)} pkt)", zorder=5)
    ax.set(xlabel="x", ylabel="y", title=title or "Regresja")
    ax.legend()
    ax.grid(alpha=0.3, lw=0.5)
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
    zone = net.predict_proba(points).argmax(axis=1).reshape(gx.shape)

    n_classes = len(net.classes_)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.contourf(gx, gy, zone, levels=np.arange(n_classes + 1) - 0.5, colors=COLORS[:n_classes], alpha=0.2)
    ax.contour(gx, gy, zone, levels=np.arange(n_classes - 1) + 0.5, colors="k", linewidths=0.6)
    for k, label in enumerate(net.classes_):
        in_class = np.asarray(y) == label
        ax.scatter(X[in_class, 0], X[in_class, 1], s=12, color=COLORS[k], marker=MARKERS[k],
                   edgecolors="white", linewidths=0.3, label=f"klasa {label}")
        if X_test is not None:
            in_class = np.asarray(y_test) == label
            ax.scatter(X_test[in_class, 0], X_test[in_class, 1], s=14, marker=MARKERS[k],
                       facecolors="none", edgecolors=COLORS[k], linewidths=0.7)
    if X_test is not None:
        ax.scatter([], [], facecolors="none", edgecolors="gray", label="zbiór testowy (puste)")
    ax.set(xlabel="x₁", ylabel="x₂", title=title or "Klasyfikacja", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def plot_loss(history):
    """Lewy wykres: błąd po iteracjach (aktualizacjach wag) - błąd mini-batcha oraz błąd na całym zbiorze
    po każdej epoce. Prawy: metryka (accuracy lub R^2) po epokach."""
    epochs = np.arange(1, len(history.loss) + 1)
    iterations_per_epoch = len(history.batch_loss) / len(history.loss)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.plot(np.arange(1, len(history.batch_loss) + 1), history.batch_loss, color=COLORS[0], alpha=0.25, lw=0.5,
             label="mini-batch")
    ax1.plot(epochs * iterations_per_epoch, history.loss, color=COLORS[0], label="zbiór uczący (po epoce)")
    ax2.plot(epochs, history.metric, color=COLORS[0], label="zbiór uczący")
    if history.val_loss:
        ax1.plot(epochs * iterations_per_epoch, history.val_loss, color=COLORS[1], label="zbiór walidacyjny")
        ax2.plot(epochs, history.val_metric, color=COLORS[1], label="zbiór walidacyjny")
    ax1.set(xlabel="iteracja (aktualizacja wag)", ylabel="funkcja straty", yscale="log", title="Funkcja straty w kolejnych iteracjach")
    ax2.set(xlabel="epoka", ylabel=history.metric_name, title=f"{history.metric_name} w kolejnych epokach")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    return fig


def layer_name(layer, n_layers):
    return "warstwa wyjściowa" if layer == n_layers - 1 else f"warstwa ukryta {layer + 1}"


def plot_weight_trajectories(history, max_lines=32):
    """Wartości wag (niebieskie, ciągłe) i biasów (pomarańczowe, przerywane) w kolejnych epokach; panel na warstwę.
    Wymaga fit(..., snapshot_every=k). Przy dużej liczbie wag rysowana jest równomierna próbka."""
    snapshots, epochs = history.snapshots, history.snapshot_epochs
    n_layers = len(snapshots[0])
    n_cols = min(n_layers, 3)
    fig, axes = plt.subplots(-(-n_layers // n_cols), n_cols, figsize=(3.0 * n_cols, 2.4 * -(-n_layers // n_cols)),
                             squeeze=False)
    for layer in range(n_layers):
        ax = axes.flat[layer]
        weights = np.array([snap[layer][0].ravel() for snap in snapshots])
        pick = np.linspace(0, weights.shape[1] - 1, min(weights.shape[1], max_lines)).astype(int)
        ax.plot(epochs, weights[:, pick], color=COLORS[0], alpha=0.5, lw=0.7)
        if len(snapshots[0][layer]) > 1:
            biases = np.array([snap[layer][1] for snap in snapshots])
            ax.plot(epochs, biases[:, :max_lines], color=COLORS[1], alpha=0.8, lw=0.7, ls="--")
        ax.set(title=layer_name(layer, n_layers), xlabel="epoka", ylabel="wartość")
        ax.plot([], [], color=COLORS[0], lw=0.7, label="wagi")
        if len(snapshots[0][layer]) > 1:
            ax.plot([], [], color=COLORS[1], lw=0.7, ls="--", label="biasy")
        ax.legend(loc="best")
        ax.grid(alpha=0.3, lw=0.5)
    for ax in axes.flat[n_layers:]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig
