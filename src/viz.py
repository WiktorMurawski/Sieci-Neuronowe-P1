"""Wizualizacje MLP (matplotlib).

Każda funkcja zwraca obiekt Figure i niczego sama nie zapisuje ani nie wyświetla:
    fig = plot_loss(hist)
    save_figure(fig, "figures/loss.png")    # albo plt.show()

Funkcje:
    plot_regression(net, X, y, ...)       dane + krzywa nauczonej sieci (R -> R)
    plot_classification(net, X, y, ...)   dane + obszary decyzyjne (R^2)
    plot_loss(history)                    błąd po iteracjach + metryka po epokach
    plot_weight_trajectories(history)     wartości wag w kolejnych epokach
    plot_weight_heatmaps(history)         macierze wag W w wybranych epokach

Sieć uczoną na danych przeskalowanych (Standardizer) wystarczy opisać parametrami
x_scaler / y_scaler - wykresy są wtedy rysowane w oryginalnych jednostkach.
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D

# --- paleta (kategorie w stałej kolejności: pierwsze 3 sloty są walidowane dla wszystkich par)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]   # drugi kanał kodowania klasy (kształt)
SERIES, SERIES2 = CLASS_COLORS[0], CLASS_COLORS[1]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])

_RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titlelocation": "left", "axes.titlesize": 11, "font.size": 10,
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 110,
}


def _tint(color, amount=0.2):
    """Jasny odcień koloru (mieszanie z tłem) - na pola obszarów decyzyjnych."""
    c, s = np.array(to_rgb(color)), np.array(to_rgb(SURFACE))
    return tuple(amount * c + (1 - amount) * s)


def save_figure(fig, path, dpi=150):
    """Zapisuje wykres (tworząc katalog) i zamyka go, żeby nie zajmował pamięci."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# regresja
def _predict_reg(net, X, x_scaler, y_scaler):
    Xin = x_scaler.transform(X) if x_scaler is not None else X
    p = net.predict(Xin).reshape(-1, 1)
    if y_scaler is not None:
        p = y_scaler.inverse_transform(p)
    return p[:, 0]


def _r2(y, p):
    return 1.0 - np.sum((y - p) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)


def plot_regression(net, X, y, X_test=None, y_test=None, x_scaler=None, y_scaler=None,
                    title=None, ax=None):
    """Regresja R -> R: punkty zbioru i krzywa wyuczona przez sieć."""
    X = np.asarray(X, float).reshape(len(X), -1)
    y = np.asarray(y, float)
    if X.shape[1] != 1:
        raise ValueError("plot_regression obsługuje tylko wejście jednowymiarowe (R -> R)")
    xs_all = X if X_test is None else np.vstack([X, np.asarray(X_test, float).reshape(len(X_test), -1)])
    lo, hi = xs_all.min(), xs_all.max()
    pad = 0.04 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, 500).reshape(-1, 1)

    r2_tr = _r2(y, _predict_reg(net, X, x_scaler, y_scaler))
    parts = [f"R² uczący = {r2_tr:.4f}"]
    with plt.rc_context(_RC):
        fig, ax = (plt.subplots(figsize=(7, 4.2)) if ax is None else (ax.figure, ax))
        ax.scatter(X[:, 0], y, s=14, color=INK2, alpha=0.55, linewidths=0,
                   label="zbiór uczący", zorder=2)
        if X_test is not None:
            Xt = np.asarray(X_test, float).reshape(len(X_test), -1)
            yt = np.asarray(y_test, float)
            ax.scatter(Xt[:, 0], yt, s=26, facecolors="none", edgecolors=SERIES2, marker="s",
                       linewidths=1.0, label="zbiór testowy", zorder=2)
            parts.append(f"R² testowy = {_r2(yt, _predict_reg(net, Xt, x_scaler, y_scaler)):.4f}")
        ax.plot(grid[:, 0], _predict_reg(net, grid, x_scaler, y_scaler), color=SERIES,
                label="sieć", zorder=3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title or "Regresja   " + "   ".join(parts))
        ax.legend(loc="best")
    return fig


# klasyfikacja
def _class_index(net, pts, x_scaler):
    if x_scaler is not None:
        pts = x_scaler.transform(pts)
    return net.predict_proba(pts).argmax(axis=1)


def plot_classification(net, X, y, X_test=None, y_test=None, x_scaler=None,
                        title=None, ax=None, resolution=300):
    """Klasyfikacja w R²: obszary decyzyjne sieci, punkty uczące (pełne),
    testowe (puste) i błędnie sklasyfikowane (obwiedzione)."""
    X = np.asarray(X, float)
    if X.shape[1] != 2:
        raise ValueError("plot_classification obsługuje tylko wejście dwuwymiarowe")
    K = len(net.classes_)
    if K > len(CLASS_COLORS):
        raise ValueError(f"Obsługiwane jest maks. {len(CLASS_COLORS)} klas")
    allX = X if X_test is None else np.vstack([X, np.asarray(X_test, float)])
    lo, hi = allX.min(axis=0), allX.max(axis=0)
    pad = 0.08 * (hi - lo)
    gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], resolution),
                         np.linspace(lo[1] - pad[1], hi[1] + pad[1], resolution))
    zz = _class_index(net, np.c_[gx.ravel(), gy.ravel()], x_scaler).reshape(gx.shape)

    yi = np.searchsorted(net.classes_, y)
    wrong = _class_index(net, X, x_scaler) != yi
    parts = [f"accuracy uczący = {1 - wrong.mean():.3f}"]

    with plt.rc_context(_RC):
        fig, ax = (plt.subplots(figsize=(6.4, 5.4)) if ax is None else (ax.figure, ax))
        ax.grid(False)
        ax.contourf(gx, gy, zz, levels=np.arange(K + 1) - 0.5,
                    colors=[_tint(c) for c in CLASS_COLORS[:K]])
        if K > 1:
            ax.contour(gx, gy, zz, levels=np.arange(K - 1) + 0.5, colors=INK2, linewidths=1.0)
        for k in range(K):
            m = yi == k
            ax.scatter(X[m, 0], X[m, 1], s=36, c=CLASS_COLORS[k], marker=MARKERS[k],
                       edgecolors=SURFACE, linewidths=0.8, zorder=3)
        ax.scatter(X[wrong, 0], X[wrong, 1], s=120, facecolors="none", edgecolors=INK,
                   linewidths=1.3, zorder=4)
        any_wrong = bool(wrong.any())

        if X_test is not None:
            Xt = np.asarray(X_test, float)
            yti = np.searchsorted(net.classes_, y_test)
            wrong_t = _class_index(net, Xt, x_scaler) != yti
            parts.append(f"testowy = {1 - wrong_t.mean():.3f}")
            for k in range(K):
                m = yti == k
                ax.scatter(Xt[m, 0], Xt[m, 1], s=42, facecolors="none", marker=MARKERS[k],
                           edgecolors=CLASS_COLORS[k], linewidths=1.7, zorder=3)
            ax.scatter(Xt[wrong_t, 0], Xt[wrong_t, 1], s=120, facecolors="none", edgecolors=INK,
                       linewidths=1.3, zorder=4)
            any_wrong = any_wrong or bool(wrong_t.any())

        handles = [Line2D([], [], ls="", marker=MARKERS[k], color=CLASS_COLORS[k], ms=7,
                          label=f"klasa {net.classes_[k]}") for k in range(K)]
        if X_test is not None:
            handles.append(Line2D([], [], ls="", marker="o", mfc="none", mec=INK2, mew=1.5,
                                  ms=7, label="zbiór testowy (puste)"))
        if any_wrong:
            handles.append(Line2D([], [], ls="", marker="o", mfc="none", mec=INK, mew=1.3,
                                  ms=10, label="błąd klasyfikacji"))
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
                  borderaxespad=0)   # poza obszarem wykresu, żeby nie zasłaniać punktów
        ax.set_xlim(gx.min(), gx.max())
        ax.set_ylim(gy.min(), gy.max())
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.set_title(title or "Klasyfikacja   " + "   ".join(parts))
    return fig


# przebieg błędu
def plot_loss(history, log_y="auto", title=None):
    """Lewy panel: błąd po iteracjach (aktualizacjach wag) - błąd mini-batcha oraz błąd
    na całym zbiorze po każdej epoce. Prawy panel: metryka (accuracy / R²) po epokach."""
    n_epochs = len(history.loss)
    if n_epochs == 0:
        raise ValueError("History jest puste - najpierw wywołaj net.fit(...)")
    per_epoch = len(history.batch_loss) / n_epochs
    epoch_x = per_epoch * np.arange(1, n_epochs + 1)
    epochs = np.arange(1, n_epochs + 1)
    has_val = len(history.val_loss) > 0

    values = np.asarray(history.loss + history.val_loss, float)
    values = values[np.isfinite(values)]
    if log_y == "auto":
        log_y = values.size > 0 and values.min() > 0 and values.max() / values.min() > 100

    with plt.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        ax1.plot(np.arange(1, len(history.batch_loss) + 1), history.batch_loss,
                 color=SERIES, alpha=0.28, lw=0.8, label="mini-batch")
        ax1.plot(epoch_x, history.loss, color=SERIES, label="uczący (cały zbiór, po epoce)")
        if has_val:
            ax1.plot(epoch_x, history.val_loss, color=SERIES2, label="walidacyjny/testowy")
        if log_y:
            ax1.set_yscale("log")
        ax1.set_xlabel("iteracja (aktualizacja wag)")
        ax1.set_ylabel("błąd")
        ax1.set_title(title or "Błąd w kolejnych iteracjach")
        ax1.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID,
                   framealpha=0.92)

        metric_label = {"r2": "R²"}.get(history.metric_name, history.metric_name)
        ax2.plot(epochs, history.metric, color=SERIES, label="uczący")
        if has_val:
            ax2.plot(epochs, history.val_metric, color=SERIES2, label="walidacyjny/testowy")
        ax2.set_xlabel("epoka")
        ax2.set_ylabel(metric_label)
        ax2.set_title(f"{metric_label} w kolejnych epokach")
        ax2.legend(loc="lower right")
    return fig


# wagi
def _require_snapshots(history):
    if not history.snapshots:
        raise ValueError("Brak migawek wag - uczenie wymaga snapshot_every=k (k >= 1)")


def plot_weight_trajectories(history, max_per_layer=40, ncols=None):
    """Wartości wag (linie ciągłe) i biasów (przerywane) w kolejnych epokach, osobny
    panel dla każdej warstwy. Przy dużej liczbie wag pokazywana jest równomierna próbka."""
    _require_snapshots(history)
    snaps, ep = history.snapshots, np.asarray(history.snapshot_epochs)
    L = len(snaps[0])
    ncols = ncols or min(L, 3)
    nrows = math.ceil(L / ncols)

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                                 squeeze=False, sharex=True, layout="constrained")
        for l in range(L):
            ax = axes.flat[l]
            W = np.stack([s[l][0] for s in snaps]) # (T, n_in, n_out)
            n_in, n_out = W.shape[1:]
            Wf = W.reshape(len(snaps), -1)
            sel = np.unique(np.linspace(0, Wf.shape[1] - 1, min(Wf.shape[1], max_per_layer)).astype(int))
            ax.plot(ep, Wf[:, sel], color=SERIES, alpha=0.55, lw=1.1)
            note = f"{len(sel)} z {Wf.shape[1]} wag" if len(sel) < Wf.shape[1] else f"{Wf.shape[1]} wag"
            if len(snaps[0][l]) > 1:
                B = np.stack([s[l][1] for s in snaps]) # (T, n_out)
                selb = np.unique(np.linspace(0, B.shape[1] - 1, min(B.shape[1], max_per_layer)).astype(int))
                ax.plot(ep, B[:, selb], color=SERIES2, alpha=0.8, lw=1.3, ls="--")
            ax.set_title(f"warstwa {l + 1}  ({n_in}→{n_out}),  {note}")
            ax.set_xlabel("epoka")
            ax.set_ylabel("wartość")
        for ax in axes.flat[L:]:
            ax.set_visible(False)
        handles = [Line2D([], [], color=SERIES, lw=2, label="wagi W")]
        if any(len(s) > 1 for s in snaps[0]):
            handles.append(Line2D([], [], color=SERIES2, lw=2, ls="--", label="biasy b"))
        axes.flat[0].legend(handles=handles, loc="best", fontsize=8)
    return fig


def plot_weight_heatmaps(history, epochs=None, n_cols=4):
    """Macierze wag W jako mapy cieplne w wybranych epokach (wiersz = warstwa).
    Skala kolorów jest symetryczna i wspólna dla warstwy, żeby można porównywać zmiany."""
    _require_snapshots(history)
    snap_epochs = np.asarray(history.snapshot_epochs)
    if epochs is None:
        idx = np.unique(np.linspace(0, len(snap_epochs) - 1, n_cols).round().astype(int))
    else:
        idx = np.array([int(np.argmin(np.abs(snap_epochs - e))) for e in epochs])
    L = len(history.snapshots[0])

    with plt.rc_context(_RC):
        fig, axes = plt.subplots(L, len(idx), figsize=(2.9 * len(idx) + 1.0, 2.6 * L + 0.4),
                                 squeeze=False, layout="constrained")
        for l in range(L):
            mats = [history.snapshots[i][l][0].T for i in idx] # (n_out, n_in)
            vmax = max(np.abs(m).max() for m in mats) or 1.0
            for c, (i, m) in enumerate(zip(idx, mats)):
                ax = axes[l, c]
                im = ax.imshow(m, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto",
                               interpolation="nearest")
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if l == 0:
                    ax.set_title(f"epoka {snap_epochs[i]}")
                if c == 0:
                    ax.set_ylabel(f"warstwa {l + 1}\n({m.shape[1]}→{m.shape[0]})")
            fig.colorbar(im, ax=axes[l, :].tolist(), shrink=0.85, pad=0.02, label="waga")
    return fig
