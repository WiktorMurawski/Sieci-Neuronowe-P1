"""
Eksperymenty na MNIST z własnym MLP.

Eksperymenty:
  A - architektura (głębokość / szerokość)
  B - funkcja aktywacji
  C - learning rate + momentum
  D - batch size (opcjonalny)

Każda konfiguracja jest uruchamiana dla seedów 0..4.
Wyniki (tabele) i wykresy zapisywane są do katalogu results/.

Uruchomienie:
    python run_experiments.py                  # wszystkie
    python run_experiments.py --exp A          # tylko architektury
    python run_experiments.py --exp B C        # wybrane
    python run_experiments.py --subset 10000   # szybsze testy na podzbiorze
    python run_experiments.py --jobs 4         # 4 procesy równolegle
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # backend bez GUI - bezpieczny w procesach równoległych
import matplotlib.pyplot as plt
import numpy as np

from mlp import MLP, History
from optimizers import SGD


# ---------------------------------------------------------------------------
# Konfiguracja globalna
# ---------------------------------------------------------------------------

MNIST_PATH = "data/mnist.npz"
RESULTS_DIR = Path("results")
SEEDS = [0, 1, 2, 3, 4]
DEFAULT_EPOCHS = 30
DEFAULT_BATCH = 64
DEFAULT_LR = 0.05
DEFAULT_MOMENTUM = 0.9
DEFAULT_ACT = "relu"
DEFAULT_ARCH = [784, 256, 128, 10]


# ---------------------------------------------------------------------------
# Pomocnicze
# ---------------------------------------------------------------------------

def load_mnist(path: str = MNIST_PATH, subset: int | None = None, seed: int = 0):
    data = np.load(path)
    X_train = data["X_train"].astype(np.float32) / 255.0
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float32) / 255.0
    y_test = data["y_test"]

    if subset is not None and subset < len(X_train):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X_train))[:subset]
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test


def arch_name(layers: list[int]) -> str:
    return "-".join(map(str, layers))


@dataclass
class RunResult:
    exp: str
    config_name: str
    seed: int
    layers: list[int]
    activation: str
    lr: float
    momentum: float
    batch_size: int
    epochs: int
    train_acc: float
    test_acc: float
    final_loss: float
    time_sec: float
    history_loss: list[float]
    history_metric: list[float]


def _accuracy_ylim(metric: list[float]) -> tuple[float, float]:
    """Dobiera dolną granicę osi Y accuracy z dyskretnego zestawu {0.0, 0.2, 0.4, 0.6, 0.8}.

    Dzięki temu wykresy są porównywalne, a jednocześnie nie marnują miejsca
    na zakresy, które nigdy nie występują.
    """
    if not metric:
        return 0.0, 1.02
    m = min(metric)
    for lo in (0.8, 0.6, 0.4, 0.2, 0.0):
        if m >= lo:
            return lo, 1.02
    return 0.0, 1.02


def save_learning_curves(
    history: History,
    path: Path,
    title: str,
):
    """Zapisuje wykres loss + accuracy z obiektu History (PDF)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    epochs = range(1, len(history.loss) + 1)

    ax1.plot(epochs, history.loss, label="train loss", color="C0")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history.metric, label="train acc", color="C1")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy (train)")
    ax2.set_ylim(*_accuracy_ylim(history.metric))
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(results: list[RunResult], group_key: str = "config_name") -> list[dict]:
    """Agreguje wyniki po konfiguracji: mean ± std test_acc, średni czas itd."""
    from collections import defaultdict

    groups: dict[str, list[RunResult]] = defaultdict(list)
    for r in results:
        groups[getattr(r, group_key)].append(r)

    summary = []
    for name, runs in sorted(groups.items()):
        test_accs = [r.test_acc for r in runs]
        times = [r.time_sec for r in runs]
        train_accs = [r.train_acc for r in runs]
        summary.append({
            "config": name,
            "n_seeds": len(runs),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "train_acc_mean": float(np.mean(train_accs)),
            "time_mean_sec": float(np.mean(times)),
            "test_acc_all": [round(a, 4) for a in test_accs],
        })
    return summary


def print_summary_table(summary: list[dict], title: str):
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    print(f"{'config':<28} {'test_acc':>12} {'train_acc':>10} {'time [s]':>10}")
    print("-" * 70)
    for s in summary:
        acc_str = f"{s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}"
        print(f"{s['config']:<28} {acc_str:>12} {s['train_acc_mean']:>10.4f} {s['time_mean_sec']:>10.1f}")
    print()


# ---------------------------------------------------------------------------
# Pojedynczy przebieg (wywoływany też w procesach roboczych)
# ---------------------------------------------------------------------------

def _limit_blas_threads(n: int = 1) -> None:
    """Ogranicza wątki BLAS w bieżącym procesie (musi być przed ciężkimi matmulami)."""
    n_str = str(max(1, n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = n_str
    # Próba twardego ustawienia po imporcie (działa dla części buildów OpenBLAS/MKL)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(n)
    except Exception:
        pass


def run_one(
    *,
    exp: str,
    config_name: str,
    layers: list[int],
    activation: str,
    lr: float,
    momentum: float,
    batch_size: int,
    epochs: int,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    plots_dir: str | Path,
) -> RunResult:
    plots_dir = Path(plots_dir)

    net = MLP(
        layers,
        hidden_activation=activation,
        task="classification",
        loss="cross_entropy",
        use_bias=True,
        seed=seed,
    )
    opt = SGD(lr=lr, momentum=momentum)

    t0 = time.perf_counter()
    history = net.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=opt,
    )
    elapsed = time.perf_counter() - t0

    train_acc = net.score(X_train, y_train)
    test_acc = net.score(X_test, y_test)
    final_loss = history.loss[-1] if history.loss else float("nan")

    # Każdy przebieg ma unikalną nazwę pliku → brak kolizji przy zapisie równoległym
    plot_path = plots_dir / f"{config_name}_seed{seed}.pdf"
    title = f"{config_name} | seed={seed} | test_acc={test_acc:.4f}"
    save_learning_curves(history, plot_path, title)

    return RunResult(
        exp=exp,
        config_name=config_name,
        seed=seed,
        layers=list(layers),
        activation=activation,
        lr=lr,
        momentum=momentum,
        batch_size=batch_size,
        epochs=epochs,
        train_acc=train_acc,
        test_acc=test_acc,
        final_loss=final_loss,
        time_sec=elapsed,
        history_loss=list(history.loss),
        history_metric=list(history.metric),
    )


def _pool_initializer(blas_threads: int = 1) -> None:
    """Uruchamiane raz w każdym procesie roboczym - zanim polecą joby."""
    _limit_blas_threads(blas_threads)
    matplotlib.use("Agg")


def _run_one_job(payload: dict) -> RunResult:
    """Worker: ładuje MNIST z dysku (bez pickle wielkich tablic) i trenuje."""
    X_train, y_train, X_test, y_test = load_mnist(
        path=payload["mnist_path"],
        subset=payload.get("subset"),
        seed=0,
    )
    return run_one(
        exp=payload["exp"],
        config_name=payload["config_name"],
        layers=payload["layers"],
        activation=payload["activation"],
        lr=payload["lr"],
        momentum=payload["momentum"],
        batch_size=payload["batch_size"],
        epochs=payload["epochs"],
        seed=payload["seed"],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        plots_dir=payload["plots_dir"],
    )


# ---------------------------------------------------------------------------
# Definicje eksperymentów
# ---------------------------------------------------------------------------

def configs_A(epochs: int = DEFAULT_EPOCHS) -> list[dict[str, Any]]:
    """Eksperyment A - architektury."""
    archs = [
        [784, 128, 10],
        [784, 256, 128, 10],          # baseline
        [784, 256, 128, 64, 10],
        [784, 512, 256, 10],
        [784, 1024, 10],
    ]
    return [
        {
            "config_name": arch_name(a),
            "layers": a,
            "activation": DEFAULT_ACT,
            "lr": DEFAULT_LR,
            "momentum": DEFAULT_MOMENTUM,
            "batch_size": DEFAULT_BATCH,
            "epochs": epochs,
        }
        for a in archs
    ]


def configs_B(
    best_layers: list[int] | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> list[dict[str, Any]]:
    """Eksperyment B - funkcje aktywacji."""
    layers = best_layers or DEFAULT_ARCH
    acts = ["relu", "tanh", "sigmoid"]
    return [
        {
            "config_name": f"{arch_name(layers)}_{act}",
            "layers": layers,
            "activation": act,
            "lr": DEFAULT_LR,
            "momentum": DEFAULT_MOMENTUM,
            "batch_size": DEFAULT_BATCH,
            "epochs": epochs,
        }
        for act in acts
    ]


def configs_C(
    best_layers: list[int] | None = None,
    best_act: str | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> list[dict[str, Any]]:
    """Eksperyment C - learning rate i momentum."""
    layers = best_layers or DEFAULT_ARCH
    act = best_act or DEFAULT_ACT
    variants = [
        (0.01, 0.9),
        (0.05, 0.9),   # baseline
        (0.10, 0.9),
        (0.05, 0.0),   # bez momentum
        (0.05, 0.99),
    ]
    return [
        {
            "config_name": f"lr{lr}_mom{mom}",
            "layers": layers,
            "activation": act,
            "lr": lr,
            "momentum": mom,
            "batch_size": DEFAULT_BATCH,
            "epochs": epochs,
        }
        for lr, mom in variants
    ]


def configs_D(
    best_layers: list[int] | None = None,
    best_act: str | None = None,
    epochs: int = DEFAULT_EPOCHS,
) -> list[dict[str, Any]]:
    """Eksperyment D - batch size."""
    layers = best_layers or DEFAULT_ARCH
    act = best_act or DEFAULT_ACT
    batches = [32, 64, 128, 256]
    return [
        {
            "config_name": f"batch{bs}",
            "layers": layers,
            "activation": act,
            "lr": DEFAULT_LR,
            "momentum": DEFAULT_MOMENTUM,
            "batch_size": bs,
            "epochs": epochs,
        }
        for bs in batches
    ]


# ---------------------------------------------------------------------------
# Uruchomienie eksperymentu
# ---------------------------------------------------------------------------

def run_experiment(
    exp_id: str,
    configs: list[dict[str, Any]],
    X_train, y_train, X_test, y_test,
    seeds: list[int] = SEEDS,
    jobs: int = 1,
    mnist_path: str = MNIST_PATH,
    subset: int | None = None,
):
    exp_dir = RESULTS_DIR / f"exp_{exp_id}"
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    jobs = max(1, int(jobs))
    total = len(configs) * len(seeds)

    print(
        f"\n>>> Eksperyment {exp_id}: {len(configs)} konfiguracji × {len(seeds)} seedów "
        f"= {total} przebiegów  (jobs={jobs})"
    )

    # Lekki payload (bez tablic) - worker sam ładuje mnist.npz z dysku
    payloads: list[dict] = []
    for cfg in configs:
        for seed in seeds:
            payloads.append({
                "exp": exp_id,
                "config_name": cfg["config_name"],
                "layers": cfg["layers"],
                "activation": cfg["activation"],
                "lr": cfg["lr"],
                "momentum": cfg["momentum"],
                "batch_size": cfg["batch_size"],
                "epochs": cfg["epochs"],
                "seed": seed,
                "plots_dir": str(plots_dir),
                "mnist_path": str(Path(mnist_path).resolve()),
                "subset": subset,
            })

    all_results: list[RunResult] = []

    if jobs == 1:
        for i, payload in enumerate(payloads, 1):
            print(
                f"  [{i}/{total}] {payload['config_name']}  seed={payload['seed']} ...",
                end=" ", flush=True,
            )
            # Sekwencyjnie: używamy już załadowanych tablic (bez ponownego I/O)
            result = run_one(
                exp=payload["exp"],
                config_name=payload["config_name"],
                layers=payload["layers"],
                activation=payload["activation"],
                lr=payload["lr"],
                momentum=payload["momentum"],
                batch_size=payload["batch_size"],
                epochs=payload["epochs"],
                seed=payload["seed"],
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                plots_dir=payload["plots_dir"],
            )
            all_results.append(result)
            print(f"test_acc={result.test_acc:.4f}  ({result.time_sec:.1f}s)")
    else:
        # 1 wątek BLAS na proces → jobs procesów ≈ jobs zajętych rdzeni (bez oversubscription)
        done = 0
        with ProcessPoolExecutor(
            max_workers=jobs,
            initializer=_pool_initializer,
            initargs=(1,),
        ) as pool:
            futures = {pool.submit(_run_one_job, p): p for p in payloads}
            for fut in as_completed(futures):
                result = fut.result()
                all_results.append(result)
                done += 1
                print(
                    f"  [{done}/{total}] {result.config_name}  seed={result.seed}  "
                    f"test_acc={result.test_acc:.4f}  ({result.time_sec:.1f}s)"
                )

    # stabilna kolejność: config, potem seed
    all_results.sort(key=lambda r: (r.config_name, r.seed))

    raw_rows = [
        {
            "config": r.config_name,
            "seed": r.seed,
            "layers": str(r.layers),
            "activation": r.activation,
            "lr": r.lr,
            "momentum": r.momentum,
            "batch_size": r.batch_size,
            "epochs": r.epochs,
            "train_acc": round(r.train_acc, 5),
            "test_acc": round(r.test_acc, 5),
            "final_loss": round(r.final_loss, 5),
            "time_sec": round(r.time_sec, 2),
        }
        for r in all_results
    ]

    # zapis surowych wyników (tylko w procesie głównym)
    write_csv(exp_dir / "raw_results.csv", raw_rows)

    # podsumowanie (mean ± std)
    summary = summarize(all_results)
    write_csv(exp_dir / "summary.csv", [
        {
            "config": s["config"],
            "n_seeds": s["n_seeds"],
            "test_acc_mean": round(s["test_acc_mean"], 5),
            "test_acc_std": round(s["test_acc_std"], 5),
            "train_acc_mean": round(s["train_acc_mean"], 5),
            "time_mean_sec": round(s["time_mean_sec"], 1),
            "test_acc_all": str(s["test_acc_all"]),
        }
        for s in summary
    ])

    # czytelny tekst
    with open(exp_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Eksperyment {exp_id}\n")
        f.write(f"Seeds: {seeds}\n")
        f.write(f"Liczba przebiegów: {len(all_results)}\n\n")
        f.write(f"{'config':<28} {'test_acc_mean':>14} {'std':>8} {'train_acc':>10} {'time_s':>8}\n")
        f.write("-" * 72 + "\n")
        for s in summary:
            f.write(
                f"{s['config']:<28} {s['test_acc_mean']:>14.4f} {s['test_acc_std']:>8.4f} "
                f"{s['train_acc_mean']:>10.4f} {s['time_mean_sec']:>8.1f}\n"
            )

    # zapis historii (do ewentualnego późniejszego rysowania uśrednionych krzywych)
    histories = {
        f"{r.config_name}_seed{r.seed}": {
            "loss": r.history_loss,
            "metric": r.history_metric,
            "test_acc": r.test_acc,
        }
        for r in all_results
    }
    with open(exp_dir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f)

    print_summary_table(summary, f"Podsumowanie eksperymentu {exp_id}")
    print(f"Wyniki zapisane w: {exp_dir}/")
    return all_results, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Eksperymenty MLP na MNIST")
    parser.add_argument(
        "--exp", nargs="*", default=["A", "B", "C", "D"],
        help="Które eksperymenty uruchomić (A B C D). Domyślnie wszystkie.",
    )
    parser.add_argument(
        "--subset", type=int, default=None,
        help="Opcjonalnie ogranicz zbiór uczący do N próbek (do szybkich testów).",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Liczba epok (domyślnie {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=SEEDS,
        help="Lista seedów (domyślnie 0 1 2 3 4).",
    )
    parser.add_argument(
        "--jobs", type=int, default=1,
        help="Liczba równoległych procesów (1 = sekwencyjnie). Np. --jobs 4",
    )
    args = parser.parse_args()

    epochs = args.epochs
    jobs = args.jobs

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Ładowanie MNIST...")
    X_train, y_train, X_test, y_test = load_mnist(subset=args.subset, seed=0)
    print(f"  train: {X_train.shape}, test: {X_test.shape}")
    print(f"  epoki: {args.epochs}, seedy: {args.seeds}, jobs: {jobs}")

    mnist_path = MNIST_PATH
    subset = args.subset

    # ----- A: architektury -----
    best_layers = DEFAULT_ARCH
    if "A" in args.exp:
        results_A, summary_A = run_experiment(
            "A_architecture", configs_A(epochs=epochs),
            X_train, y_train, X_test, y_test,
            seeds=args.seeds, jobs=jobs,
            mnist_path=mnist_path, subset=subset,
        )
        # wybierz najlepszą architekturę (po średnim test_acc)
        best = max(summary_A, key=lambda s: s["test_acc_mean"])
        best_layers = [int(x) for x in best["config"].split("-")]
        print(f"→ Najlepsza architektura z A: {best_layers}  (test_acc={best['test_acc_mean']:.4f})")

    # ----- B: aktywacje -----
    best_act = DEFAULT_ACT
    if "B" in args.exp:
        results_B, summary_B = run_experiment(
            "B_activation", configs_B(best_layers, epochs=epochs),
            X_train, y_train, X_test, y_test,
            seeds=args.seeds, jobs=jobs,
            mnist_path=mnist_path, subset=subset,
        )
        best = max(summary_B, key=lambda s: s["test_acc_mean"])
        # nazwa ma postać "784-256-128-10_relu"
        best_act = best["config"].rsplit("_", 1)[-1]
        print(f"→ Najlepsza aktywacja z B: {best_act}  (test_acc={best['test_acc_mean']:.4f})")

    # ----- C: lr / momentum -----
    if "C" in args.exp:
        run_experiment(
            "C_lr_momentum", configs_C(best_layers, best_act, epochs=epochs),
            X_train, y_train, X_test, y_test,
            seeds=args.seeds, jobs=jobs,
            mnist_path=mnist_path, subset=subset,
        )

    # ----- D: batch size -----
    if "D" in args.exp:
        run_experiment(
            "D_batch_size", configs_D(best_layers, best_act, epochs=epochs),
            X_train, y_train, X_test, y_test,
            seeds=args.seeds, jobs=jobs,
            mnist_path=mnist_path, subset=subset,
        )

    print("\nWszystkie wybrane eksperymenty zakończone.")
    print(f"Katalog wyników: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()