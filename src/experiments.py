"""Eksperymenty: A - funkcje aktywacji, B - liczba warstw i neuronów, C - miary błędu.

Uruchomienie bez argumentów:  python experiments.py

Skrypt bierze wszystkie pary data.<nazwa>.train.<N>.csv + data.<nazwa>.test.<N>.csv z folderów
classification/ i regression/. Dla każdej pary i każdej konfiguracji sieci:
  1. z pliku train odkłada 20% na walidację,
  2. wybiera lr z LRS (po błędzie walidacyjnym, ziarno SEEDS[0]),
  3. uczy sieć dla wszystkich ziaren i liczy wyniki na pliku test (test nie służy do wyborów).
Na końcu wypisuje tabele i zapisuje je (z wykresami sieci referencyjnej) do folderu results/.
"""
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import viz
from data import Standardizer, load_csv, train_test_split
from mlp import MLP

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SEEDS = range(5)                # ziarna generatora liczb losowych
LRS = (0.1, 0.01, 0.001)        # sprawdzane kroki uczenia
MOMENTUM = 0.9                  # SGD z momentum (0 = zwykły SGD)
MIN_EPOCHS = 100
MIN_ITERATIONS = 3000           # tyle aktualizacji wag co najmniej (dla małych zbiorów epok jest więcej)
DEFAULT_LOSS = {"classification": "cross_entropy", "regression": "mse"}
REFERENCE = ("A", "tanh")       # konfiguracja, której wykresy zapisujemy
METRIC_NAMES = {("classification", "metric"): "accuracy (test)", ("classification", "metric2"): "cross-entropy (test)",
                ("regression", "metric"): "RMSE (test)", ("regression", "metric2"): "MAE (test)"}
EXPERIMENT_NAMES = {"A": "funkcje aktywacji", "B": "liczba warstw i neuronów", "C": "miary błędu"}


def configs(task):
    """Lista (eksperyment, etykieta, szerokości warstw ukrytych, aktywacja, strata)."""
    loss = DEFAULT_LOSS[task]
    width = 16 if task == "classification" else 32
    result = []
    for activation in ("sigmoid", "tanh", "relu"):                       # A
        result.append(("A", activation, [width, width], activation, loss))
    result.append(("B", "0 warstw", [], "tanh", loss))                   # B
    for n_layers in (1, 2, 3, 4):
        for w in (8, 32):
            result.append(("B", f"{n_layers}×{w}", [w] * n_layers, "tanh", loss))
    other_loss = "mse" if task == "classification" else "mae"            # C
    result.append(("C", loss, [width, width], "tanh", loss))
    result.append(("C", other_loss, [width, width], "tanh", other_loss))
    return result


class Data:
    """Dane jednej pary plików: uczący (80% train), walidacyjny (20% train) i testowy."""

    def __init__(self, task, train_path, test_path):
        self.task = task
        X, y = load_csv(train_path)
        self.X_test, self.y_test = load_csv(test_path)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, seed=0, stratify=(task == "classification"))
        self.x_scaler = Standardizer().fit(self.X_train)                  # skalery uczone tylko na zb. uczącym
        self.y_scaler = Standardizer().fit(self.y_train.reshape(-1, 1))   # używany tylko w regresji

    def inputs(self, X):
        return self.x_scaler.transform(X)

    def targets(self, y):
        """Klasyfikacja: etykiety bez zmian. Regresja: przeskalowana kolumna."""
        return y if self.task == "classification" else self.y_scaler.transform(y.reshape(-1, 1))


def train(data, hidden, activation, loss, lr, seed, snapshots=False):
    n_out = len(np.unique(data.y_train)) if data.task == "classification" else 1
    net = MLP([data.X_train.shape[1]] + hidden + [n_out], hidden_activation=activation,
              task=data.task, loss=loss, seed=seed)
    batch = max(32, len(data.X_train) // 30)
    epochs = max(MIN_EPOCHS, MIN_ITERATIONS // (len(data.X_train) // batch))
    history = net.fit(data.inputs(data.X_train), data.targets(data.y_train), epochs=epochs, batch_size=batch,
                      lr=lr, optimizer="sgd", optimizer_params={"momentum": MOMENTUM},
                      X_val=data.inputs(data.X_val), y_val=data.targets(data.y_val),
                      snapshot_every=max(1, epochs // 100) if snapshots else None)
    return net, history


def test_metrics(net, data):
    """(metryka główna, metryka dodatkowa) na zbiorze testowym.
    Klasyfikacja: accuracy i cross-entropy. Regresja: RMSE i MAE (w oryginalnych jednostkach y)."""
    X = data.inputs(data.X_test)
    if data.task == "classification":
        probabilities = net.predict_proba(X)
        true = np.searchsorted(net.classes_, data.y_test)
        accuracy = np.mean(probabilities.argmax(axis=1) == true)
        cross_entropy = -np.mean(np.log(np.clip(probabilities[np.arange(len(true)), true], 1e-12, 1.0)))
        return accuracy, cross_entropy
    prediction = data.y_scaler.inverse_transform(net.predict(X).reshape(-1, 1))[:, 0]
    error = prediction - data.y_test
    return np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error))


def run_once(data, hidden, activation, loss, lr, seed):
    net, history = train(data, hidden, activation, loss, lr, seed)
    if not (np.isfinite(history.loss[-1]) and np.isfinite(history.val_loss[-1])):
        return {"val_loss": np.inf, "metric": np.nan, "metric2": np.nan}
    metric, metric2 = test_metrics(net, data)
    return {"val_loss": history.val_loss[-1], "metric": metric, "metric2": metric2}


def run_config(data, hidden, activation, loss):
    """Wybiera lr (ziarno SEEDS[0], błąd walidacyjny), potem uczy dla wszystkich ziaren."""
    first = {lr: run_once(data, hidden, activation, loss, lr, SEEDS[0]) for lr in LRS}
    best_lr = min(LRS, key=lambda lr: first[lr]["val_loss"])
    results = [first[best_lr]] + [run_once(data, hidden, activation, loss, best_lr, seed) for seed in SEEDS[1:]]
    return best_lr, results


def save_figures(data, hidden, lr, name):
    """Wykresy sieci referencyjnej: wynik, błąd po iteracjach, ewolucja wag."""
    net, history = train(data, hidden, "tanh", DEFAULT_LOSS[data.task], lr, seed=0, snapshots=True)
    X, y = data.X_train[:500], data.y_train[:500]
    sample = np.random.default_rng(0).permutation(len(data.X_test))[:500]
    X_test, y_test = data.X_test[sample], data.y_test[sample]
    title = f"{name} (lr={lr})"
    if data.task == "classification":
        figure = viz.plot_classification(net, X, y, X_test, y_test, x_scaler=data.x_scaler, title=title)
    else:
        figure = viz.plot_regression(net, X, y, X_test, y_test, x_scaler=data.x_scaler,
                                     y_scaler=data.y_scaler, title=title)
    viz.save_figure(figure, OUT / "figures" / f"{name}_wynik.png")
    viz.save_figure(viz.plot_loss(history), OUT / "figures" / f"{name}_blad.png")
    viz.save_figure(viz.plot_weight_trajectories(history), OUT / "figures" / f"{name}_wagi_trajektorie.png")
    viz.save_figure(viz.plot_weight_heatmaps(history), OUT / "figures" / f"{name}_wagi_heatmapy.png")


def run_pair(task, train_path, test_path):
    dataset, n = re.fullmatch(r"data\.(.+)\.train\.(\d+)\.csv", train_path.name).groups()
    data = Data(task, train_path, test_path)
    rows, reference_lr, reference_hidden = [], None, None
    for experiment, label, hidden, activation, loss in configs(task):
        lr, results = run_config(data, hidden, activation, loss)
        for seed, result in zip(SEEDS, results):
            rows.append({"task": task, "dataset": dataset, "N": int(n), "exp": experiment,
                         "label": label, "lr": lr, "seed": seed, **result})
        if (experiment, label) == REFERENCE:
            reference_lr, reference_hidden = lr, hidden
    save_figures(data, reference_hidden, reference_lr, f"{task}_{dataset}_{n}")
    return rows


def make_tables(df):
    """Tabele: wiersze = zbiór i N, kolumny = konfiguracje, komórka = średnia ± odch. std (wybrane lr)."""
    text = [f"Uczenie na 80% pliku train, lr wybrane z {LRS} po błędzie walidacyjnym (20% train), ocena na pliku test. "
            f"SGD z momentum {MOMENTUM}, {len(SEEDS)} ziaren. Komórka: średnia ± odchylenie std (wybrane lr)."]
    for task in ("classification", "regression"):
        for experiment, experiment_name in EXPERIMENT_NAMES.items():
            part = df[(df["task"] == task) & (df["exp"] == experiment)]
            for column in ("metric", "metric2") if experiment == "C" else ("metric",):
                g = part.groupby(["dataset", "N", "label"], sort=False).agg(
                    mean=(column, "mean"), std=(column, "std"), lr=("lr", "first")).reset_index()
                g["cell"] = g["mean"].map("{:.4g}".format) + " ± " + g["std"].map("{:.2g}".format) \
                    + " (" + g["lr"].astype(str) + ")"
                table = g.pivot(index=["dataset", "N"], columns="label", values="cell")
                table = table[list(dict.fromkeys(part["label"]))]
                text.append(f"\n{experiment}. {experiment_name} - {task} - {METRIC_NAMES[(task, column)]}\n"
                            + table.to_string())
    return "\n".join(text)


def main():
    pairs = []
    for task in ("classification", "regression"):
        for train_path in sorted((ROOT / task).glob("data.*.train.*.csv")):
            test_path = train_path.with_name(train_path.name.replace(".train.", ".test."))
            if test_path.exists():
                pairs.append((task, train_path, test_path))
            else:
                print(f"Pomijam {train_path.name}: brak pliku {test_path.name}")
    if not pairs:
        raise SystemExit(f"Nie znaleziono plików data.*.train.*.csv w {ROOT / 'classification'} ani {ROOT / 'regression'}")

    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (task, train_path, test_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {task}/{train_path.name}", flush=True)
        rows += run_pair(task, train_path, test_path)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "results.csv", index=False)
    text = make_tables(df)
    (OUT / "results.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"\nZapisano: {OUT / 'results.txt'}, {OUT / 'results.csv'} i wykresy w {OUT / 'figures'}")


if __name__ == "__main__":
    main()