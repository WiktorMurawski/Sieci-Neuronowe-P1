"""Wczytywanie danych, podział na zbiory i standaryzacja (tylko numpy)."""
import numpy as np


def load_csv(path):
    """Wczytuje CSV z nagłówkiem; ostatnia kolumna to cel.

    Zwraca X (n, d) i y (n,). Jeśli kolumna celu nazywa się cls/class/label,
    y jest zwracane jako int (klasyfikacja), w przeciwnym razie jako float.
    """
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    X, y = arr[:, :-1], arr[:, -1]
    if header[-1].strip().lower() in ("cls", "class", "label"):
        y = y.astype(int)
    return X, y


def train_test_split(X, y, test_size=0.2, seed=None, stratify=False):
    """Losowy (powtarzalny dzięki seed) podział; stratify=True zachowuje proporcje klas."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if stratify:
        parts = []
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            rng.shuffle(idx)
            parts.append(idx[: int(round(len(idx) * test_size))])
        test_idx = np.concatenate(parts)
    else:
        test_idx = rng.permutation(n)[: int(round(n * test_size))]
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    train_idx = rng.permutation(np.flatnonzero(~mask))
    test_idx = rng.permutation(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class Standardizer:
    """(x - średnia) / odch. std. po kolumnach"""

    def fit(self, A):
        A = np.asarray(A, dtype=float)
        self.mean_ = A.mean(axis=0)
        std = A.std(axis=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self

    def transform(self, A):
        return (np.asarray(A, dtype=float) - self.mean_) / self.std_

    def inverse_transform(self, A):
        return np.asarray(A, dtype=float) * self.std_ + self.mean_

    def fit_transform(self, A):
        return self.fit(A).transform(A)
