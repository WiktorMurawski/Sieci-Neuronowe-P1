"""Pobiera oryginalny zbiór MNIST (Yann LeCun) i zapisuje go jako mnist.npz.

Uruchomienie:
    python download_mnist.py

Jeśli plik mnist.npz już istnieje, nic nie robi.
"""
import os
import struct
import gzip
import urllib.request

import numpy as np

BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

OUT_DIR = "data/"
OUT_FILENAME = "mnist.npz"
OUT_PATH = OUT_DIR + OUT_FILENAME

def download(url: str, path: str) -> None:
    print(f"  pobieram {os.path.basename(path)} ...", end=" ", flush=True)
    urllib.request.urlretrieve(url, path)
    print("OK")


def load_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols)


def load_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


def main():
    if os.path.exists(OUT_PATH):
        print(f"{OUT_PATH} już istnieje - pomijam pobieranie.")
        return

    print("Pobieranie oryginalnego MNIST...")
    for name, fname in FILES.items():
        if not os.path.exists(OUT_DIR+fname):
            download(BASE_URL + fname, OUT_DIR+fname)

    print("Konwersja do tablic NumPy...")
    X_train = load_images(OUT_DIR+FILES["train_images"])
    y_train = load_labels(OUT_DIR+FILES["train_labels"])
    X_test = load_images(OUT_DIR+FILES["test_images"])
    y_test = load_labels(OUT_DIR+FILES["test_labels"])

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    print(f"Zapisywanie {OUT_PATH} ...")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    print("Gotowe.")


if __name__ == "__main__":
    main()
