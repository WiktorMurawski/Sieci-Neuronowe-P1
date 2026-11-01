"""Prototyp klasyfikatora cyfr MNIST z użyciem własnej implementacji MLP.

Użycie:
    python mnist_prototype.py

Parametry są ustawione tak, aby dało się szybko zobaczyć, czy wszystko działa.
Później można je zmieniać (liczba warstw/neuronów, aktywacja, lr, momentum, epochs...).
"""
import time
import numpy as np

from mlp import MLP
from optimizers import SGD


def load_mnist(path="data/mnist.npz"):
    data = np.load(path)
    X_train = data["X_train"].astype(np.float32) / 255.0
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float32) / 255.0
    y_test = data["y_test"]
    return X_train, y_train, X_test, y_test


def main():
    print("Ładowanie MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  train: {X_train.shape}, test: {X_test.shape}")

    # --- konfiguracja bazowa (prototyp) ---
    # Dla szybkości bierzemy podzbiór danych uczących.
    # Później można ustawić SUBSET = None, żeby uczyć na całych 60k.
    SUBSET = None         # None = cały zbiór
    SEED = 0
    LAYER_SIZES = [784, 128, 64, 10]
    HIDDEN_ACT = "relu"
    LR = 0.1
    MOMENTUM = 0.9
    EPOCHS = 15
    BATCH_SIZE = 64

    rng = np.random.default_rng(SEED)
    if SUBSET is not None:
        idx = rng.permutation(len(X_train))[:SUBSET]
        X_tr, y_tr = X_train[idx], y_train[idx]
        print(f"  używam podzbioru uczacego: {len(X_tr)} próbek")
    else:
        X_tr, y_tr = X_train, y_train

    print("\nKonfiguracja sieci:")
    print(f"  warstwy:           {LAYER_SIZES}")
    print(f"  aktywacja ukryta:  {HIDDEN_ACT}")
    print(f"  optymalizator:     SGD(lr={LR}, momentum={MOMENTUM})")
    print(f"  epochs / batch:    {EPOCHS} / {BATCH_SIZE}")
    print(f"  seed:              {SEED}")

    net = MLP(
        LAYER_SIZES,
        hidden_activation=HIDDEN_ACT,
        task="classification",
        loss="cross_entropy",
        use_bias=True,
        seed=SEED,
    )

    print("\nUczenie...")
    t0 = time.perf_counter()
    history = net.fit(
        X_tr, y_tr,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        optimizer=SGD(lr=LR, momentum=MOMENTUM),
    )
    elapsed = time.perf_counter() - t0
    print(f"  czas: {elapsed:.1f} s")

    # wyniki na zbiorze uczącym (podzbiorze) i testowym
    train_acc = net.score(X_tr, y_tr)
    test_acc = net.score(X_test, y_test)
    print(f"\nAccuracy (train subset): {train_acc:.4f}")
    print(f"Accuracy (test):         {test_acc:.4f}")

    # krótki przebieg uczenia
    print("\nPrzebieg (ostatnie 5 epok):")
    for i in range(max(0, len(history.loss) - 5), len(history.loss)):
        print(f"  epoka {i+1:3d}: loss={history.loss[i]:.4f}  acc={history.metric[i]:.4f}")

    # kilka przykładowych predykcji
    print("\nPrzykładowe predykcje (test):")
    preds = net.predict(X_test[:10])
    for i in range(10):
        print(f"  prawdziwa={y_test[i]}  przewidywana={preds[i]}")

    return net, history


if __name__ == "__main__":
    main()
