"""Train a small MLP on MNIST using tensor-autograd.

End-to-end proof of life for the framework:

  784 → 128 → 10  feed-forward network
  ReLU activation
  Adam optimiser
  Cross-entropy loss

No external runtime dependencies — the dataset is downloaded from a public
mirror with stdlib ``urllib``, parsed from the IDX binary format with
``struct`` + ``gzip``, and shoved into the engine via ``ctypes`` for a
direct memcpy (the only way to move 47M float values without spending a
minute on Python overhead).

Run from the project root:

    cmake --build build         # build the engine if you haven't
    python3 examples/mnist.py

Expected runtime on a recent laptop: ~1 minute for the download (first run
only), then ~30–90 seconds per epoch.
"""

from __future__ import annotations

import array
import ctypes
import gzip
import struct
import sys
import time
import urllib.request
from pathlib import Path

# Allow running `python3 examples/mnist.py` from the project root without
# needing to `pip install -e .` first. We prepend the repo root to sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import tensor_autograd as tag
from tensor_autograd import nn, no_grad, optim
from tensor_autograd._ffi import DType, _check, lib
from tensor_autograd._tensor import _wrap
from tensor_autograd.nn import functional as F


# --------------------------------------------------------------------------- #
# Data                                                                        #
# --------------------------------------------------------------------------- #

DATA_DIR = Path(__file__).resolve().parent / "data" / "mnist"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def _download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        dst = DATA_DIR / fname
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = f"{MIRROR}/{fname}"
        sys.stderr.write(f"downloading {url} ...\n")
        urllib.request.urlretrieve(url, dst)


def _load_images(path: Path) -> tag.Tensor:
    """Parse an IDX image file. Returns (N, 784) Float32 with pixels in [0, 1]."""
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic {magic} in {path}")
        raw = f.read()
    feats = rows * cols
    inv255 = 1.0 / 255.0

    # uint8 → float32 normalization. We use array.array for a contiguous
    # C-level buffer; the generator avoids an intermediate Python list. The
    # ~47M-iteration cost is the dominant load time but only paid once per
    # run (and once per epoch shuffle, if added later).
    floats = array.array("f", (b * inv255 for b in raw))
    addr, _ = floats.buffer_info()

    shape = (ctypes.c_int64 * 2)(n, feats)
    handle = ctypes.c_void_p()
    _check(lib.tensor_from_buffer(
        ctypes.c_void_p(addr), shape, 2, int(DType.FLOAT32), ctypes.byref(handle)
    ))
    return _wrap(handle)


def _load_labels(path: Path) -> tag.Tensor:
    """Parse an IDX label file. Returns (N,) Int64."""
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic {magic} in {path}")
        raw = f.read()
    # uint8 → int64 — only n values (≤ 60K), the loop is cheap.
    int64s = array.array("q", list(raw))
    addr, _ = int64s.buffer_info()

    shape = (ctypes.c_int64 * 1)(n)
    handle = ctypes.c_void_p()
    _check(lib.tensor_from_buffer(
        ctypes.c_void_p(addr), shape, 1, int(DType.INT64), ctypes.byref(handle)
    ))
    return _wrap(handle)


def load_mnist():
    _download()
    return {
        "X_train": _load_images(DATA_DIR / FILES["train_images"]),
        "Y_train": _load_labels(DATA_DIR / FILES["train_labels"]),
        "X_test":  _load_images(DATA_DIR / FILES["test_images"]),
        "Y_test":  _load_labels(DATA_DIR / FILES["test_labels"]),
    }


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


# --------------------------------------------------------------------------- #
# Train + evaluate                                                            #
# --------------------------------------------------------------------------- #


def evaluate(model: nn.Module, X: tag.Tensor, Y: tag.Tensor, batch_size: int = 512) -> float:
    """Top-1 accuracy over the dataset."""
    n = X.shape[0]
    correct = 0
    with no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            xb = X.slice(0, start, stop, 1)
            yb = Y.slice(0, start, stop, 1)
            preds = model(xb).argmax(axes=1)
            for a, b in zip(preds.tolist(), yb.tolist()):
                if a == b:
                    correct += 1
    return correct / n


def train(model, data, epochs=3, batch_size=64, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    X, Y = data["X_train"], data["Y_train"]
    n_train = X.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        running = 0.0
        seen = 0

        for start in range(0, n_train, batch_size):
            stop = min(start + batch_size, n_train)
            xb = X.slice(0, start, stop, 1)
            yb = Y.slice(0, start, stop, 1)

            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

            running += loss.item() * (stop - start)
            seen += stop - start

        train_loss = running / seen
        test_acc = evaluate(model, data["X_test"], data["Y_test"])
        dt = time.time() - t0
        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"test_acc={test_acc:.4f}  "
            f"({dt:.1f}s, {n_batches} batches)"
        )

    return test_acc


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main():
    tag.manual_seed(42)

    print("loading MNIST ...")
    t0 = time.time()
    data = load_mnist()
    print(
        f"  train: {data['X_train'].shape[0]} samples,  "
        f"test: {data['X_test'].shape[0]} samples  "
        f"({time.time() - t0:.1f}s)"
    )

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: 784 → 128 → 10  ({n_params:,} parameters)")
    print()

    final_acc = train(model, data, epochs=3, batch_size=64, lr=1e-3)

    print()
    print(f"final test accuracy: {final_acc:.4f}")
    if final_acc < 0.95:
        sys.exit("target test accuracy (0.95) not reached")


if __name__ == "__main__":
    main()
