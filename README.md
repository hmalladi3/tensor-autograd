# tensor-autograd

A small deep learning framework built from scratch, with no runtime dependencies.

Two layers:

- a **tensor engine** in C++ — strided n-d arrays, broadcasting, blocked GEMM, the usual primitives, exposed through a small (26-symbol) C ABI;
- an **autograd + nn library** in Python — wraps the engine over `ctypes`, records a dynamic graph, runs reverse-mode autodiff, and provides `nn.Module`, `Linear`, `Sequential`, activations, losses, SGD, Adam.

No BLAS. No numpy. No pybind11. The Python side does no arithmetic of its own; every floating-point op crosses the FFI into C++.

## Demo

`python3 examples/mnist.py` downloads MNIST, builds a 784→128→10 MLP, and trains with Adam:

```
loading MNIST ...
  train: 60000 samples,  test: 10000 samples  (3.8s)
model: 784 → 128 → 10  (101,770 parameters)

epoch 1/3  train_loss=0.3108  test_acc=0.9520  (6.6s, 938 batches)
epoch 2/3  train_loss=0.1399  test_acc=0.9623  (6.6s, 938 batches)
epoch 3/3  train_loss=0.0978  test_acc=0.9678  (6.6s, 938 batches)

final test accuracy: 0.9678
```

96.78% test accuracy in 20 seconds of training on a laptop CPU.

## Build

Requires a C++20 compiler, CMake ≥3.20, Python ≥3.10 (stdlib only).

```sh
cmake -S . -B build
cmake --build build
```

The shared library lands in `tensor_autograd/` next to the Python package so `ctypes` can find it.

## Test

```sh
cd build && ctest                                       # C++ + Python via ctest
python3 -m unittest discover -s tests/python -v         # Python only
./build/tests/cpp/test_tensor_impl                       # one C++ binary
```

## Layout

```
include/tensor_engine.h     # public C ABI (26 symbols)
src/                        # engine internals
src/ops/                    # one .cpp per op family
tensor_autograd/            # Python package
tensor_autograd/nn/         # Module, Parameter, Linear, layers, losses, init
tensor_autograd/optim.py    # SGD, Adam
tests/cpp/                  # C++ unit tests (hand-rolled harness, ~110 lines)
tests/python/               # Python tests (stdlib unittest)
examples/mnist.py           # end-to-end demo
```
