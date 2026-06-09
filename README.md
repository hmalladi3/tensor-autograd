# tensor-autograd

A small deep learning framework built from scratch, with **zero runtime dependencies** — no BLAS, no numpy, no Eigen, no pybind11.

Two layers, each idiomatic in its language, meeting at a narrow stable boundary:

- a **tensor engine** in C++ — strided n-dimensional arrays, views, broadcasting, dtype dispatch, a cache-blocked GEMM, the usual elementwise/reduction primitives — exposed through a small (26-symbol) C ABI;
- an **autograd + nn library** in Python — wraps the engine over `ctypes`, records a dynamic computation graph, runs reverse-mode autodiff, and provides `nn.Module`, `Linear`, `Sequential`, activations, losses, SGD, and Adam.

The Python side does **no arithmetic of its own**. Every floating-point operation — including each step of the backward pass and every optimizer update — crosses the FFI and is executed by the C++ engine. The split is the point: a fast numerical kernel layer separated from the graph/dispatch layer, exactly how production frameworks (PyTorch, JAX) are actually built, but with both halves in-tree and legible.

~7,000 lines: ~3,600 C++, ~3,500 Python.

## Demo

`python3 examples/mnist.py` downloads MNIST, builds a 784→128→10 MLP, and trains it with Adam — using nothing but this framework:

```
loading MNIST ...
  train: 60000 samples,  test: 10000 samples  (3.8s)
model: 784 → 128 → 10  (101,770 parameters)

epoch 1/3  train_loss=0.3108  test_acc=0.9520  (6.6s, 938 batches)
epoch 2/3  train_loss=0.1399  test_acc=0.9623  (6.6s, 938 batches)
epoch 3/3  train_loss=0.0978  test_acc=0.9678  (6.6s, 938 batches)

final test accuracy: 0.9678
```

96.78% test accuracy in ~20 seconds of training on a laptop CPU, with no `pip install` beyond CPython itself.

```python
import tensor_autograd as ta
from tensor_autograd import nn, optim

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
opt = optim.Adam(model.parameters(), lr=1e-3)

logits = model(x)                       # forward: builds the tape
loss = nn.functional.cross_entropy(logits, y)
opt.zero_grad()
loss.backward()                         # reverse-mode autodiff
opt.step()                              # param -= lr * m̂ / (√v̂ + eps)
```

## Architecture

```
┌───────────────────────────── Python — ML library ──────────────────────────────┐
│                                                                                  │
│   user code ──▶ nn.Module / nn.Sequential ──▶ Tensor (wrapper: handle + grad)    │
│                       │                              │                            │
│                   optimizer (SGD/Adam)          autograd tape                     │
│                       │                              │  Function.apply records    │
│                       │                              │  a _Node per op; .backward │
│                       └──────────────┬───────────────┘  walks it in reverse       │
│                                      ▼                                            │
└──────────────────────────────── ctypes FFI ─────────────────────────────────────┘
                                       │   26 C symbols (tensor_*, op_*)
┌──────────────────────────────── C ABI boundary ─────────────────────────────────┐
│   opaque tensor_handle_t · tensor_status_t error codes · explicit refcounting    │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │
┌───────────────────────────── C++ — tensor engine ──────────────────────────────┐
│                                                                                  │
│   TensorImpl (shape + strides + offset + dtype)                                  │
│        │ shares                                                                   │
│        ▼                                                                          │
│   Storage (refcounted raw buffer) ──▶ Allocator (aligned malloc/free)            │
│                                                                                  │
│   Ops:  elementwise · reductions · broadcasting · cache-blocked matmul · gather  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Signal flow (forward + backward)

1. User writes `y = model(x)`. Each layer's `forward` builds new `Tensor` objects.
2. Every `Tensor` op calls a C function via `ctypes`, gets back a fresh engine handle, and wraps it.
3. The same call also records a node on the autograd tape: `(Function, Context, inputs)`.
4. `loss.backward()` topologically sorts the reachable graph and walks it in reverse, calling each node's `backward` (its vjp) and accumulating gradients into leaf `.grad` slots — *every* numerical step still routes through the C++ engine, inside a `no_grad()` region.
5. The optimizer iterates `model.parameters()` and applies the update rule — again through the engine.

## The FFI boundary

The boundary is the design exercise, so it is deliberately small and explicit. `include/tensor_engine.h` is the entire contract — 26 functions, printable on one page:

```c
/* Lifetime */            tensor_incref, tensor_decref
/* Construction */        tensor_empty, tensor_full, tensor_from_buffer,
                          tensor_arange, tensor_random, tensor_seed
/* Metadata */            tensor_ndim, tensor_numel, tensor_dtype,
                          tensor_shape, tensor_strides
/* Egress */              tensor_copy_to_buffer
/* Views */               tensor_reshape, tensor_transpose, tensor_slice,
                          tensor_contiguous, tensor_cast
/* Ops */                 op_binary, op_unary, op_reduce, op_matmul,
                          op_gather, op_inplace
/* Errors */              tensor_last_error
```

Two rules carry the whole boundary:

- **Refcount discipline.** Returned handles have refcount 1; the caller owns exactly one `tensor_decref`. Argument handles are borrowed and never mutated. Views share `Storage`, not `TensorImpl`. `from_buffer`/`copy_to_buffer` always copy — no raw engine pointer ever escapes into Python, and no Python buffer is ever retained by the engine.
- **Error model.** Every fallible call returns a `tensor_status_t` (0 = OK). On error, output handles are nulled and a thread-local `tensor_last_error()` string is set, which the Python layer raises as an exception.

### Why a hand-rolled C ABI instead of pybind11?

pybind11 is the obvious, defensible choice. Choosing a C ABI + `ctypes` instead:

- keeps the dependency count *honest* — zero, not "zero except pybind";
- makes the boundary **visible** — every function Python can call is a symbol in a header you can read in two minutes;
- forces explicit ownership and lifetime decisions instead of letting smart-pointer magic hide them.

The cost is more boilerplate and manual refcounting at the seam. Accepted, on purpose.

## The tensor engine (C++)

- **Storage / TensorImpl split.** `Storage` owns a refcounted, aligned raw buffer and a dtype. `TensorImpl` is a *view onto* a `Storage`: shape, strides, offset. Reshape / transpose / slice return new `TensorImpl`s sharing the same `Storage` where possible (zero-copy), materializing a contiguous copy only when forced (`contiguous()`).
- **Strided, numpy-semantics tensors.** Everything is shape + strides + offset + dtype + a contiguous flag. Broadcasting is implemented *once* in the engine (numpy rules) rather than re-emulated per op.
- **Dtype dispatch.** `float32`, `int64`, `bool`, dispatched through a small op-id table (`src/op_ids.h`) shared numerically across the FFI. One `.cpp` per op family under `src/ops/`.
- **Cache-blocked GEMM.** `op_matmul` is a hand-rolled blocked kernel (64×64×32 tiles) that reads arbitrary input strides directly — so it multiplies transposed views without first materializing a copy — and pins the K-accumulator in a register on the innermost loop. No BLAS, no assembly, no autotuning; correctness first, the free wins taken.

## The autograd engine (Python)

Define-by-run, PyTorch-style. The tape is *implicit*: each differentiable op is a `Function` subclass with static `forward`/`backward`, and `Function.apply` records a `_Node(fn, ctx, inputs)` onto the output's `_grad_fn` whenever grad is enabled and some input requires grad.

- **`backward()`** topologically sorts the DAG reachable from the output, then walks it in reverse, accumulating gradients into a `pending` map and finally into each leaf's `.grad`. Multi-path gradients sum correctly; the entire pass runs inside `no_grad()` so computing gradients doesn't itself build a graph.
- **Broadcasting-aware vjps.** A shared `_unbroadcast` helper collapses each gradient back to its pre-broadcast input shape, so broadcasting "just works" through the backward pass too.
- **Coverage.** add/sub/mul/div/pow/max/min, neg/exp/log/sqrt/relu/sigmoid/tanh, sum/mean/max/argmax reductions, matmul, and the view ops (reshape, transpose, contiguous). Comparisons, cast, slice, and gather are intentionally non-differentiable in v1.
- **`nn` & `optim`.** `Module`/`Parameter` system, `Linear` (Kaiming init), `Sequential`, activation modules, `mse_loss` and a numerically-stable `cross_entropy`; `SGD` (with momentum) and `Adam` (bias-corrected), both stepping through the engine under `no_grad()`.

## Build

Requires a C++20 compiler, CMake ≥ 3.20, and Python ≥ 3.10 (stdlib only).

```sh
cmake -S . -B build
cmake --build build
```

The shared library lands in `tensor_autograd/` next to the Python package so `ctypes` finds it with no install step.

## Test

124 Python tests and 11 C++ test binaries, all hand-rolled (no pytest, no GoogleTest):

```sh
cd build && ctest                                  # C++ unit tests (11/11)
python3 -m unittest discover -s tests/python -v    # Python tests (124)
./build/tests/cpp/test_tensor_impl                 # a single C++ binary
```

## Layout

```
include/tensor_engine.h     # public C ABI (26 symbols, the whole contract)
src/                        # engine internals: storage, tensor_impl, broadcast, prng, ffi
src/ops/                    # one .cpp per op family (binary, unary, reduce, matmul, gather, …)
tensor_autograd/            # Python package: Tensor wrapper, ctypes bindings
tensor_autograd/autograd.py # reverse-mode tape (Function/Context/_Node, backward)
tensor_autograd/nn/         # Module, Parameter, Linear, Sequential, activations, losses, init
tensor_autograd/optim.py    # SGD, Adam
tests/cpp/                  # C++ unit tests (hand-rolled harness)
tests/python/               # Python tests (stdlib unittest)
examples/mnist.py           # end-to-end MNIST demo
```

## Design decisions

- **Two-language split, not pure-Python or pure-C++.** Pure Python would be too slow to be interesting and wouldn't demonstrate the systems half; pure C++ would be authentic but illegible and tedious to write models in. The boundary itself is the exercise.
- **Dynamic graph (define-by-run), not static.** Models read straightforwardly and are debuggable. The framework exists for *understanding*.
- **Reverse-mode autodiff only.** Forward-mode would double the surface area for no benefit at the goal (training an MLP).
- **Correctness over speed, speed taken where free.** Loop ordering, contiguous access, register-pinned accumulation — but no SIMD intrinsics-chasing, no autotuning, no assembly.

### Non-goals

GPU/CUDA, a `pip install`-able public API, matching PyTorch's op surface, distributed/mixed-precision/quantization, and beating numpy on benchmarks. This is a reference framework, not a product.

## FAQ

**Why not numpy under the hood?** Then the engine isn't an engine — it's a wrapper. The point of writing a tensor library is writing the loops.

**Why MNIST?** It's the smallest end-to-end demonstration that the *whole* stack works together: shapes, broadcasting, matmul, softmax, cross-entropy, a real reverse-mode tape, and an optimizer. XOR doesn't exercise enough; CIFAR/transformers are out of scope.

**Why C++ and not C?** Templates for dtype dispatch, RAII for the engine's few internal allocations, `std::vector`/`std::span` for shape and stride bookkeeping. The hot loops themselves are essentially C.

## References

- **micrograd** (Karpathy) — scalar autograd, the pedagogical gold standard for the *graph* half.
- **tinygrad** — production-shaped, but leans on numpy/accelerators for the kernels.
- **PyTorch internals** — the architectural reference for the strided-tensor + tape-autograd design.
- **The CPython C API docs** — for FFI shape and refcount discipline at the boundary.
