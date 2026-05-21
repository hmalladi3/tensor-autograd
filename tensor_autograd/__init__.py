"""tensor-autograd: a zero-dependency deep learning framework built from scratch.

Two architectural layers:

* a **Tensor Engine** in C++ exposed via a small C ABI (see
  ``include/tensor_engine.h``);
* this Python layer, which wraps the engine in a ``Tensor`` type, builds an
  autograd tape on top of it, and (in upcoming modules) provides neural-network
  primitives and optimizers.

The Python side does no floating-point arithmetic of its own — every
``Tensor`` operation crosses the FFI into C++.
"""

from ._ffi import DType, EngineError
from ._tensor import (
    Tensor,
    arange,
    empty,
    full,
    manual_seed,
    ones,
    ones_like,
    randn,
    tensor,
    uniform,
    zeros,
    zeros_like,
)

# Importing autograd installs the autograd-aware op dispatchers into
# ``_tensor`` and attaches ``Tensor.backward``. Order matters — this must
# run after ``_tensor`` is fully imported.
from .autograd import Function, no_grad  # noqa: E402
from . import nn, optim  # noqa: E402,F401

__version__ = "0.1.0"

__all__ = [
    "DType",
    "EngineError",
    "Tensor",
    "Function",
    "arange",
    "empty",
    "full",
    "manual_seed",
    "no_grad",
    "ones",
    "ones_like",
    "randn",
    "tensor",
    "uniform",
    "zeros",
    "zeros_like",
]
