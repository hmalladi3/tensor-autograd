"""Raw ctypes binding to libtensor_engine.

This module is private. It loads the engine shared library once at import time,
sets argtypes/restype for every exported symbol, and exposes the dtype/op_id
enums and an EngineError exception type.

The package's public surface lives in ``tensor_autograd.tensor`` and friends.
"""

from __future__ import annotations

import ctypes
import os
import sys
from enum import IntEnum

# --------------------------------------------------------------------------- #
# Library loading                                                             #
# --------------------------------------------------------------------------- #


def _find_library() -> str:
    """Locate libtensor_engine.{so,dylib,dll} next to this package on disk."""
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "darwin":
        name = "libtensor_engine.dylib"
    elif sys.platform == "win32":
        name = "tensor_engine.dll"
    else:
        name = "libtensor_engine.so"
    path = os.path.join(here, name)
    if not os.path.exists(path):
        raise ImportError(
            f"tensor_autograd: engine library not found at {path}.\n"
            "Build it first with: cmake -S . -B build && cmake --build build"
        )
    return path


lib = ctypes.CDLL(_find_library())

# --------------------------------------------------------------------------- #
# Enums                                                                       #
# --------------------------------------------------------------------------- #


class DType(IntEnum):
    """Engine dtype tags. Values match the C ABI's DTYPE_* constants."""
    FLOAT32 = 0
    INT64 = 1
    BOOL = 2


_DTYPE_SIZE = {DType.FLOAT32: 4, DType.INT64: 8, DType.BOOL: 1}


def dtype_size(dt: DType) -> int:
    return _DTYPE_SIZE[dt]


class BinaryOp(IntEnum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    POW = 4
    MAX = 5
    MIN = 6
    EQ = 7
    LT = 8
    GT = 9


class UnaryOp(IntEnum):
    NEG = 0
    EXP = 1
    LOG = 2
    SQRT = 3
    RELU = 4
    SIGMOID = 5
    TANH = 6


class ReduceOp(IntEnum):
    SUM = 0
    MEAN = 1
    MAX = 2
    ARGMAX = 3


class RandomOp(IntEnum):
    UNIFORM = 0
    NORMAL = 1


class InplaceOp(IntEnum):
    AXPY = 0


# --------------------------------------------------------------------------- #
# Error model                                                                 #
# --------------------------------------------------------------------------- #


class EngineError(RuntimeError):
    """Raised when the C engine returns a nonzero status code."""


def _check(status: int) -> None:
    """Translate a nonzero status into an EngineError carrying last_error."""
    if status != 0:
        raw = lib.tensor_last_error()
        msg = raw.decode("utf-8", errors="replace") if raw else "(no message)"
        raise EngineError(f"engine status {status}: {msg}")


# --------------------------------------------------------------------------- #
# Function signatures                                                         #
# --------------------------------------------------------------------------- #
# Setting argtypes/restype explicitly is the difference between safe ctypes
# and silent memory corruption. Every symbol below mirrors the declaration in
# include/tensor_engine.h exactly.

_Handle = ctypes.c_void_p
_PHandle = ctypes.POINTER(_Handle)
_Status = ctypes.c_int32
_Dtype = ctypes.c_int32
_I64 = ctypes.c_int64
_PI64 = ctypes.POINTER(ctypes.c_int64)

# Error reporting
lib.tensor_last_error.argtypes = []
lib.tensor_last_error.restype = ctypes.c_char_p

# Lifetime
lib.tensor_incref.argtypes = [_Handle]
lib.tensor_incref.restype = None
lib.tensor_decref.argtypes = [_Handle]
lib.tensor_decref.restype = None

# Construction
lib.tensor_empty.argtypes = [_PI64, _I64, _Dtype, _PHandle]
lib.tensor_empty.restype = _Status

lib.tensor_full.argtypes = [_PI64, _I64, _Dtype, ctypes.c_double, _PHandle]
lib.tensor_full.restype = _Status

lib.tensor_from_buffer.argtypes = [ctypes.c_void_p, _PI64, _I64, _Dtype, _PHandle]
lib.tensor_from_buffer.restype = _Status

lib.tensor_arange.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double, _Dtype, _PHandle,
]
lib.tensor_arange.restype = _Status

lib.tensor_random.argtypes = [
    ctypes.c_int32, _PI64, _I64, _Dtype,
    ctypes.c_double, ctypes.c_double, _PHandle,
]
lib.tensor_random.restype = _Status

lib.tensor_seed.argtypes = [ctypes.c_uint64]
lib.tensor_seed.restype = None

# Metadata
lib.tensor_ndim.argtypes = [_Handle]
lib.tensor_ndim.restype = _I64
lib.tensor_numel.argtypes = [_Handle]
lib.tensor_numel.restype = _I64
lib.tensor_dtype.argtypes = [_Handle]
lib.tensor_dtype.restype = _Dtype
lib.tensor_shape.argtypes = [_Handle, _PI64]
lib.tensor_shape.restype = None
lib.tensor_strides.argtypes = [_Handle, _PI64]
lib.tensor_strides.restype = None

# Data egress
lib.tensor_copy_to_buffer.argtypes = [_Handle, ctypes.c_void_p, ctypes.c_size_t]
lib.tensor_copy_to_buffer.restype = _Status

# Views
lib.tensor_reshape.argtypes = [_Handle, _PI64, _I64, _PHandle]
lib.tensor_reshape.restype = _Status
lib.tensor_transpose.argtypes = [_Handle, _I64, _I64, _PHandle]
lib.tensor_transpose.restype = _Status
lib.tensor_slice.argtypes = [_Handle, _I64, _I64, _I64, _I64, _PHandle]
lib.tensor_slice.restype = _Status
lib.tensor_contiguous.argtypes = [_Handle, _PHandle]
lib.tensor_contiguous.restype = _Status
lib.tensor_cast.argtypes = [_Handle, _Dtype, _PHandle]
lib.tensor_cast.restype = _Status

# Ops
lib.op_binary.argtypes = [ctypes.c_int32, _Handle, _Handle, _PHandle]
lib.op_binary.restype = _Status
lib.op_unary.argtypes = [ctypes.c_int32, _Handle, _PHandle]
lib.op_unary.restype = _Status
lib.op_reduce.argtypes = [
    ctypes.c_int32, _Handle, _PI64, _I64, ctypes.c_int32, _PHandle,
]
lib.op_reduce.restype = _Status
lib.op_matmul.argtypes = [_Handle, _Handle, _PHandle]
lib.op_matmul.restype = _Status
lib.op_gather.argtypes = [_Handle, _I64, _Handle, _PHandle]
lib.op_gather.restype = _Status
lib.op_inplace.argtypes = [ctypes.c_int32, _Handle, _Handle, ctypes.c_double]
lib.op_inplace.restype = _Status
