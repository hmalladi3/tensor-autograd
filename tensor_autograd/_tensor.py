"""User-facing Tensor class and module-level construction helpers.

Every arithmetic method here is a one-line dispatch into the FFI op layer —
no floating-point work happens in Python. The autograd integration (the
``Function.apply`` indirection that records graph nodes) lives in a separate
module and is wired in by importing it after this one; until then, ops run
forward only.
"""

from __future__ import annotations

import ctypes
from typing import Iterable, Optional, Sequence, Union

from . import _ffi
from ._ffi import (
    BinaryOp,
    DType,
    EngineError,
    InplaceOp,
    RandomOp,
    ReduceOp,
    UnaryOp,
    _check,
    dtype_size,
    lib,
)

# Bound at import time so __del__ can still call decref even when the module
# globals are being torn down during interpreter shutdown.
_tensor_decref = lib.tensor_decref

ShapeLike = Union[int, Sequence[int]]


# --------------------------------------------------------------------------- #
# Tensor                                                                      #
# --------------------------------------------------------------------------- #


class Tensor:
    """A handle to engine memory plus enough Python state for autograd.

    Construction goes through one of the module-level helpers (``tensor``,
    ``zeros``, ``randn``, …) or the result of an op. Calling ``Tensor(...)``
    directly is for the FFI bridge — you pass the raw engine handle and a
    description of what it points to.
    """

    __slots__ = (
        "_handle",
        "_dtype",
        "_shape",
        "requires_grad",
        "_grad",
        "_grad_fn",
    )

    def __init__(
        self,
        handle: ctypes.c_void_p,
        dtype: DType,
        shape: Sequence[int],
        requires_grad: bool = False,
    ) -> None:
        self._handle = handle
        self._dtype = dtype
        self._shape = tuple(shape)
        self.requires_grad = requires_grad
        self._grad: Optional["Tensor"] = None
        self._grad_fn = None  # populated by py-autograd

    def __del__(self) -> None:
        # Defensive: __del__ may run during interpreter shutdown, after the
        # module globals (including `_tensor_decref`) have been cleared.
        h = getattr(self, "_handle", None)
        if h is None or not h.value:
            return
        try:
            _tensor_decref(h)
        except Exception:
            pass
        self._handle = None

    # -------- metadata --------
    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def numel(self) -> int:
        n = 1
        for s in self._shape:
            n *= s
        return n

    def __len__(self) -> int:
        if not self._shape:
            raise TypeError("len() of a 0-dim tensor")
        return self._shape[0]

    def __repr__(self) -> str:
        return f"Tensor(shape={self._shape}, dtype={self._dtype.name})"

    # -------- arithmetic --------
    # Every dunder dispatches through module-level _do_* hooks. The autograd
    # module replaces those hooks at import time with versions that wrap each
    # op in Function.apply (and so build the implicit tape). When autograd
    # has not been loaded, these hooks default to raw FFI ops.
    def __add__(self, other):       return _do_binary(BinaryOp.ADD, self, _coerce(other, self._dtype))
    def __radd__(self, other):      return _do_binary(BinaryOp.ADD, _coerce(other, self._dtype), self)
    def __sub__(self, other):       return _do_binary(BinaryOp.SUB, self, _coerce(other, self._dtype))
    def __rsub__(self, other):      return _do_binary(BinaryOp.SUB, _coerce(other, self._dtype), self)
    def __mul__(self, other):       return _do_binary(BinaryOp.MUL, self, _coerce(other, self._dtype))
    def __rmul__(self, other):      return _do_binary(BinaryOp.MUL, _coerce(other, self._dtype), self)
    def __truediv__(self, other):   return _do_binary(BinaryOp.DIV, self, _coerce(other, self._dtype))
    def __rtruediv__(self, other):  return _do_binary(BinaryOp.DIV, _coerce(other, self._dtype), self)
    def __pow__(self, other):       return _do_binary(BinaryOp.POW, self, _coerce(other, self._dtype))
    def __neg__(self):              return _do_unary(UnaryOp.NEG, self)
    def __matmul__(self, other):    return _do_matmul(self, other)

    # Elementwise comparisons return Bool tensors. They override the default
    # identity-based __eq__, which makes tensors unhashable (intentional —
    # PyTorch / numpy do the same).
    def __eq__(self, other):  return _do_binary(BinaryOp.EQ, self, _coerce(other, self._dtype))
    def __lt__(self, other):  return _do_binary(BinaryOp.LT, self, _coerce(other, self._dtype))
    def __gt__(self, other):  return _do_binary(BinaryOp.GT, self, _coerce(other, self._dtype))
    def __le__(self, other):  return _do_binary(BinaryOp.GT, self, _coerce(other, self._dtype)).__invert__()  # noqa: E501

    __hash__ = None  # type: ignore[assignment]

    def __bool__(self) -> bool:
        if self._shape:
            raise RuntimeError(
                "ambiguous truth value of a multi-element tensor — "
                "use .item() on a 0-dim tensor or compute .any() / .all() yourself"
            )
        return bool(self.item())

    def __invert__(self):
        # Bitwise NOT on Bool tensors (== returns Bool; ~ inverts). Implemented
        # as (1 - x) cast back to Bool. Only meaningful for Bool dtype.
        if self._dtype != DType.BOOL:
            raise TypeError("~ is only defined for Bool tensors")
        ones_b = full(self._shape, DType.BOOL, 1)
        return ones_b - self.cast(DType.INT64).cast(DType.BOOL)  # crude but correct enough

    # -------- unary ops as methods --------
    def exp(self):     return _do_unary(UnaryOp.EXP, self)
    def log(self):     return _do_unary(UnaryOp.LOG, self)
    def sqrt(self):    return _do_unary(UnaryOp.SQRT, self)
    def relu(self):    return _do_unary(UnaryOp.RELU, self)
    def sigmoid(self): return _do_unary(UnaryOp.SIGMOID, self)
    def tanh(self):    return _do_unary(UnaryOp.TANH, self)

    # -------- reductions --------
    def sum(self, axes=None, keepdim: bool = False):
        return _do_reduce(ReduceOp.SUM, self, axes, keepdim)

    def mean(self, axes=None, keepdim: bool = False):
        return _do_reduce(ReduceOp.MEAN, self, axes, keepdim)

    def max(self, axes=None, keepdim: bool = False):
        return _do_reduce(ReduceOp.MAX, self, axes, keepdim)

    def argmax(self, axes=None, keepdim: bool = False):
        return _do_reduce(ReduceOp.ARGMAX, self, axes, keepdim)

    # -------- views --------
    def reshape(self, *shape):
        shape = _normalize_shape(shape)
        # Resolve a single -1 by inferring it from numel — PyTorch convention.
        if -1 in shape:
            if shape.count(-1) > 1:
                raise ValueError("only one dimension can be -1")
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            if known == 0 or self.numel() % known != 0:
                raise ValueError(
                    f"cannot reshape {self.numel()} elements to shape with -1: {shape}"
                )
            inferred = self.numel() // known
            shape = tuple(inferred if s == -1 else s for s in shape)

        return _do_reshape(self, shape)

    def transpose(self, dim_a: int, dim_b: int):
        return _do_transpose(self, dim_a, dim_b)

    @property
    def T(self):
        """Transpose the last two dimensions (PyTorch convention for 2-D+)."""
        if self.ndim < 2:
            return self
        return self.transpose(-1, -2)

    def squeeze(self, dim: int):
        # Implemented via reshape — sufficient for v1 since FFI does not expose
        # tensor_squeeze. For non-contiguous tensors reshape will materialize
        # a copy, which is a minor cost but semantically equivalent.
        if dim < 0:
            dim += self.ndim
        if dim < 0 or dim >= self.ndim:
            raise IndexError(f"squeeze: dim {dim} out of range for {self.ndim}-D tensor")
        if self._shape[dim] != 1:
            raise ValueError(
                f"squeeze: dim {dim} has size {self._shape[dim]}, not 1"
            )
        new_shape = self._shape[:dim] + self._shape[dim + 1 :]
        return self.reshape(*new_shape)

    def unsqueeze(self, dim: int):
        if dim < 0:
            dim += self.ndim + 1
        if dim < 0 or dim > self.ndim:
            raise IndexError(f"unsqueeze: dim {dim} out of range")
        new_shape = self._shape[:dim] + (1,) + self._shape[dim:]
        return self.reshape(*new_shape)

    def slice(self, dim: int, start: int, stop: int, step: int = 1):
        return _do_slice(self, dim, start, stop, step)

    def contiguous(self):
        return _do_contiguous(self)

    def cast(self, dtype: DType):
        return _do_cast(self, dtype)

    # -------- indexing --------
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)

        result = self
        squeezed = 0  # how many dims have been dropped by int indices so far
        for d_orig, k in enumerate(key):
            d = d_orig - squeezed
            if isinstance(k, int):
                size = result._shape[d]
                idx = k if k >= 0 else k + size
                if idx < 0 or idx >= size:
                    raise IndexError(
                        f"index {k} out of range for dim {d_orig} with size {size}"
                    )
                result = result.slice(d, idx, idx + 1, 1).squeeze(d)
                squeezed += 1
            elif isinstance(k, slice):
                size = result._shape[d]
                start = 0 if k.start is None else k.start
                stop = size if k.stop is None else k.stop
                step = 1 if k.step is None else k.step
                if step <= 0:
                    raise ValueError("slice step must be positive")
                result = result.slice(d, start, stop, step)
            else:
                raise TypeError(f"unsupported index type: {type(k).__name__}")
        return result

    # -------- data egress --------
    def tolist(self):
        n = self.numel()
        if n == 0:
            return _build_nested([], self._shape)

        if self._dtype == DType.FLOAT32:
            buf = (ctypes.c_float * n)()
        elif self._dtype == DType.INT64:
            buf = (ctypes.c_int64 * n)()
        elif self._dtype == DType.BOOL:
            buf = (ctypes.c_uint8 * n)()
        else:
            raise ValueError(f"unsupported dtype {self._dtype}")

        _check(lib.tensor_copy_to_buffer(self._handle, buf, ctypes.sizeof(buf)))

        if self._dtype == DType.BOOL:
            flat = [bool(buf[i]) for i in range(n)]
        else:
            flat = [buf[i] for i in range(n)]
        return _build_nested(flat, self._shape)

    def item(self):
        if self._shape:
            raise RuntimeError(
                f"item() requires a 0-dim tensor, got shape {self._shape}"
            )
        # tolist() returns the scalar directly for 0-dim.
        return self.tolist()

    # -------- in-place (used by optimizers) --------
    def _inplace_axpy(self, src: "Tensor", alpha: float) -> "Tensor":
        """In-place AXPY: ``self += alpha * src``. Mutates self's storage."""
        _check(lib.op_inplace(int(InplaceOp.AXPY), self._handle, src._handle, float(alpha)))
        return self


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


def _i64_array(seq: Sequence[int]):
    return (ctypes.c_int64 * len(seq))(*[int(s) for s in seq])


def _wrap(handle: ctypes.c_void_p) -> Tensor:
    """Wrap a freshly-returned engine handle (refcount 1) in a Python Tensor."""
    if not handle.value:
        raise EngineError("engine returned null handle")
    ndim = lib.tensor_ndim(handle)
    if ndim > 0:
        shape_buf = (ctypes.c_int64 * ndim)()
        lib.tensor_shape(handle, shape_buf)
        shape = tuple(shape_buf[i] for i in range(ndim))
    else:
        shape = ()
    dtype = DType(lib.tensor_dtype(handle))
    return Tensor(handle, dtype, shape)


def _build_nested(flat, shape):
    if not shape:
        return flat[0] if flat else None
    if len(shape) == 1:
        return list(flat)
    chunk = 1
    for s in shape[1:]:
        chunk *= s
    return [
        _build_nested(flat[i * chunk : (i + 1) * chunk], shape[1:])
        for i in range(shape[0])
    ]


def _normalize_shape(shape) -> tuple:
    """Accept either ``f(2, 3)`` or ``f((2, 3))`` / ``f([2, 3])``."""
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(shape)


def _coerce(other, dtype: DType) -> Tensor:
    """Promote a Python scalar / list operand to a Tensor with `dtype`."""
    if isinstance(other, Tensor):
        return other
    if isinstance(other, bool):
        return full((), DType.BOOL, 1 if other else 0)
    if isinstance(other, (int, float)):
        return full((), dtype, other)
    if isinstance(other, (list, tuple)):
        return tensor(other, dtype=dtype)
    raise TypeError(f"cannot coerce {type(other).__name__} to Tensor")


def _binary(op_id: BinaryOp, a: Tensor, b: Tensor) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.op_binary(int(op_id), a._handle, b._handle, ctypes.byref(out)))
    return _wrap(out)


def _unary(op_id: UnaryOp, a: Tensor) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.op_unary(int(op_id), a._handle, ctypes.byref(out)))
    return _wrap(out)


def _reduce(op_id: ReduceOp, a: Tensor, axes, keepdim: bool) -> Tensor:
    if axes is None:
        axes_seq: tuple = ()
    elif isinstance(axes, int):
        axes_seq = (axes,)
    else:
        axes_seq = tuple(axes)
    axes_arr = _i64_array(axes_seq) if axes_seq else (ctypes.c_int64 * 0)()
    out = ctypes.c_void_p()
    _check(
        lib.op_reduce(
            int(op_id), a._handle, axes_arr, len(axes_seq),
            1 if keepdim else 0, ctypes.byref(out),
        )
    )
    return _wrap(out)


def _matmul(a: Tensor, b: Tensor) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.op_matmul(a._handle, b._handle, ctypes.byref(out)))
    return _wrap(out)


# Raw view helpers. These bypass autograd recording — the autograd Function
# classes call these directly so a forward() doesn't recurse through the very
# dispatcher that called it.
def _raw_reshape(t: Tensor, shape) -> Tensor:
    shape_arr = _i64_array(shape)
    out = ctypes.c_void_p()
    _check(lib.tensor_reshape(t._handle, shape_arr, len(shape), ctypes.byref(out)))
    return _wrap(out)


def _raw_transpose(t: Tensor, dim_a: int, dim_b: int) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.tensor_transpose(t._handle, dim_a, dim_b, ctypes.byref(out)))
    return _wrap(out)


def _raw_slice(t: Tensor, dim: int, start: int, stop: int, step: int) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.tensor_slice(t._handle, dim, start, stop, step, ctypes.byref(out)))
    return _wrap(out)


def _raw_contiguous(t: Tensor) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.tensor_contiguous(t._handle, ctypes.byref(out)))
    return _wrap(out)


def _raw_cast(t: Tensor, dtype: DType) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.tensor_cast(t._handle, int(dtype), ctypes.byref(out)))
    return _wrap(out)


def _raw_gather(a: Tensor, dim: int, indices: Tensor) -> Tensor:
    out = ctypes.c_void_p()
    _check(lib.op_gather(a._handle, dim, indices._handle, ctypes.byref(out)))
    return _wrap(out)


# Dispatcher hooks. The autograd module overrides these at import time to
# wrap each op in Function.apply. When autograd has not been loaded, the
# defaults call straight into the raw helpers — useful in tests that only
# exercise the forward path.
def _do_binary(op_id, a, b):       return _binary(op_id, a, b)
def _do_unary(op_id, a):           return _unary(op_id, a)
def _do_reduce(op_id, a, axes, keepdim):  return _reduce(op_id, a, axes, keepdim)
def _do_matmul(a, b):              return _matmul(a, b)
def _do_reshape(t, shape):         return _raw_reshape(t, shape)
def _do_transpose(t, dim_a, dim_b): return _raw_transpose(t, dim_a, dim_b)
def _do_slice(t, dim, start, stop, step):  return _raw_slice(t, dim, start, stop, step)
def _do_contiguous(t):             return _raw_contiguous(t)
def _do_cast(t, dtype):            return _raw_cast(t, dtype)
def _do_gather(a, dim, indices):   return _raw_gather(a, dim, indices)


# --------------------------------------------------------------------------- #
# Module-level constructors                                                   #
# --------------------------------------------------------------------------- #


def empty(*shape, dtype: DType = DType.FLOAT32) -> Tensor:
    shape = _normalize_shape(shape)
    shape_arr = _i64_array(shape)
    out = ctypes.c_void_p()
    _check(lib.tensor_empty(shape_arr, len(shape), int(dtype), ctypes.byref(out)))
    return _wrap(out)


def full(shape: ShapeLike, dtype: DType, value: float) -> Tensor:
    shape_t = (shape,) if isinstance(shape, int) else tuple(shape)
    shape_arr = _i64_array(shape_t)
    out = ctypes.c_void_p()
    _check(
        lib.tensor_full(
            shape_arr, len(shape_t), int(dtype), float(value), ctypes.byref(out)
        )
    )
    return _wrap(out)


def zeros(*shape, dtype: DType = DType.FLOAT32) -> Tensor:
    return full(_normalize_shape(shape), dtype, 0.0)


def ones(*shape, dtype: DType = DType.FLOAT32) -> Tensor:
    return full(_normalize_shape(shape), dtype, 1.0)


def zeros_like(t: Tensor) -> Tensor:
    return full(t._shape, t._dtype, 0.0)


def ones_like(t: Tensor) -> Tensor:
    return full(t._shape, t._dtype, 1.0)


def arange(start, stop=None, step=1.0, dtype: DType = DType.FLOAT32) -> Tensor:
    if stop is None:
        start, stop = 0.0, start
    out = ctypes.c_void_p()
    _check(
        lib.tensor_arange(
            float(start), float(stop), float(step), int(dtype), ctypes.byref(out)
        )
    )
    return _wrap(out)


def uniform(lo: float, hi: float, *shape, dtype: DType = DType.FLOAT32) -> Tensor:
    shape = _normalize_shape(shape)
    shape_arr = _i64_array(shape)
    out = ctypes.c_void_p()
    _check(
        lib.tensor_random(
            int(RandomOp.UNIFORM), shape_arr, len(shape), int(dtype),
            float(lo), float(hi), ctypes.byref(out),
        )
    )
    return _wrap(out)


def randn(*shape, dtype: DType = DType.FLOAT32) -> Tensor:
    shape = _normalize_shape(shape)
    shape_arr = _i64_array(shape)
    out = ctypes.c_void_p()
    _check(
        lib.tensor_random(
            int(RandomOp.NORMAL), shape_arr, len(shape), int(dtype),
            0.0, 1.0, ctypes.byref(out),
        )
    )
    return _wrap(out)


def manual_seed(seed: int) -> None:
    lib.tensor_seed(int(seed) & ((1 << 64) - 1))


def tensor(data, dtype: Optional[DType] = None) -> Tensor:
    """Construct a Tensor from a Python scalar, list, or nested list.

    dtype is inferred when not given: if any value in the structure is a
    Python ``float``, the result is Float32; if every value is a bare ``int``
    or ``bool``, it's Int64 or Bool respectively. Pass ``dtype=`` to override.
    """
    shape, flat, inferred = _infer(data)
    if dtype is None:
        dtype = inferred

    if not shape:
        return full((), dtype, flat[0] if flat else 0)

    if dtype == DType.FLOAT32:
        arr = (ctypes.c_float * len(flat))(*[float(v) for v in flat])
    elif dtype == DType.INT64:
        arr = (ctypes.c_int64 * len(flat))(*[int(v) for v in flat])
    elif dtype == DType.BOOL:
        arr = (ctypes.c_uint8 * len(flat))(*[1 if v else 0 for v in flat])
    else:
        raise ValueError(f"unsupported dtype {dtype}")

    shape_arr = _i64_array(shape)
    out = ctypes.c_void_p()
    _check(
        lib.tensor_from_buffer(
            arr, shape_arr, len(shape), int(dtype), ctypes.byref(out)
        )
    )
    return _wrap(out)


def _infer(data):
    """Determine (shape, flat_values, inferred_dtype) for a Python structure."""
    if isinstance(data, bool):
        return (), [data], DType.BOOL
    if isinstance(data, int):
        return (), [data], DType.INT64
    if isinstance(data, float):
        return (), [data], DType.FLOAT32
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,), [], DType.FLOAT32
        inner_shape = None
        flat: list = []
        promote_float = False
        for item in data:
            sub_shape, sub_flat, sub_dtype = _infer(item)
            if inner_shape is None:
                inner_shape = sub_shape
            elif sub_shape != inner_shape:
                raise ValueError(
                    f"inconsistent nested shape: {sub_shape} vs {inner_shape}"
                )
            if sub_dtype == DType.FLOAT32:
                promote_float = True
            flat.extend(sub_flat)
        shape = (len(data),) + inner_shape
        dtype = DType.FLOAT32 if promote_float else DType.INT64
        return shape, flat, dtype
    raise TypeError(f"cannot create Tensor from {type(data).__name__}")
