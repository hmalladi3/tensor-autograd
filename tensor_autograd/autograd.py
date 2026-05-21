"""Reverse-mode automatic differentiation.

Define-by-run, PyTorch-style. The tape is implicit: every op records a
``_Node`` referencing its inputs (via their own ``_grad_fn`` pointers), and
``backward()`` walks the resulting DAG from a scalar output back to the
leaves.

Each differentiable op gets a ``Function`` subclass that knows how to compute
the op forward and how to push gradients backward through it. Importing this
module installs the autograd-aware dispatchers into ``tensor`` so that
ordinary ``Tensor`` arithmetic builds the tape automatically.
"""

from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from . import _ffi
from . import _tensor as _t
from ._ffi import BinaryOp, DType, ReduceOp, UnaryOp
from ._tensor import (
    Tensor,
    _binary,
    _matmul,
    _raw_cast,
    _raw_contiguous,
    _raw_gather,
    _raw_reshape,
    _raw_slice,
    _raw_transpose,
    _reduce,
    _unary,
    ones_like,
    zeros,
)


# --------------------------------------------------------------------------- #
# Grad-enabled state                                                          #
# --------------------------------------------------------------------------- #

_grad_state = threading.local()


def is_grad_enabled() -> bool:
    """Whether tape recording is active on the current thread."""
    return getattr(_grad_state, "enabled", True)


class no_grad:
    """Context manager that disables autograd recording on the current thread.

    Inside ``no_grad()``, ``Function.apply`` still runs the forward op but
    does not create a graph node. Used for inference, optimizer steps, and
    inside ``backward()`` itself so that gradient computation does not in turn
    record a graph.
    """

    def __enter__(self):
        self._prev = is_grad_enabled()
        _grad_state.enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        _grad_state.enabled = self._prev
        return False


# --------------------------------------------------------------------------- #
# Context / graph node                                                        #
# --------------------------------------------------------------------------- #


class Context:
    """Per-call scratch space passed between ``Function.forward`` and
    ``Function.backward``. ``save_for_backward`` keeps strong references to
    tensors needed during the backward pass; other scalars (shapes, axes)
    can be stashed as ordinary attributes.
    """

    def __init__(self):
        self._saved: Tuple[Tensor, ...] = ()

    def save_for_backward(self, *tensors: Tensor) -> None:
        self._saved = tensors

    @property
    def saved_tensors(self) -> Tuple[Tensor, ...]:
        return self._saved


class _Node:
    """A node on the implicit tape: which Function produced an output and
    which Tensor inputs fed into it. The tape is the DAG of these nodes
    reachable from a Tensor's ``_grad_fn``.
    """

    __slots__ = ("fn", "ctx", "inputs")

    def __init__(self, fn, ctx: Context, inputs: List[Tensor]) -> None:
        self.fn = fn
        self.ctx = ctx
        self.inputs = inputs


# --------------------------------------------------------------------------- #
# Function base                                                               #
# --------------------------------------------------------------------------- #


class Function:
    """Base class for differentiable ops. Subclasses override ``forward`` and
    (when differentiable) ``backward``. Override ``differentiable = False``
    for ops whose outputs should never carry gradients (comparisons, argmax).
    """

    differentiable: bool = True

    @classmethod
    def apply(cls, *args):
        ctx = Context()
        out = cls.forward(ctx, *args)

        tensor_inputs = [a for a in args if isinstance(a, Tensor)]
        if (
            cls.differentiable
            and is_grad_enabled()
            and any(t.requires_grad for t in tensor_inputs)
        ):
            out._grad_fn = _Node(cls, ctx, tensor_inputs)
            out.requires_grad = True
        return out


# --------------------------------------------------------------------------- #
# Backward helpers                                                            #
# --------------------------------------------------------------------------- #


def _unbroadcast(grad: Tensor, target_shape) -> Tensor:
    """Collapse a gradient to match the shape of its original (pre-broadcast)
    input. Sums out leading extra dims, then sums size-1-against-N dims with
    keepdim=True.
    """
    target = tuple(target_shape)
    extra = grad.ndim - len(target)
    for _ in range(extra):
        grad = grad.sum(axes=0)
    for i, (g_size, t_size) in enumerate(zip(grad.shape, target)):
        if t_size == 1 and g_size != 1:
            grad = grad.sum(axes=i, keepdim=True)
    return grad


def _expand_to_input(grad: Tensor, input_shape, axes, keepdim: bool) -> Tensor:
    """Broadcast a reduction's gradient back to the input shape.

    If ``keepdim=False``, the reduced dims are inserted as size 1 first via a
    reshape; then broadcasting expands them.
    """
    input_shape = tuple(input_shape)
    if axes is None or len(axes) == 0:
        reduced_axes = list(range(len(input_shape)))
    else:
        reduced_axes = sorted({a % len(input_shape) for a in axes})

    if not keepdim:
        intermediate = list(grad.shape)
        for ax in reduced_axes:
            intermediate.insert(ax, 1)
        grad = grad.reshape(*intermediate)

    # Broadcast via addition to zeros of the input shape. Cheap and avoids
    # adding a tensor_broadcast_to FFI symbol.
    return grad + zeros(*input_shape, dtype=grad.dtype)


# --------------------------------------------------------------------------- #
# Elementwise binary                                                          #
# --------------------------------------------------------------------------- #


class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return _binary(BinaryOp.ADD, a, b)

    @staticmethod
    def backward(ctx, grad_out):
        return (
            _unbroadcast(grad_out, ctx.a_shape),
            _unbroadcast(grad_out, ctx.b_shape),
        )


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.a_shape, ctx.b_shape = a.shape, b.shape
        return _binary(BinaryOp.SUB, a, b)

    @staticmethod
    def backward(ctx, grad_out):
        return (
            _unbroadcast(grad_out, ctx.a_shape),
            _unbroadcast(-grad_out, ctx.b_shape),
        )


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _binary(BinaryOp.MUL, a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        return (
            _unbroadcast(grad_out * b, a.shape),
            _unbroadcast(grad_out * a, b.shape),
        )


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _binary(BinaryOp.DIV, a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        ga = grad_out / b
        gb = -grad_out * a / (b * b)
        return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)


class Pow(Function):
    @staticmethod
    def forward(ctx, a, b):
        out = _binary(BinaryOp.POW, a, b)
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        a, b, out = ctx.saved_tensors
        ga = grad_out * b * (a ** (b - 1.0))
        gb = grad_out * out * a.log()
        return _unbroadcast(ga, a.shape), _unbroadcast(gb, b.shape)


class Maximum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _binary(BinaryOp.MAX, a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        # When equal, route to a (PyTorch convention).
        a_mask = ((a > b) + (a == b)).cast(DType.FLOAT32)
        b_mask = (b > a).cast(DType.FLOAT32)
        return _unbroadcast(grad_out * a_mask, a.shape), _unbroadcast(grad_out * b_mask, b.shape)


class Minimum(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _binary(BinaryOp.MIN, a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        a_mask = ((a < b) + (a == b)).cast(DType.FLOAT32)
        b_mask = (b < a).cast(DType.FLOAT32)
        return _unbroadcast(grad_out * a_mask, a.shape), _unbroadcast(grad_out * b_mask, b.shape)


# Non-differentiable comparisons.
class _NonDiffBinary(Function):
    differentiable = False
    op_id: BinaryOp = BinaryOp.EQ

    @classmethod
    def forward(cls, ctx, a, b):
        return _binary(cls.op_id, a, b)


class Eq(_NonDiffBinary): op_id = BinaryOp.EQ
class Lt(_NonDiffBinary): op_id = BinaryOp.LT
class Gt(_NonDiffBinary): op_id = BinaryOp.GT


_BINARY_FUNCS = {
    BinaryOp.ADD: Add, BinaryOp.SUB: Sub, BinaryOp.MUL: Mul, BinaryOp.DIV: Div,
    BinaryOp.POW: Pow, BinaryOp.MAX: Maximum, BinaryOp.MIN: Minimum,
    BinaryOp.EQ: Eq, BinaryOp.LT: Lt, BinaryOp.GT: Gt,
}


# --------------------------------------------------------------------------- #
# Elementwise unary                                                           #
# --------------------------------------------------------------------------- #


class Neg(Function):
    @staticmethod
    def forward(ctx, a): return _unary(UnaryOp.NEG, a)
    @staticmethod
    def backward(ctx, grad_out): return (-grad_out,)


class Exp(Function):
    @staticmethod
    def forward(ctx, a):
        out = _unary(UnaryOp.EXP, a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        return (grad_out * out,)


class Log(Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return _unary(UnaryOp.LOG, a)

    @staticmethod
    def backward(ctx, grad_out):
        (a,) = ctx.saved_tensors
        return (grad_out / a,)


class Sqrt(Function):
    @staticmethod
    def forward(ctx, a):
        out = _unary(UnaryOp.SQRT, a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        # d/dx sqrt(x) = 1 / (2 sqrt(x)).
        return (grad_out / (out + out),)


class Relu(Function):
    @staticmethod
    def forward(ctx, a):
        out = _unary(UnaryOp.RELU, a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        # Gradient is 1 where the activation was strictly positive, 0 otherwise
        # (subgradient at 0 picked as 0, matching PyTorch).
        mask = (out > 0.0).cast(DType.FLOAT32)
        return (grad_out * mask,)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, a):
        out = _unary(UnaryOp.SIGMOID, a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        return (grad_out * out * (1.0 - out),)


class Tanh(Function):
    @staticmethod
    def forward(ctx, a):
        out = _unary(UnaryOp.TANH, a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (out,) = ctx.saved_tensors
        return (grad_out * (1.0 - out * out),)


_UNARY_FUNCS = {
    UnaryOp.NEG: Neg, UnaryOp.EXP: Exp, UnaryOp.LOG: Log, UnaryOp.SQRT: Sqrt,
    UnaryOp.RELU: Relu, UnaryOp.SIGMOID: Sigmoid, UnaryOp.TANH: Tanh,
}


# --------------------------------------------------------------------------- #
# Reductions                                                                  #
# --------------------------------------------------------------------------- #


class Sum(Function):
    @staticmethod
    def forward(ctx, a, axes, keepdim):
        ctx.input_shape = a.shape
        ctx.axes = axes
        ctx.keepdim = keepdim
        return _reduce(ReduceOp.SUM, a, axes, keepdim)

    @staticmethod
    def backward(ctx, grad_out):
        return (_expand_to_input(grad_out, ctx.input_shape, ctx.axes, ctx.keepdim),)


class Mean(Function):
    @staticmethod
    def forward(ctx, a, axes, keepdim):
        ctx.input_shape = a.shape
        ctx.axes = axes
        ctx.keepdim = keepdim
        return _reduce(ReduceOp.MEAN, a, axes, keepdim)

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.axes is None or len(ctx.axes) == 0:
            reduced = range(len(ctx.input_shape))
        else:
            reduced = (a % len(ctx.input_shape) for a in ctx.axes)
        n = 1
        for ax in reduced:
            n *= ctx.input_shape[ax]
        return (_expand_to_input(grad_out / float(n), ctx.input_shape, ctx.axes, ctx.keepdim),)


class MaxReduce(Function):
    @staticmethod
    def forward(ctx, a, axes, keepdim):
        ctx.save_for_backward(a)
        ctx.axes = axes
        ctx.keepdim = keepdim
        ctx.input_shape = a.shape
        # Compute with keepdim=True so the saved max broadcasts back against a.
        # If the caller wanted keepdim=False we reshape (= squeeze) below.
        out_kept = _reduce(ReduceOp.MAX, a, axes, True)
        ctx.max_kept = out_kept
        if keepdim:
            return out_kept
        if axes is None or len(axes) == 0:
            new_shape = ()
        else:
            reduced = {ax % len(a.shape) for ax in axes}
            new_shape = tuple(s for i, s in enumerate(a.shape) if i not in reduced)
        return _raw_reshape(out_kept, new_shape)

    @staticmethod
    def backward(ctx, grad_out):
        (a,) = ctx.saved_tensors
        mask = (a == ctx.max_kept).cast(DType.FLOAT32)
        expanded = _expand_to_input(grad_out, ctx.input_shape, ctx.axes, ctx.keepdim)
        return (expanded * mask,)


class Argmax(Function):
    differentiable = False

    @staticmethod
    def forward(ctx, a, axes, keepdim):
        return _reduce(ReduceOp.ARGMAX, a, axes, keepdim)


_REDUCE_FUNCS = {
    ReduceOp.SUM: Sum, ReduceOp.MEAN: Mean,
    ReduceOp.MAX: MaxReduce, ReduceOp.ARGMAX: Argmax,
}


# --------------------------------------------------------------------------- #
# Matmul                                                                      #
# --------------------------------------------------------------------------- #


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return _matmul(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        # For batched inputs, .transpose(-1, -2) flips just the last two dims,
        # which is what matmul backward needs.
        if a.ndim < 2 or b.ndim < 2:
            raise RuntimeError("matmul backward expects ≥2-D inputs")
        ga = grad_out @ b.transpose(-1, -2)
        gb = a.transpose(-1, -2) @ grad_out
        return ga, gb


# --------------------------------------------------------------------------- #
# View / movement ops                                                         #
# --------------------------------------------------------------------------- #


class Reshape(Function):
    @staticmethod
    def forward(ctx, a, shape):
        ctx.input_shape = a.shape
        return _raw_reshape(a, shape)

    @staticmethod
    def backward(ctx, grad_out):
        return (grad_out.reshape(*ctx.input_shape),)


class Transpose(Function):
    @staticmethod
    def forward(ctx, a, dim_a, dim_b):
        ctx.dim_a = dim_a
        ctx.dim_b = dim_b
        return _raw_transpose(a, dim_a, dim_b)

    @staticmethod
    def backward(ctx, grad_out):
        return (grad_out.transpose(ctx.dim_a, ctx.dim_b),)


class Contiguous(Function):
    @staticmethod
    def forward(ctx, a):
        return _raw_contiguous(a)

    @staticmethod
    def backward(ctx, grad_out):
        return (grad_out,)


# Slice, Cast, Gather are non-differentiable in v1 (the engine has no scatter
# primitive yet, and the MNIST training loop does not require gradients
# flowing back through any of them). Outputs lose requires_grad regardless of
# input, matching the spec for non-differentiable ops.

class Cast(Function):
    differentiable = False

    @staticmethod
    def forward(ctx, a, dtype):
        return _raw_cast(a, dtype)


class Slice(Function):
    differentiable = False

    @staticmethod
    def forward(ctx, a, dim, start, stop, step):
        return _raw_slice(a, dim, start, stop, step)


class Gather(Function):
    differentiable = False

    @staticmethod
    def forward(ctx, a, dim, indices):
        return _raw_gather(a, dim, indices)


# --------------------------------------------------------------------------- #
# Dispatcher installation                                                     #
# --------------------------------------------------------------------------- #
# Replace tensor.py's forward-only dispatchers with versions that route through
# Function.apply, so the implicit tape gets built whenever a user does ordinary
# ``Tensor`` arithmetic.


def _do_binary_autograd(op_id, a, b):
    return _BINARY_FUNCS[op_id].apply(a, b)


def _do_unary_autograd(op_id, a):
    return _UNARY_FUNCS[op_id].apply(a)


def _do_reduce_autograd(op_id, a, axes, keepdim):
    # Normalize axes once so every Function sees the same shape.
    if axes is None:
        axes = ()
    elif isinstance(axes, int):
        axes = (axes,)
    else:
        axes = tuple(axes)
    return _REDUCE_FUNCS[op_id].apply(a, axes, keepdim)


def _do_matmul_autograd(a, b):
    return MatMul.apply(a, b)


def _do_reshape_autograd(t, shape):
    return Reshape.apply(t, shape)


def _do_transpose_autograd(t, dim_a, dim_b):
    return Transpose.apply(t, dim_a, dim_b)


def _do_slice_autograd(t, dim, start, stop, step):
    return Slice.apply(t, dim, start, stop, step)


def _do_contiguous_autograd(t):
    return Contiguous.apply(t)


def _do_cast_autograd(t, dtype):
    return Cast.apply(t, dtype)


def _do_gather_autograd(a, dim, indices):
    return Gather.apply(a, dim, indices)


_t._do_binary = _do_binary_autograd
_t._do_unary = _do_unary_autograd
_t._do_reduce = _do_reduce_autograd
_t._do_matmul = _do_matmul_autograd
_t._do_reshape = _do_reshape_autograd
_t._do_transpose = _do_transpose_autograd
_t._do_slice = _do_slice_autograd
_t._do_contiguous = _do_contiguous_autograd
_t._do_cast = _do_cast_autograd
_t._do_gather = _do_gather_autograd


# --------------------------------------------------------------------------- #
# backward()                                                                  #
# --------------------------------------------------------------------------- #


def _topo_sort(root: _Node) -> List[_Node]:
    """Returns the nodes reachable from ``root`` in dependency order
    (dependencies first, ``root`` last). The backward pass iterates this
    in reverse.
    """
    order: List[_Node] = []
    seen: set = set()

    def visit(n: _Node) -> None:
        if id(n) in seen:
            return
        seen.add(id(n))
        for inp in n.inputs:
            if inp._grad_fn is not None:
                visit(inp._grad_fn)
        order.append(n)

    visit(root)
    return order


def _backward_impl(tensor_out: Tensor, grad: Optional[Tensor]) -> None:
    if not tensor_out.requires_grad or tensor_out._grad_fn is None:
        return

    if grad is None:
        if tensor_out.numel() != 1:
            raise RuntimeError(
                "backward(): non-scalar tensor requires an explicit grad argument"
            )
        grad = ones_like(tensor_out)

    order = _topo_sort(tensor_out._grad_fn)
    pending = {id(tensor_out._grad_fn): grad}

    with no_grad():
        for node in reversed(order):
            out_grad = pending.pop(id(node), None)
            if out_grad is None:
                continue
            in_grads = node.fn.backward(node.ctx, out_grad)
            if not isinstance(in_grads, tuple):
                in_grads = (in_grads,)
            if len(in_grads) != len(node.inputs):
                raise RuntimeError(
                    f"{node.fn.__name__}.backward returned {len(in_grads)} grads "
                    f"for {len(node.inputs)} inputs"
                )
            for inp, ig in zip(node.inputs, in_grads):
                if ig is None:
                    continue
                if inp._grad_fn is None:
                    # Leaf: accumulate into .grad if the leaf wants gradients.
                    if inp.requires_grad:
                        inp._grad = ig if inp._grad is None else inp._grad + ig
                else:
                    key = id(inp._grad_fn)
                    pending[key] = ig if key not in pending else pending[key] + ig


def _tensor_backward(self: Tensor, grad: Optional[Tensor] = None) -> None:
    """``loss.backward()`` — walk the tape from this tensor and accumulate
    gradients into every leaf with ``requires_grad=True``.
    """
    _backward_impl(self, grad)


# Patch Tensor.backward in place. Tensor itself doesn't import autograd, so
# importing this module is what wires it up.
Tensor.backward = _tensor_backward  # type: ignore[attr-defined]
