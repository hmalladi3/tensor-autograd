"""Stateless neural-network ops.

These are plain functions (no Module wrapping). Activations are exposed both
here and as ``Module`` subclasses in ``nn.layers``; loss functions are only
here because they hold no state.
"""

from __future__ import annotations

from .._ffi import DType
from .._tensor import Tensor, arange


# ---- Activations ----


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


# ---- Losses ----


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error — mean of ``(pred - target) ** 2`` over all elements."""
    diff = pred - target
    return (diff * diff).mean()


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross-entropy loss with integer class targets.

    Args:
      logits:  ``(N, C)`` Float32.
      targets: ``(N,)``   Int64 class indices in ``[0, C)``.

    Implementation note: we build the one-hot matrix from a comparison
    instead of using ``gather``, because ``gather`` is non-differentiable in
    v1 and we need the gradient w.r.t. the logits to flow through. The
    forward cost is one extra ``(N, C)`` allocation per call — fine at MNIST
    scale, would matter for very wide C.
    """
    if logits.ndim != 2:
        raise ValueError(f"cross_entropy: logits must be 2-D, got shape {logits.shape}")
    if targets.ndim != 1:
        raise ValueError(f"cross_entropy: targets must be 1-D, got shape {targets.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"cross_entropy: batch sizes disagree: logits {logits.shape}, "
            f"targets {targets.shape}"
        )

    N, C = logits.shape

    # log_softmax via log-sum-exp for numerical stability.
    max_logits = logits.max(axes=1, keepdim=True)
    shifted = logits - max_logits
    log_sum_exp = shifted.exp().sum(axes=1, keepdim=True).log()
    log_softmax = shifted - log_sum_exp

    # One-hot via comparison — gather is non-differentiable in v1.
    # Cast to Float32 before comparing because the engine's binary ops are
    # dispatch_floating in v1; the cast doesn't need a gradient because
    # neither operand contributes one.
    arange_C = arange(C, dtype=DType.FLOAT32).reshape(1, C)
    targets_col = targets.cast(DType.FLOAT32).reshape(N, 1)
    one_hot = (arange_C == targets_col).cast(DType.FLOAT32)

    return -(one_hot * log_softmax).sum(axes=1).mean()
