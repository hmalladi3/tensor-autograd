"""Parameter initialization helpers.

Each returns a fresh ``Tensor`` of the requested shape and dtype. To use as
a Parameter, wrap with ``nn.Parameter(...)``. v1 ships construction-style
helpers only — no in-place reinit, since the MNIST MLP doesn't need to
overwrite an existing parameter's storage.
"""

from __future__ import annotations

import math
from typing import Sequence, Union

from .._ffi import DType
from .._tensor import Tensor, full, randn, uniform

ShapeLike = Union[int, Sequence[int]]


def _as_tuple(shape: ShapeLike) -> tuple:
    return (shape,) if isinstance(shape, int) else tuple(shape)


def zeros(shape: ShapeLike, dtype: DType = DType.FLOAT32) -> Tensor:
    return full(_as_tuple(shape), dtype, 0.0)


def ones(shape: ShapeLike, dtype: DType = DType.FLOAT32) -> Tensor:
    return full(_as_tuple(shape), dtype, 1.0)


def uniform_(
    lo: float, hi: float, shape: ShapeLike, dtype: DType = DType.FLOAT32
) -> Tensor:
    """``uniform_(lo, hi, shape)`` — returns a fresh tensor of uniform samples
    in ``[lo, hi)``. The trailing underscore mirrors PyTorch's naming
    convention for parameter initializers (even though this version is
    construction-style, not in-place)."""
    return uniform(lo, hi, *_as_tuple(shape), dtype=dtype)


def normal_(
    mean: float = 0.0,
    std: float = 1.0,
    shape: ShapeLike = (),
    dtype: DType = DType.FLOAT32,
) -> Tensor:
    """Returns a fresh tensor sampled from N(mean, std^2)."""
    shape_t = _as_tuple(shape)
    if not shape_t:
        return Tensor.__new__(Tensor)  # unreachable in practice; placate type checker
    t = randn(*shape_t, dtype=dtype)
    if mean != 0.0 or std != 1.0:
        return t * std + mean
    return t


def kaiming_uniform(
    fan_in: int,
    shape: ShapeLike,
    dtype: DType = DType.FLOAT32,
) -> Tensor:
    """Kaiming-uniform initialization: samples in ``[-bound, bound)`` where
    ``bound = sqrt(6 / fan_in)``. Sensible default for ReLU-activated layers.
    """
    bound = math.sqrt(6.0 / fan_in)
    return uniform(-bound, bound, *_as_tuple(shape), dtype=dtype)


def xavier_uniform(
    fan_in: int,
    fan_out: int,
    shape: ShapeLike,
    dtype: DType = DType.FLOAT32,
) -> Tensor:
    """Xavier-uniform initialization: samples in ``[-bound, bound)`` where
    ``bound = sqrt(6 / (fan_in + fan_out))``. Sensible default for tanh /
    sigmoid layers."""
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    return uniform(-bound, bound, *_as_tuple(shape), dtype=dtype)
