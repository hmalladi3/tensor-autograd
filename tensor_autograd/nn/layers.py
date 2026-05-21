"""Layer modules — ``Linear``, ``Sequential``, and activation wrappers."""

from __future__ import annotations

from typing import List

from .._tensor import Tensor
from . import functional as F
from . import init
from .module import Module, Parameter


class Linear(Module):
    """Affine transform: ``y = x @ W.T + b``.

    Weight is ``(out_features, in_features)`` and initialised by Kaiming
    uniform (good default for ReLU). Bias, when present, starts at zero.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            init.kaiming_uniform(in_features, (out_features, in_features))
        )
        if bias:
            self.bias = Parameter(init.zeros((out_features,)))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight.transpose(-1, -2)
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        bias = "True" if self.bias is not None else "False"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={bias})"


class Sequential(Module):
    """Compose modules in order; ``forward(x)`` threads ``x`` through each."""

    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self._layer_names: List[str] = []
        for i, layer in enumerate(layers):
            name = f"_layer_{i}"
            setattr(self, name, layer)
            self._layer_names.append(name)

    def forward(self, x: Tensor) -> Tensor:
        for name in self._layer_names:
            x = getattr(self, name)(x)
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor: return F.relu(x)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor: return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor: return F.tanh(x)
