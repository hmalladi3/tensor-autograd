"""Optimizers — ``SGD`` (with momentum) and ``Adam``.

Both update parameters in place via the engine's ``op_inplace(OP_AXPY, ...)``
primitive so the parameter's engine handle is preserved across steps.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from ._tensor import Tensor, zeros_like
from .autograd import no_grad


class Optimizer:
    """Base optimizer: keeps a list of parameters, an lr, and a per-parameter
    state dict keyed by ``id(param)`` for whatever bookkeeping the subclass
    needs (momentum buffers, Adam moments, …).
    """

    def __init__(self, params: Iterable[Tensor], lr: float) -> None:
        # Materialize the iterable into a list — generators are single-pass.
        self.params: list = list(params)
        self.lr = float(lr)
        self.state: dict = {id(p): {} for p in self.params}

    def zero_grad(self) -> None:
        for p in self.params:
            p._grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent, optionally with momentum.

    Update (no momentum):     ``p ← p - lr * g``
    Update (with momentum m): ``buf ← m * buf + g;  p ← p - lr * buf``
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.0,
    ) -> None:
        if momentum < 0:
            raise ValueError(f"SGD: momentum must be non-negative, got {momentum}")
        super().__init__(params, lr)
        self.momentum = float(momentum)

    def step(self) -> None:
        with no_grad():
            for p in self.params:
                if p._grad is None:
                    continue
                g = p._grad

                if self.momentum > 0.0:
                    st = self.state[id(p)]
                    buf = st.get("momentum_buffer")
                    if buf is None:
                        # First step: initialize buffer to the gradient itself.
                        buf = g
                    else:
                        buf = buf * self.momentum + g
                    st["momentum_buffer"] = buf
                    p._inplace_axpy(buf, -self.lr)
                else:
                    p._inplace_axpy(g, -self.lr)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014). Maintains per-parameter first/second moment
    estimates with bias correction.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Adam: betas must be in [0, 1), got {betas}")
        if eps <= 0:
            raise ValueError(f"Adam: eps must be positive, got {eps}")
        super().__init__(params, lr)
        self.b1, self.b2 = float(betas[0]), float(betas[1])
        self.eps = float(eps)

    def step(self) -> None:
        with no_grad():
            for p in self.params:
                if p._grad is None:
                    continue
                st = self.state[id(p)]
                st["step"] = st.get("step", 0) + 1
                t = st["step"]

                m = st.get("m")
                v = st.get("v")
                if m is None:
                    m = zeros_like(p)
                    v = zeros_like(p)

                g = p._grad
                m = m * self.b1 + g * (1.0 - self.b1)
                v = v * self.b2 + (g * g) * (1.0 - self.b2)
                st["m"], st["v"] = m, v

                bc1 = 1.0 - self.b1 ** t
                bc2 = 1.0 - self.b2 ** t
                update = (m / bc1) / ((v / bc2).sqrt() + self.eps)
                p._inplace_axpy(update, -self.lr)
