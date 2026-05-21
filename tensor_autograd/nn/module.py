"""``Module`` — the base class for things that hold parameters — and
``Parameter`` — a ``Tensor`` subclass marked as trainable."""

from __future__ import annotations

from typing import Generator

from .._tensor import Tensor


class Parameter(Tensor):
    """A ``Tensor`` that participates in optimization.

    Created by wrapping an existing Tensor: ownership of the underlying
    engine handle transfers to the new Parameter, and the source Tensor is
    hollowed out so its ``__del__`` won't double-decref. This is the load-
    bearing trick — without it we'd have two Python objects pointing at the
    same engine handle, and the second one to be garbage-collected would
    underflow the refcount.
    """

    def __init__(self, data: Tensor) -> None:
        super().__init__(
            handle=data._handle,
            dtype=data._dtype,
            shape=data._shape,
            requires_grad=True,
        )
        # Hollow out the source: prevent double-decref on GC.
        data._handle = None  # type: ignore[assignment]
        data.requires_grad = False


class Module:
    """Base class for stateful neural-network components.

    Assigning a ``Parameter`` or another ``Module`` as an attribute auto-
    registers it. ``parameters()`` recursively yields every Parameter the
    Module owns (directly or via sub-Modules), deduplicated by identity.
    """

    def __init__(self) -> None:
        # Bypass our own __setattr__ for the bootstrap dicts.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value) -> None:
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")

        # If we're replacing a previously-registered slot with something of a
        # different kind, unregister it first.
        if params is not None and name in params and not isinstance(value, Parameter):
            del params[name]
        if modules is not None and name in modules and not isinstance(value, Module):
            del modules[name]

        if isinstance(value, Parameter):
            if params is None:
                raise RuntimeError(
                    f"cannot assign Parameter '{name}' before calling Module.__init__()"
                )
            params[name] = value
        elif isinstance(value, Module):
            if modules is None:
                raise RuntimeError(
                    f"cannot assign Module '{name}' before calling Module.__init__()"
                )
            modules[name] = value

        object.__setattr__(self, name, value)

    # ---- traversal ----
    def parameters(self) -> Generator[Parameter, None, None]:
        seen: set = set()
        for p in self._parameters.values():
            if id(p) not in seen:
                seen.add(id(p))
                yield p
        for m in self._modules.values():
            for p in m.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    yield p

    def zero_grad(self) -> None:
        for p in self.parameters():
            p._grad = None

    # ---- mode ----
    def train(self) -> "Module":
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self) -> "Module":
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    # ---- call protocol ----
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    # ---- repr ----
    def __repr__(self) -> str:
        body = []
        for name, m in self._modules.items():
            body.append(f"  ({name}): {m!r}")
        if not body:
            return f"{type(self).__name__}()"
        return f"{type(self).__name__}(\n" + "\n".join(body) + "\n)"
