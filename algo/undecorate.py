# algo/undecorate.py

import torch


def differentiable(forward_op):
    r"""The operator's `__call__` with every `@torch.no_grad()` on the path bypassed.

    WHY THIS IS NEEDED. `inverse_problems/navier_stokes.py` decorates BOTH `__call__` and
    `forward` with `@torch.no_grad()`. A decorator runs its body under no_grad whatever the
    caller does, so an outer `enable_grad` cannot undo it, and the three differentiation paths
    then disagree:

        torch.autograd.grad   raises
        torch.func.vjp        SILENTLY RETURNS ZEROS      <- the trap
        torch.func.jvp        works anyway, because forward-mode duals ignore no_grad

    So the operator looks differentiable -- |jvp| = 184 on Navier-Stokes -- while |vjp| = 0.
    Anything building A^T from a vjp gets a zero adjoint and no error. In the KPS G update that
    makes `step = V(At(sol))` identically zero: the method does not degrade, it does not run.

    BOTH decorators have to go. Undecorating only `__call__` still enters the decorated
    `forward` inside it, which then raises instead.

    The recovered adjoint is real: on Navier-Stokes <u, Jv> = 425.869 against <J^T u, v> =
    425.868, a relative difference of 2.2e-6.

    Navier-Stokes is the only affected operator -- inv-scatter and blackhole carry no decorator
    and their gradients were always fine.

    NOTE ON MEMORY: with the decorator gone the operator records a graph whenever the caller
    allows it, so callers that do not want one must say `torch.no_grad()` themselves. The KPS
    H update and `inference` already do.
    """

    cls = type(forward_op)
    patched = {}

    for name in ("__call__", "forward"):
        fn = getattr(cls, name, None)
        inner = getattr(fn, "__wrapped__", None)
        if inner is not None:
            patched[name] = inner

    if not patched:
        return lambda data: forward_op(data)

    call = patched.get("__call__")
    fwd = patched.get("forward")

    if call is None:
        return lambda data: forward_op(data)

    if fwd is None:
        return lambda data: call(forward_op, data)

    # `__call__` reaches `forward` through the instance, so shadow it there for the duration
    def op(data):
        bound = fwd.__get__(forward_op, cls)
        object.__setattr__(forward_op, "forward", bound)
        try:
            return call(forward_op, data)
        finally:
            try:
                object.__delattr__(forward_op, "forward")
            except AttributeError:
                pass

    return op
