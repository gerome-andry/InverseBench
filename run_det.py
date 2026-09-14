r"""Deterministic launcher for InverseBench.

WHY THIS EXISTS. The Navier-Stokes pipeline is NOT reproducible by default. Two runs of an
identical configuration -- same seed, same instance, verified identical observation -- produce
final ensembles differing by 85% in relative norm. A minimal two-sweep run already diverges by
1e-5 and the chaotic simulator amplifies it. `torch.manual_seed` does not control this: the
denoiser, the ODE draw and `torch.randn` are each deterministic on fixed input, so the source
is non-deterministic GPU reduction (most plausibly autograd's atomic accumulation, since V_t is
built from backward passes).

MEASURED NOISE FLOOR, five runs of one configuration:

    err    mean 0.5808   sd 0.0406   range 0.1036
    |log|  mean 0.1362   sd 0.1023   range 0.2450

so runs per arm to resolve an effect E at ~2 sem, unpaired: E=0.02 -> n~33, E=0.05 -> n~5,
E=0.10 -> n~1. Much of this project's tuning evidence sits in the 0.02-0.05 band and was
collected at n=3, which resolves ~0.065 in err and essentially nothing in calibration.

WHAT THIS BUYS. Bitwise reproducibility (verified rel||d|| = 0.000e+00), at ~no cost -- a full
run measured 492 s. The seed then becomes a proper blocking variable: two configurations at the
same (sample_id, seed) consume identical random draws, so their paired difference isolates the
configuration change instead of measuring it on top of 0.041-0.102 of chaotic noise.

It has to be a LAUNCHER rather than a flag inside the script, because CUBLAS_WORKSPACE_CONFIG
must be set before cuBLAS initialises.

NOTE. Deterministic mode follows a different (but fixed) trajectory than the default kernels,
so baselines must be re-run under it -- old numbers cannot be mixed with new ones.

    python run_det.py problem=navier-stokes algorithm=kpsh <overrides>
"""

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

torch.use_deterministic_algorithms(True, warn_only=False)

import runpy  # noqa: E402

sys.argv = ["main.py"] + sys.argv[1:]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
               run_name="__main__")
