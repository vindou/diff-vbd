"""Test-wide JAX configuration.

x64 is enabled for the whole suite because contact needs it. The IPC barrier resolves a gap
that is orders of magnitude smaller than the mesh coordinates, and it gets that gap by
*subtracting* those coordinates: in float32 a small positive gap can round to zero or below,
at which point ``log(gap / d_hat)`` is NaN and intersection-freedom is lost to rounding
before any solver logic runs.

This does not change the existing solver tests, which pin ``dtype=jnp.float32`` explicitly
on every array they build. Enabling x64 only widens what dtypes are *available*; it does not
promote an array that already declared itself float32.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
