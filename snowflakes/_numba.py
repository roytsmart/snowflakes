"""
Steps i to iv of the model, compiled with Numba.

This computes exactly what :func:`snowflakes._numpy.step` does, bit for bit,
in three passes over the grid, each parallel over its rows: diffusion and
freezing, then attachment, then melting. Each pass writes only to the cell it
visits and reads its neighbors only from the pass before.
"""

import numpy as np
import numba

from ._numpy import wrap as _wrap_python

__all__ = [
    "step",
]

_wrap = numba.njit(_wrap_python)


@numba.njit(parallel=True)
def step(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    count: np.ndarray,
    alpha: float,
    beta: float,
    theta: float,
    kappa: float,
    one_minus_kappa: float,
    mu: float,
    one_minus_mu: float,
    gamma: float,
    one_minus_gamma: float,
    weights: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Diffusion, freezing, attachment, and melting: steps i to iv of
    Gravner and Griffeath (2008).

    The parameters and return values are those of
    :func:`snowflakes._numpy.step`.
    """
    num_y = a.shape[0] - 2
    num_x = a.shape[1] - 2

    _wrap(a)
    _wrap(d)

    # i. Diffusion, and ii. freezing on the boundary.
    for i in numba.prange(1, num_y + 1):
        for j in range(1, num_x + 1):
            k = (
                a[i - 1, j]
                + a[i - 1, j + 1]
                + a[i, j - 1]
                + a[i, j + 1]
                + a[i + 1, j - 1]
                + a[i + 1, j]
            )
            count[i, j] = k
            if a[i, j]:
                e[i, j] = d[i, j]
                continue
            v = d[i, j] * weights[k]
            v += d[i - 1, j]
            v += d[i - 1, j + 1]
            v += d[i, j - 1]
            v += d[i, j + 1]
            v += d[i + 1, j - 1]
            v += d[i + 1, j]
            v /= weights[6]
            if k > 0:
                b[i, j] += one_minus_kappa * v
                c[i, j] += kappa * v
                e[i, j] = 0
            else:
                e[i, j] = v

    _wrap(e)

    # iii. Attachment, decided from the attached neighbors before it.
    for i in numba.prange(1, num_y + 1):
        for j in range(1, num_x + 1):
            k = count[i, j]
            if a[i, j] or k == 0:
                continue
            if k <= 2:
                attach = b[i, j] >= beta
            elif k == 3:
                if b[i, j] >= 1:
                    attach = True
                else:
                    vapor = e[i, j]
                    vapor += e[i - 1, j]
                    vapor += e[i - 1, j + 1]
                    vapor += e[i, j - 1]
                    vapor += e[i, j + 1]
                    vapor += e[i + 1, j - 1]
                    vapor += e[i + 1, j]
                    attach = vapor < theta and b[i, j] >= alpha
            else:
                attach = True
            if attach:
                a[i, j] = 1
                c[i, j] += b[i, j]
                b[i, j] = 0

    _wrap(a)

    # iv. Melting on the new boundary.
    for i in numba.prange(1, num_y + 1):
        for j in range(1, num_x + 1):
            if a[i, j]:
                continue
            if (
                a[i - 1, j]
                | a[i - 1, j + 1]
                | a[i, j - 1]
                | a[i, j + 1]
                | a[i + 1, j - 1]
                | a[i + 1, j]
            ):
                e[i, j] += mu * b[i, j] + gamma * c[i, j]
                b[i, j] *= one_minus_mu
                c[i, j] *= one_minus_gamma

    return e, d
