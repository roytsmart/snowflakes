"""
Steps i to iv of the model, compiled with Numba.

This is the fast implementation on a computer with Numba installed.
It computes the same update as :mod:`snowflakes._numpy`, one cell at a time
in parallel.
"""

import numpy as np
import numba

__all__ = [
    "step",
]


def step(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    theta: float,
    kappa: float,
    mu: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Diffusion, freezing, attachment, and melting: steps i to iv of
    Gravner and Griffeath (2008).

    The parameters are those of :func:`snowflakes._numpy.step`.
    """
    a, b, c, d = _diffusion(a_n=a, b_n=b, c_n=c, d_n=d)
    a, b, c, d = _freezing(a_n=a, b_n=b, c_n=c, d_n=d, kappa=kappa)
    a, b, c, d = _attachment(
        a_n=a,
        b_n=b,
        c_n=c,
        d_n=d,
        alpha=alpha,
        beta=beta,
        theta=theta,
    )
    a, b, c, d = _melting(a_n=a, b_n=b, c_n=c, d_n=d, mu=mu, gamma=gamma)
    return a, b, c, d


@numba.njit(parallel=True)
def _diffusion(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in range(num_y):
            a_result[i, j] = a_n[i, j]
            b_result[i, j] = b_n[i, j]
            c_result[i, j] = c_n[i, j]

            if a_n[i, j]:
                d_result[i, j] = d_n[i, j]

            else:
                d_sum = 0.0

                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if m == n == -1:
                            continue
                        if m == n == 1:
                            continue

                        p = (i + m) % num_x
                        q = (j + n) % num_y

                        if a_n[p, q]:
                            d_npq = d_n[i, j]
                        else:
                            d_npq = d_n[p, q]

                        d_sum += d_npq

                d_result[i, j] = d_sum / 7

    return a_result, b_result, c_result, d_result


@numba.njit
def _on_boundary(
    a_n: np.ndarray,
    i: int,
    j: int,
) -> bool:
    num_x, num_y = a_n.shape

    for m in range(-1, 2):
        for n in range(-1, 2):
            if m == n == -1:
                continue
            if m == n == 0:
                continue
            if m == n == 1:
                continue

            if a_n[(i + m) % num_x, (j + n) % num_y]:
                return True

    return False


@numba.njit(parallel=True)
def _freezing(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
    kappa: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in range(num_y):
            a_result[i, j] = a_n[i, j]

            if not a_n[i, j] and _on_boundary(a_n=a_n, i=i, j=j):
                b_result[i, j] = b_n[i, j] + (1 - kappa) * d_n[i, j]
                c_result[i, j] = c_n[i, j] + kappa * d_n[i, j]
                d_result[i, j] = 0
            else:
                b_result[i, j] = b_n[i, j]
                c_result[i, j] = c_n[i, j]
                d_result[i, j] = d_n[i, j]

    return a_result, b_result, c_result, d_result


@numba.njit(parallel=True)
def _attachment(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
    alpha: float,
    beta: float,
    theta: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in range(num_y):
            a_result[i, j] = a_n[i, j]
            b_result[i, j] = b_n[i, j]
            c_result[i, j] = c_n[i, j]
            d_result[i, j] = d_n[i, j]

            if a_n[i, j]:
                continue

            num_attached_neighbors = 0
            diffusive_mass = d_n[i, j]

            for m in range(-1, 2):
                for n in range(-1, 2):
                    if m == n == -1:
                        continue
                    if m == n == 0:
                        continue
                    if m == n == 1:
                        continue

                    p = (i + m) % num_x
                    q = (j + n) % num_y

                    if a_n[p, q]:
                        num_attached_neighbors += 1

                    diffusive_mass += d_n[p, q]

            if num_attached_neighbors == 0:
                attach = False
            elif num_attached_neighbors <= 2:
                attach = b_n[i, j] >= beta
            elif num_attached_neighbors == 3:
                attach = (b_n[i, j] >= 1) or (
                    (diffusive_mass < theta) and (b_n[i, j] >= alpha)
                )
            else:
                attach = True

            if attach:
                a_result[i, j] = True
                b_result[i, j] = 0
                c_result[i, j] = b_n[i, j] + c_n[i, j]

    return a_result, b_result, c_result, d_result


@numba.njit(parallel=True)
def _melting(
    a_n: np.ndarray,
    b_n: np.ndarray,
    c_n: np.ndarray,
    d_n: np.ndarray,
    mu: float,
    gamma: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    num_x, num_y = a_n.shape

    a_result = np.empty_like(a_n)
    b_result = np.empty_like(b_n)
    c_result = np.empty_like(c_n)
    d_result = np.empty_like(d_n)

    for i in numba.prange(num_x):
        for j in range(num_y):
            a_result[i, j] = a_n[i, j]

            if not a_n[i, j] and _on_boundary(a_n=a_n, i=i, j=j):
                b_result[i, j] = (1 - mu) * b_n[i, j]
                c_result[i, j] = (1 - gamma) * c_n[i, j]
                d_result[i, j] = d_n[i, j] + mu * b_n[i, j] + gamma * c_n[i, j]
            else:
                b_result[i, j] = b_n[i, j]
                c_result[i, j] = c_n[i, j]
                d_result[i, j] = d_n[i, j]

    return a_result, b_result, c_result, d_result
