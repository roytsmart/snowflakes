"""
Steps i to iv of the model, vectorized with NumPy.

This is the reference implementation, and the one used wherever Numba is
missing. :mod:`snowflakes._numba` and the Rust kernel of the web app compute
exactly the same thing, bit for bit: the same operations, on the same values,
in the same order.

Every array is padded by one cell on each side, which :func:`wrap` fills from
the opposite edge so that the grid wraps around. The arrays are updated in
place.
"""

import numpy as np

__all__ = [
    "NEIGHBORS",
    "wrap",
    "step",
]

NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
"""
The offsets, in rows and columns, of the six nearest neighbors of a cell, in
the order their values are summed.

The grid is a triangular lattice stored on a square array: each cell
neighbors the cells above, below, left, and right of it, and the two along
the diagonal from lower left to upper right.
"""


def wrap(x: np.ndarray) -> None:
    """
    Fill the padding of `x` from the opposite edges of the grid.

    Parameters
    ----------
    x
        A padded field.
    """
    x[0, 1:-1] = x[-2, 1:-1]
    x[-1, 1:-1] = x[1, 1:-1]
    x[:, 0] = x[:, -2]
    x[:, -1] = x[:, 1]


def _around(x: np.ndarray) -> "list[np.ndarray]":
    """The six neighbors of every cell inside the padding of `x`."""
    num_y = x.shape[0] - 2
    num_x = x.shape[1] - 2
    return [x[1 + i : 1 + i + num_y, 1 + j : 1 + j + num_x] for i, j in NEIGHBORS]


def step(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    e: np.ndarray,
    count: np.ndarray,
    alpha: np.floating,
    beta: np.floating,
    theta: np.floating,
    kappa: np.floating,
    one_minus_kappa: np.floating,
    mu: np.floating,
    one_minus_mu: np.floating,
    gamma: np.floating,
    one_minus_gamma: np.floating,
    weights: np.ndarray,
) -> "tuple[np.ndarray, np.ndarray]":
    """
    Diffusion, freezing, attachment, and melting: steps i to iv of
    Gravner and Griffeath (2008).

    Parameters
    ----------
    a
        Whether each cell is attached to the crystal, as 0 or 1.
    b
        The boundary mass, the quasi-liquid layer, of each cell.
    c
        The crystal mass, the ice, of each cell.
    d
        The diffusive mass, the vapor, of each cell. It is zero on attached
        cells, as it always is in the model.
    e
        Space for the new vapor.
    count
        Space for the number of attached neighbors of each cell.
    alpha, beta, theta, kappa, mu, gamma
        The parameters of the model, as scalars of the arrays' type.
    one_minus_kappa, one_minus_mu, one_minus_gamma
        One minus the parameters, rounded to the arrays' type once.
    weights
        The numbers 1 to 7, of the arrays' type.

    Returns
    -------
    The new vapor, which was `e`, and space for the next step, which was `d`.
    """
    wrap(a)
    wrap(d)
    attached_now = a[1:-1, 1:-1]
    boundary_mass = b[1:-1, 1:-1]
    crystal_mass = c[1:-1, 1:-1]
    vapor_old = d[1:-1, 1:-1]
    vapor = e[1:-1, 1:-1]
    num_attached = count[1:-1, 1:-1]

    num_attached[...] = 0
    for neighbor in _around(a):
        num_attached += neighbor
    attached = attached_now.astype(bool)

    # i. Diffusion: each unattached cell takes the average of itself and its
    # neighbors, with any attached neighbor reflecting the cell's own vapor.
    # Attached cells hold no vapor, so they add nothing to the sum.
    np.multiply(vapor_old, weights[num_attached], out=vapor)
    for neighbor in _around(d):
        vapor += neighbor
    vapor /= weights[6]
    np.copyto(vapor, vapor_old, where=attached)

    # ii. Freezing: on the boundary, proportion kappa of the vapor becomes ice
    # and the rest quasi-liquid.
    boundary = ~attached & (num_attached > 0)
    boundary_mass[boundary] += one_minus_kappa * vapor[boundary]
    crystal_mass[boundary] += kappa * vapor[boundary]
    vapor[boundary] = 0

    # iii. Attachment: the more attached neighbors a boundary cell has, the
    # less boundary mass it needs to join the crystal.
    wrap(e)
    vapor_nearby = vapor.copy()
    for neighbor in _around(e):
        vapor_nearby += neighbor
    attach = boundary & (
        ((num_attached <= 2) & (boundary_mass >= beta))
        | (
            (num_attached == 3)
            & ((boundary_mass >= 1) | ((vapor_nearby < theta) & (boundary_mass >= alpha)))
        )
        | (num_attached >= 4)
    )
    attached_now[attach] = 1
    crystal_mass[attach] += boundary_mass[attach]
    boundary_mass[attach] = 0

    # iv. Melting: on the new boundary, proportions mu of the quasi-liquid and
    # gamma of the ice return to vapor.
    wrap(a)
    near = np.zeros(attached_now.shape, dtype=bool)
    for neighbor in _around(a):
        near |= neighbor.astype(bool)
    melt = ~attached_now.astype(bool) & near
    vapor[melt] += mu * boundary_mass[melt] + gamma * crystal_mass[melt]
    boundary_mass[melt] *= one_minus_mu
    crystal_mass[melt] *= one_minus_gamma

    return e, d
