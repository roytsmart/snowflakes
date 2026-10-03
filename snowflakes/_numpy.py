"""
Steps i to iv of the model, vectorized with NumPy.

This is the implementation used wherever Numba is unavailable, such as in the
browser under Pyodide.
It computes the same update as :mod:`snowflakes._numba` but works on whole
arrays at once, so it needs no compiler.
"""

import numpy as np

__all__ = [
    "step",
]

_NEIGHBORS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))
"""
The offsets, in rows and columns, of the six nearest neighbors of a cell.

The grid is a triangular lattice stored on a square array: each cell
neighbors the cells above, below, left, and right of it, and the two along
one diagonal, but not along the other.
"""


def _neighbors(x: np.ndarray) -> "list[np.ndarray]":
    """
    The six neighbors of every cell of `x`, wrapping around its edges.

    Parameters
    ----------
    x
        A field on the grid.
    """
    padded = np.pad(x, 1, mode="wrap")
    num_y, num_x = x.shape
    return [padded[1 + i : 1 + i + num_y, 1 + j : 1 + j + num_x] for i, j in _NEIGHBORS]


def _num_attached_neighbors(a: np.ndarray) -> np.ndarray:
    """
    The number of attached neighbors of every cell.

    Parameters
    ----------
    a
        Whether each cell is attached to the crystal.
    """
    result = np.zeros(a.shape, dtype=np.int8)
    for neighbor in _neighbors(a):
        result += neighbor
    return result


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

    Parameters
    ----------
    a
        Whether each cell is attached to the crystal.
    b
        The boundary mass, the quasi-liquid layer, of each cell.
    c
        The crystal mass, the ice, of each cell.
    d
        The diffusive mass, the vapor, of each cell.
        It must be zero on attached cells, as it always is in the model.
    alpha
        The boundary mass a cell with three attached neighbors needs to
        attach once the vapor around it is below `theta`.
    beta
        The boundary mass a cell with one or two attached neighbors needs to
        attach.
    gamma
        The proportion of the crystal mass of boundary cells which melts back
        into vapor each step.
    theta
        The vapor in the neighborhood of a cell below which it can attach
        with only `alpha` of boundary mass.
    kappa
        The proportion of the vapor reaching the boundary which freezes
        directly into crystal mass.
    mu
        The proportion of the boundary mass of boundary cells which melts
        back into vapor each step.
    """
    num_attached = _num_attached_neighbors(a)

    # i. Diffusion: each unattached cell takes the average of itself and its
    # neighbors, with any attached neighbor reflecting the cell's own vapor.
    # Attached cells hold no vapor, so they add nothing to the sum.
    total = d * (1 + num_attached)
    for neighbor in _neighbors(d):
        total += neighbor
    d = np.where(a, d, total / 7)

    # ii. Freezing: on the boundary, proportion kappa of the vapor becomes ice
    # and the rest quasi-liquid.
    boundary = ~a & (num_attached > 0)
    b = np.where(boundary, b + (1 - kappa) * d, b)
    c = np.where(boundary, c + kappa * d, c)
    d = np.where(boundary, 0, d)

    # iii. Attachment: the more attached neighbors a boundary cell has, the
    # less boundary mass it needs to join the crystal.
    vapor_nearby = d.copy()
    for neighbor in _neighbors(d):
        vapor_nearby += neighbor
    attach = boundary & (
        ((num_attached <= 2) & (b >= beta))
        | ((num_attached == 3) & ((b >= 1) | ((vapor_nearby < theta) & (b >= alpha))))
        | (num_attached >= 4)
    )
    a = a | attach
    c = np.where(attach, b + c, c)
    b = np.where(attach, 0, b)

    # iv. Melting: on the new boundary, proportions mu of the quasi-liquid and
    # gamma of the ice return to vapor.
    boundary = ~a & (_num_attached_neighbors(a) > 0)
    d = np.where(boundary, d + mu * b + gamma * c, d)
    b = np.where(boundary, (1 - mu) * b, b)
    c = np.where(boundary, (1 - gamma) * c, c)

    return a, b, c, d
