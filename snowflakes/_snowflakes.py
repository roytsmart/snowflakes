import types
import numpy as np

__all__ = [
    "initial",
    "step",
    "snowflake",
]


def _kernels(backend: str) -> types.ModuleType:
    """
    The implementation of steps i to iv to use.

    Parameters
    ----------
    backend
        ``"numba"``, ``"numpy"``, or ``"auto"`` for Numba wherever it can be
        imported and NumPy elsewhere, such as in the browser under Pyodide.
    """
    if backend == "auto":
        try:
            from . import _numba as kernels
        except ImportError:
            from . import _numpy as kernels
    elif backend == "numba":
        from . import _numba as kernels
    elif backend == "numpy":
        from . import _numpy as kernels
    else:
        raise ValueError(f"unknown backend {backend!r}")
    return kernels


def initial(
    shape: "tuple[int, int]",
    rho: float,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    A single cell of ice in the middle of vapor of uniform density.

    Parameters
    ----------
    shape
        The number of rows and columns of the grid.
    rho
        The density of the vapor.

    Returns
    -------
    a
        Whether each cell is attached to the crystal.
    b
        The boundary mass, the quasi-liquid layer, of each cell.
    c
        The crystal mass, the ice, of each cell.
    d
        The diffusive mass, the vapor, of each cell.
    """
    a = np.zeros(shape, dtype=bool)
    a[shape[0] // 2, shape[1] // 2] = True
    b = np.zeros(shape)
    c = a.astype(float)
    d = np.where(a, 0, float(rho))
    return a, b, c, d


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
    sigma: float = 0,
    rng: "None | np.random.Generator" = None,
    backend: str = "auto",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Advance the crystal by one update of the model of
    Gravner and Griffeath (2008).

    The grid is a triangular lattice stored on a square array, as in the
    paper: each cell neighbors the cells above, below, left, and right of it,
    and the two along the diagonal from lower left to upper right.
    Its edges wrap around.

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
    sigma
        The size of the random perturbation of the vapor in each cell, the
        noise of step v.
        Without it the model is deterministic and conserves mass exactly.
    rng
        The source of the noise. A new one is used if it is omitted.
    backend
        ``"numba"``, ``"numpy"``, or ``"auto"`` for Numba wherever it can be
        imported and NumPy elsewhere, such as in the browser under Pyodide.
    """
    a, b, c, d = _kernels(backend).step(
        a,
        b,
        c,
        d,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        theta=theta,
        kappa=kappa,
        mu=mu,
    )

    # v. Noise: the vapor in each cell grows or shrinks by proportion sigma,
    # each with probability one half.
    if sigma:
        if rng is None:
            rng = np.random.default_rng()
        factor = np.where(rng.random(d.shape) < 0.5, 1 - sigma, 1 + sigma)
        d = d * factor.astype(d.dtype)

    return a, b, c, d


def snowflake(
    a_0: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    theta: float,
    kappa: float,
    mu: float,
    rho: float,
    sigma: float = 0,
    num_steps: int = 1000,
    num_frames: "None | int" = None,
    seed: "None | int" = None,
    backend: str = "auto",
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Grow a snow crystal from a seed with the model of
    Gravner and Griffeath (2008).

    Parameters
    ----------
    a_0
        Which cells start out as ice. Each holds one unit of crystal mass,
        and every other cell holds vapor of density `rho`.
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
    rho
        The density of the vapor the crystal grows in.
    sigma
        The size of the random perturbation of the vapor in each cell.
    num_steps
        The number of updates to run.
    num_frames
        The number of states to return, evenly spaced from the seed to the
        end of the last update. By default, every state is returned.
    seed
        The seed of the noise, so that a noisy crystal can be grown again.
    backend
        ``"numba"``, ``"numpy"``, or ``"auto"`` for Numba wherever it can be
        imported and NumPy elsewhere.

    Returns
    -------
    a
        Whether each cell is attached to the crystal, in each frame.
    b
        The boundary mass of each cell in each frame.
    c
        The crystal mass of each cell in each frame.
    d
        The diffusive mass of each cell in each frame.
    """
    if num_frames is None:
        num_frames = num_steps + 1
    if num_frames == 1:
        frame_steps = np.array([num_steps])
    else:
        frame_steps = np.round(np.linspace(0, num_steps, num_frames)).astype(int)

    a_n = np.asarray(a_0, dtype=bool)
    b_n = np.zeros(a_n.shape)
    c_n = a_n.astype(float)
    d_n = np.where(a_n, 0, float(rho))
    rng = np.random.default_rng(seed)

    shape = (num_frames,) + a_n.shape
    a = np.empty(shape, dtype=bool)
    b = np.empty(shape)
    c = np.empty(shape)
    d = np.empty(shape)

    f = 0
    for n in range(num_steps + 1):
        while f < num_frames and frame_steps[f] == n:
            a[f], b[f], c[f], d[f] = a_n, b_n, c_n, d_n
            f += 1
        if n == num_steps:
            break
        a_n, b_n, c_n, d_n = step(
            a_n,
            b_n,
            c_n,
            d_n,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            theta=theta,
            kappa=kappa,
            mu=mu,
            sigma=sigma,
            rng=rng,
            backend=backend,
        )

    return a, b, c, d
