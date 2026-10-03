import types
import typing
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


def _pad(x: np.ndarray, dtype: "type | np.dtype") -> np.ndarray:
    """`x` as `dtype`, padded by one cell on each side."""
    result = np.zeros((x.shape[0] + 2, x.shape[1] + 2), dtype=dtype)
    result[1:-1, 1:-1] = x
    return result


class _Grid:
    """
    The state of the model, padded as the kernels need it, so that it can be
    advanced many times without copying.
    """

    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        backend: str,
    ):
        dtype = np.result_type(b, c, d)
        if dtype.kind != "f":
            dtype = np.dtype(float)
        self.dtype = dtype
        self.a = _pad(a, np.uint8)
        self.b = _pad(b, dtype)
        self.c = _pad(c, dtype)
        self.d = _pad(d, dtype)
        self.e = np.zeros_like(self.d)
        self.count = np.zeros_like(self.a)
        self.kernels = _kernels(backend)

    def coefficients(
        self,
        alpha: float,
        beta: float,
        gamma: float,
        theta: float,
        kappa: float,
        mu: float,
    ) -> "dict[str, typing.Any]":
        """The parameters as the kernels take them, rounded to the grid's type once."""
        t = self.dtype.type
        return dict(
            alpha=t(alpha),
            beta=t(beta),
            theta=t(theta),
            kappa=t(kappa),
            one_minus_kappa=t(1 - kappa),
            mu=t(mu),
            one_minus_mu=t(1 - mu),
            gamma=t(gamma),
            one_minus_gamma=t(1 - gamma),
            weights=np.arange(1, 8, dtype=self.dtype),
        )

    def step(
        self,
        coefficients: "dict[str, typing.Any]",
        sigma: float,
        rng: "None | np.random.Generator",
    ) -> None:
        """Advance the crystal by one update."""
        self.d, self.e = self.kernels.step(
            self.a,
            self.b,
            self.c,
            self.d,
            self.e,
            self.count,
            **coefficients,
        )

        # v. Noise: the vapor in each cell grows or shrinks by proportion
        # sigma, each with probability one half.
        if sigma:
            if rng is None:
                rng = np.random.default_rng()
            vapor = self.d[1:-1, 1:-1]
            factor = np.where(rng.random(vapor.shape) < 0.5, 1 - sigma, 1 + sigma)
            vapor *= factor.astype(self.dtype)

    def state(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Copies of the unpadded state."""
        inside = (slice(1, -1), slice(1, -1))
        return (
            self.a[inside].astype(bool),
            self.b[inside].copy(),
            self.c[inside].copy(),
            self.d[inside].copy(),
        )


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
    num_steps: int = 1,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """
    Advance a crystal by updates of the model of Gravner and Griffeath (2008).

    The grid is a triangular lattice stored on a square array, as in the
    paper: each cell neighbors the cells above, below, left, and right of it,
    and the two along the diagonal from lower left to upper right.
    Its edges wrap around.

    The state is computed in the precision of `b`, `c`, and `d`, so single
    precision arrays give a faster, single precision result.

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
    num_steps
        The number of updates to make.

    Returns
    -------
    The new `a`, `b`, `c`, and `d`. The arguments are left unchanged.
    """
    grid = _Grid(a, b, c, d, backend=backend)
    coefficients = grid.coefficients(alpha, beta, gamma, theta, kappa, mu)
    for _ in range(num_steps):
        grid.step(coefficients, sigma=sigma, rng=rng)
    return grid.state()


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
    dtype: "type | np.dtype" = float,
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
    dtype
        The precision to compute in. Single precision is about twice as fast.

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

    a_0 = np.asarray(a_0, dtype=bool)
    grid = _Grid(
        a=a_0,
        b=np.zeros(a_0.shape, dtype=dtype),
        c=a_0.astype(dtype),
        d=np.where(a_0, 0, rho).astype(dtype),
        backend=backend,
    )
    coefficients = grid.coefficients(alpha, beta, gamma, theta, kappa, mu)
    rng = np.random.default_rng(seed)

    shape = (num_frames,) + a_0.shape
    a = np.empty(shape, dtype=bool)
    b = np.empty(shape, dtype=grid.dtype)
    c = np.empty(shape, dtype=grid.dtype)
    d = np.empty(shape, dtype=grid.dtype)

    f = 0
    for n in range(num_steps + 1):
        while f < num_frames and frame_steps[f] == n:
            a[f], b[f], c[f], d[f] = grid.state()
            f += 1
        if n == num_steps:
            break
        grid.step(coefficients, sigma=sigma, rng=rng)

    return a, b, c, d
