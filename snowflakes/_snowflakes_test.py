import pathlib
import re

import numpy as np
import pytest

import snowflakes

PARAMS = dict(alpha=0.4, beta=1.6, gamma=0.0005, theta=0.025, kappa=0.005, mu=0.015)
"""Fig. 11 of Gravner and Griffeath (2008), with kappa = 0.005: a stellar dendrite."""

RHO = 0.635
"""The vapor density of Fig. 11."""

def _backends() -> "list[str]":
    """The backends that can run here: NumPy always, and Numba if it is installed."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return ["numpy"]
    return ["numpy", "numba"]


def _grow(
    backend: str,
    num_steps: int,
    shape: "tuple[int, int]" = (41, 41),
    sigma: float = 0,
    seed: "None | int" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    state = snowflakes.initial(shape, RHO)
    rng = np.random.default_rng(seed)
    for _ in range(num_steps):
        state = snowflakes.step(*state, **PARAMS, sigma=sigma, rng=rng, backend=backend)
    return state


@pytest.mark.parametrize("backend", _backends())
def test_mass_is_conserved(backend: str):
    a, b, c, d = snowflakes.initial((41, 41), RHO)
    total = b.sum() + c.sum() + d.sum()
    a, b, c, d = _grow(backend, num_steps=200)
    assert a.sum() > 100
    assert np.isclose(b.sum() + c.sum() + d.sum(), total, rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_backends_agree_bit_for_bit(dtype: type):
    pytest.importorskip("numba")
    a, b, c, d = snowflakes.initial((61, 61), RHO)
    state = (a, b.astype(dtype), c.astype(dtype), d.astype(dtype))
    numpy_state = snowflakes.step(*state, **PARAMS, backend="numpy", num_steps=500)
    numba_state = snowflakes.step(*state, **PARAMS, backend="numba", num_steps=500)
    assert numpy_state[0].sum() > 500
    for x, y in zip(numpy_state, numba_state):
        assert x.dtype == y.dtype
        assert np.array_equal(x, y)


def test_step_leaves_its_arguments_alone():
    state = snowflakes.initial((21, 21), RHO)
    copies = tuple(x.copy() for x in state)
    snowflakes.step(*state, **PARAMS, backend="numpy", num_steps=10)
    for x, y in zip(state, copies):
        assert np.array_equal(x, y)


def test_num_steps():
    once = _grow("numpy", num_steps=20)
    together = snowflakes.step(*snowflakes.initial((41, 41), RHO), **PARAMS, backend="numpy", num_steps=20)
    for x, y in zip(once, together):
        assert np.array_equal(x, y)


@pytest.mark.parametrize("backend", _backends())
def test_six_fold_symmetry(backend: str):
    """Without noise, the crystal is unchanged by a turn of 60 degrees about its seed."""
    n = 41
    a, _, c, _ = _grow(backend, num_steps=300, shape=(n, n))
    rows, cols = np.indices(a.shape)
    q = cols - n // 2
    r = rows - n // 2
    # A turn of 60 degrees on the lattice takes (q, r) to (-r, q + r).
    rows_turned = (q + r + n // 2) % n
    cols_turned = (-r + n // 2) % n
    assert a[a].size > 100
    assert np.array_equal(a[rows_turned, cols_turned], a)
    assert np.allclose(c[rows_turned, cols_turned], c, rtol=1e-9, atol=1e-12)


def test_noise():
    noisy = _grow("numpy", num_steps=200, sigma=0.01, seed=1)
    again = _grow("numpy", num_steps=200, sigma=0.01, seed=1)
    quiet = _grow("numpy", num_steps=200)
    for x, y in zip(noisy, again):
        assert np.array_equal(x, y)
    assert not np.array_equal(noisy[3], quiet[3])


def test_single_precision():
    a, b, c, d = snowflakes.initial((41, 41), RHO)
    state = (a, b.astype(np.float32), c.astype(np.float32), d.astype(np.float32))
    for _ in range(50):
        state = snowflakes.step(*state, **PARAMS, sigma=0.001, backend="numpy")
    assert all(x.dtype == np.float32 for x in state[1:])


@pytest.mark.parametrize(
    argnames="num_steps,num_frames,frame_steps",
    argvalues=[
        (10, 3, [0, 5, 10]),
        (10, 4, [0, 3, 7, 10]),
        (10, 1, [10]),
        (4, None, [0, 1, 2, 3, 4]),
    ],
)
def test_snowflake_frames(num_steps: int, num_frames: "None | int", frame_steps: "list[int]"):
    a_0 = np.zeros((21, 21), dtype=bool)
    a_0[10, 10] = True
    a, b, c, d = snowflakes.snowflake(
        a_0=a_0,
        **PARAMS,
        rho=RHO,
        num_steps=num_steps,
        num_frames=num_frames,
        backend="numpy",
    )
    assert a.shape == (len(frame_steps), 21, 21)
    assert a.dtype == bool
    for frame, n in enumerate(frame_steps):
        expected = _grow("numpy", num_steps=n, shape=(21, 21))
        for x, y in zip((a, b, c, d), expected):
            assert np.array_equal(x[frame], y)


def test_unknown_backend():
    with pytest.raises(ValueError):
        snowflakes.step(*snowflakes.initial((5, 5), RHO), **PARAMS, backend="cuda")


def test_numpy_backend_needs_no_numba():
    """The NumPy implementation runs where Numba cannot, such as under Pyodide."""
    package = pathlib.Path(snowflakes.__file__).parent
    for name in ("__init__.py", "_snowflakes.py", "_numpy.py"):
        imports = re.findall(r"^\s*(?:import|from) (\w+)", (package / name).read_text(), re.M)
        assert "numba" not in imports
