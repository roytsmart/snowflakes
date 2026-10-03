"""
Check the Rust kernel of the web app against the package.

The crate in ``rust/`` is built natively with ``cargo build --release``, and
its C interface, the same one the web app calls in WebAssembly, is loaded with
ctypes. These tests are skipped if it has not been built, unless
``SNOWFLAKES_REQUIRE_RUST`` is set, as it is in CI.
"""

import ctypes
import os
import pathlib
import sys

import numpy as np
import pytest

import snowflakes

PRESETS = {
    "fern": dict(rho=0.8, alpha=0.004, beta=1.9, gamma=0.0001, theta=0.001, kappa=0.05, mu=0.015),
    "stellar dendrite": dict(rho=0.635, alpha=0.4, beta=1.6, gamma=0.0005, theta=0.025, kappa=0.005, mu=0.015),
    "plate with dendrite ends": dict(rho=0.38, alpha=0.35, beta=1.06, gamma=0.0006, theta=0.112, kappa=0.001, mu=0.14),
    "plate with ribs": dict(rho=0.37, alpha=0.02, beta=1.09, gamma=0.000001, theta=0.09, kappa=0.003, mu=0.12),
}
"""Parameter sets from the web app, which exercise every branch of attachment."""


def _library() -> "None | ctypes.CDLL":
    """The natively built Rust kernel, if there is one."""
    name = {"win32": "snowflakes.dll", "darwin": "libsnowflakes.dylib"}.get(sys.platform, "libsnowflakes.so")
    crate = pathlib.Path(__file__).parent.parent / "rust"
    targets = [pathlib.Path(os.environ["CARGO_TARGET_DIR"])] if "CARGO_TARGET_DIR" in os.environ else []
    targets.append(crate / "target")
    for target in targets:
        path = target / "release" / name
        if path.exists():
            library = ctypes.CDLL(str(path))
            library.crystal_new.restype = ctypes.c_void_p
            library.crystal_new.argtypes = [ctypes.c_uint32, ctypes.c_double, ctypes.c_uint32]
            library.crystal_free.argtypes = [ctypes.c_void_p]
            library.crystal_grow.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_double] * 7
            library.crystal_steps.restype = ctypes.c_uint32
            library.crystal_steps.argtypes = [ctypes.c_void_p]
            library.crystal_radius.restype = ctypes.c_uint32
            library.crystal_radius.argtypes = [ctypes.c_void_p]
            library.crystal_frame.restype = ctypes.c_float
            library.crystal_frame.argtypes = [ctypes.c_void_p]
            for field in "abcd":
                function = getattr(library, f"crystal_frame_{field}")
                function.restype = ctypes.c_void_p
                function.argtypes = [ctypes.c_void_p]
            return library
    return None


library = _library()

if library is None and os.environ.get("SNOWFLAKES_REQUIRE_RUST"):
    raise RuntimeError("the Rust kernel is not built, but SNOWFLAKES_REQUIRE_RUST is set")

pytestmark = pytest.mark.skipif(library is None, reason="the Rust kernel is not built")


def _grow_rust(size: int, num_steps: int, sigma: float = 0, **params: float):
    assert library is not None
    rho = params.pop("rho")
    crystal = library.crystal_new(size, rho, 7)
    try:
        library.crystal_grow(
            crystal,
            num_steps,
            params["alpha"],
            params["beta"],
            params["gamma"],
            params["theta"],
            params["kappa"],
            params["mu"],
            sigma,
        )
        library.crystal_frame(crystal)
        fields = []
        for field, dtype in zip("abcd", (np.uint8, np.float32, np.float32, np.float32)):
            pointer = getattr(library, f"crystal_frame_{field}")(crystal)
            buffer = (ctypes.c_byte * (size * size * np.dtype(dtype).itemsize)).from_address(pointer)
            fields.append(np.frombuffer(memoryview(buffer), dtype=dtype).reshape(size, size).copy())
        return fields, library.crystal_steps(crystal), library.crystal_radius(crystal)
    finally:
        library.crystal_free(crystal)


@pytest.mark.parametrize("preset", list(PRESETS))
def test_rust_matches_the_package_bit_for_bit(preset: str):
    size, num_steps = 61, 600
    params = dict(PRESETS[preset])
    (a, b, c, d), steps, radius = _grow_rust(size, num_steps, **params)

    rho = params.pop("rho")
    a_0, b_0, c_0, d_0 = snowflakes.initial((size, size), rho)
    expected = snowflakes.step(
        a_0,
        b_0.astype(np.float32),
        c_0.astype(np.float32),
        d_0.astype(np.float32),
        **params,
        backend="numpy",
        num_steps=num_steps,
    )

    assert steps == num_steps
    assert a.sum() > 50
    assert np.array_equal(a.astype(bool), expected[0])
    for x, y in zip((b, c, d), expected[1:]):
        assert np.array_equal(x, y)

    rows, cols = np.nonzero(expected[0])
    q, r = cols - size // 2, rows - size // 2
    assert radius == np.max(np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r)))


def test_rust_noise():
    params = dict(PRESETS["stellar dendrite"])
    (_, _, _, d), _, _ = _grow_rust(41, 200, sigma=0.01, **params)
    (_, _, _, quiet), _, _ = _grow_rust(41, 200, **dict(PRESETS["stellar dendrite"]))
    assert not np.array_equal(d, quiet)
    assert np.all(d >= 0)
