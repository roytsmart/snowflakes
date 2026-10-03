# snowflakes

A Python implementation of the snow crystal growth model of
[Gravner and Griffeath (2008)](https://doi.org/10.1016/j.physd.2007.09.008),
accelerated with Numba where it is installed and running on plain NumPy
elsewhere, with a web app that grows crystals in the browser.

## The web app

**<https://roytsmart.github.io/snowflakes/>** grows a crystal on your phone or
computer, starting from parameter sets from the paper's figures. It runs the
model compiled from Rust to WebAssembly, a 30 KB module which computes exactly
what this package computes in single precision, bit for bit; the tests check
that the two agree.

To run it locally, build the kernel and serve the app:

```bash
cargo build --release --target wasm32-unknown-unknown --manifest-path rust/Cargo.toml
python web/serve.py
```

and open <http://localhost:8000>.

## Installation

```bash
pip install git+https://github.com/roytsmart/snowflakes
```

## Usage

Grow a stellar dendrite, with the parameters of Fig. 11 of the paper, from a
single cell of ice:

```python
import numpy as np
import snowflakes

a_0 = np.zeros((301, 301), dtype=bool)
a_0[150, 150] = True

a, b, c, d = snowflakes.snowflake(
    a_0=a_0,
    alpha=0.4,
    beta=1.6,
    gamma=0.0005,
    theta=0.025,
    kappa=0.005,
    mu=0.015,
    rho=0.635,
    num_steps=2000,
    num_frames=50,
)
```

`a` says which cells are attached to the crystal and `b`, `c`, and `d` hold
the boundary mass, crystal mass, and vapor of each cell, in each of the 50
frames from the seed to the last step. `sigma` adds the paper's noise, with
`seed` to make it repeatable, and `dtype=np.float32` computes in single
precision, which is faster.

`snowflakes.initial()` and `snowflakes.step()` advance a crystal a given
number of updates at a time instead.

The grid is a triangular lattice stored on a square array, as in the paper:
each cell neighbors the cells above, below, left, and right of it and the two
along the diagonal from lower left to upper right, and the edges wrap around.
To see the crystal with its true hexagonal symmetry, put the cell in row `i`
and column `j` at `(j + i / 2, i * sqrt(3) / 2)`.

## Implementations

Steps i to iv of the model are implemented three times, which all compute the
same thing bit for bit: the same operations, on the same values, in the same
order.

- `snowflakes/_numpy.py` is the reference, vectorized with NumPy. It is used
  wherever Numba is missing.
- `snowflakes/_numba.py` makes three passes over the grid in parallel, and is
  used wherever Numba is installed. It is about six times faster than NumPy
  on one thread and over 7,000 updates per second on a 301 by 301 grid with
  many.
- `rust/` is the kernel of the web app, with a C interface which JavaScript
  calls in WebAssembly. Built natively with `cargo build --release`, the tests
  call it with ctypes and compare it with the reference.

The tests also check that the deterministic model conserves mass and that its
crystals keep their six-fold symmetry.
