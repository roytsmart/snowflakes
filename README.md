# snowflakes

A Python implementation of the snow crystal growth model of
[Gravner and Griffeath (2008)](https://doi.org/10.1016/j.physd.2007.09.008),
accelerated with Numba where it is installed and running on plain NumPy
elsewhere, including in the browser.

## The web app

**<https://roytsmart.github.io/snowflakes/>** grows a crystal on your phone or
computer. It runs this package in Python in the browser with
[Pyodide](https://pyodide.org), using the NumPy implementation of the model,
and offers parameter sets from the paper's figures to start from.

To run it locally, serve it with

```bash
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
`seed` to make it repeatable.

`snowflakes.initial()` and `snowflakes.step()` advance a crystal one update at
a time instead, as the web app does.

The grid is a triangular lattice stored on a square array, as in the paper:
each cell neighbors the cells above, below, left, and right of it and the two
along one diagonal, and the edges wrap around. To see the crystal with its true
hexagonal symmetry, put the cell in row `i` and column `j` at
`(j + i / 2, i * sqrt(3) / 2)`.

## Implementations

Steps i to iv of the model are implemented twice: once compiled with Numba,
cell by cell, and once vectorized with NumPy for wherever Numba is missing.
`backend="auto"`, the default, picks Numba if it can be imported. The tests
check that the two agree, that the deterministic model conserves mass, and that
its crystals keep their six-fold symmetry.
