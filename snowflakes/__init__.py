"""
Grow snow crystals with the mesoscopic model of Gravner and Griffeath (2008),
"Modeling snow crystal growth II: A mesoscopic lattice map with plausible
dynamics", Physica D 237, 385.

The model runs on Numba where it is installed and on NumPy elsewhere, such as
in the browser under Pyodide.
"""

from ._snowflakes import initial, step, snowflake

__all__ = [
    "initial",
    "step",
    "snowflake",
]
