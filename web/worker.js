// Runs the snowflakes package in Python, in Pyodide, off the main thread so
// the page stays responsive while the crystal grows. It is a module worker:
// Edge refuses to load Pyodide's classic script into a worker from another
// origin with importScripts, but imports its module fine.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

// The full distribution, which carries NumPy; the npm package has only the
// core of Pyodide.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";

// The parts of the package the browser needs. Numba cannot run in the
// browser, so the NumPy implementation stands in for it.
const FILES = ["__init__.py", "_snowflakes.py", "_numpy.py"];

const SIMULATION = `
import numpy as np
import snowflakes


class Simulation:
    """A crystal growing on a periodic grid, one batch of updates at a time."""

    def __init__(self, size, rho, seed):
        a, b, c, d = snowflakes.initial((size, size), rho)
        # Single precision is plenty for the picture and half the memory
        # traffic, which is what limits NumPy here.
        self.state = (a, b.astype(np.float32), c.astype(np.float32), d.astype(np.float32))
        self.size = size
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.params = {}

    def set_params(self, **params):
        self.params = params

    def advance(self, num_steps):
        state = self.state
        for _ in range(num_steps):
            state = snowflakes.step(*state, **self.params, rng=self.rng, backend="numpy")
        self.state = state
        self.steps += num_steps

    def radius(self):
        """The distance, in cells, from the seed to the farthest attached cell."""
        rows, cols = np.nonzero(self.state[0])
        r = rows - self.size // 2
        q = cols - self.size // 2
        return int(np.max(np.maximum(np.maximum(np.abs(q), np.abs(r)), np.abs(q + r))))

    def frame(self):
        a, b, c, d = self.state
        ice = c[a]
        top = float(np.percentile(ice, 99)) if ice.size > 1 else 1.0
        return a.view(np.uint8), c, d, top
`;

let pyodide = null;
let simulation = null;
// Bound methods of the simulation, looked up once rather than on every call,
// since each lookup makes a proxy that has to be released.
let methods = null;
let running = false;
let budget = 60; // milliseconds of Python per frame sent

function copy(proxy, type) {
  const buffer = proxy.getBuffer(type);
  const data = buffer.data.slice();
  buffer.release();
  proxy.destroy();
  return data;
}

function sendFrame(extra = {}) {
  const frame = methods.frame();
  const [a, c, d, top] = frame.toJs({ depth: 1 });
  frame.destroy();
  const message = {
    type: "frame",
    size: simulation.size,
    steps: simulation.steps,
    radius: methods.radius(),
    a: copy(a, "u8"),
    c: copy(c, "f32"),
    d: copy(d, "f32"),
    top,
    ...extra,
  };
  postMessage(message, [message.a.buffer, message.c.buffer, message.d.buffer]);
}

async function load() {
  postMessage({ type: "status", text: "Loading Python…" });
  pyodide = await loadPyodide({ indexURL: PYODIDE });
  postMessage({ type: "status", text: "Loading NumPy…" });
  await pyodide.loadPackage("numpy");
  postMessage({ type: "status", text: "Loading the model…" });
  pyodide.FS.mkdirTree("/home/pyodide/snowflakes");
  for (const name of FILES) {
    const response = await fetch(`snowflakes/${name}`);
    if (!response.ok) throw new Error(`could not fetch snowflakes/${name}`);
    pyodide.FS.writeFile(`/home/pyodide/snowflakes/${name}`, await response.text());
  }
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
  pyodide.runPython(SIMULATION);
  postMessage({ type: "ready", python: pyodide.runPython("import sys; sys.version.split()[0]") });
}

async function run(limit) {
  const pause = () => new Promise((resolve) => setTimeout(resolve, 0));
  let chunk = 1;
  while (running) {
    const start = performance.now();
    let elapsed = 0;
    let steps = 0;
    while (elapsed < budget) {
      methods.advance(chunk);
      steps += chunk;
      elapsed = performance.now() - start;
      // Aim for a handful of calls into Python per frame.
      chunk = Math.max(1, Math.round((steps / elapsed) * (budget / 4)));
    }
    const done = methods.radius() >= limit;
    sendFrame({ rate: (1000 * steps) / elapsed });
    if (done) {
      running = false;
      postMessage({ type: "done" });
    }
    // Let pause, restart, and new parameters through.
    await pause();
  }
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      await load();
    } else if (message.type === "reset") {
      running = false;
      if (simulation) {
        for (const method of Object.values(methods)) method.destroy();
        simulation.destroy();
      }
      const Simulation = pyodide.globals.get("Simulation");
      simulation = Simulation(message.size, message.rho, message.seed);
      Simulation.destroy();
      methods = {
        advance: simulation.advance,
        radius: simulation.radius,
        frame: simulation.frame,
        setParams: simulation.set_params,
      };
      methods.setParams.callKwargs(message.params);
      sendFrame({ rate: 0 });
    } else if (message.type === "params") {
      methods.setParams.callKwargs(message.params);
    } else if (message.type === "play") {
      if (!running) {
        running = true;
        run(message.limit);
      }
    } else if (message.type === "pause") {
      running = false;
    }
  } catch (error) {
    running = false;
    postMessage({ type: "error", text: String(error && error.message ? error.message : error) });
  }
};
