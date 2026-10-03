// Grows the crystal off the main thread, so the page stays responsive, with
// the model compiled from the Rust kernel in rust/ to WebAssembly. It matches
// the snowflakes Python package bit for bit, as the package's tests check.

let wasm = null;
let crystal = 0;
let size = 0;
let params = null;
let running = false;

// Milliseconds of growth between frames.
const BUDGET = 40;

async function load() {
  postMessage({ type: "status", text: "Loading…" });
  const response = await fetch("snowflakes.wasm");
  if (!response.ok) throw new Error("could not fetch snowflakes.wasm");
  const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {});
  wasm = instance.exports;
  postMessage({ type: "ready" });
}

function grow(numSteps) {
  const p = params;
  wasm.crystal_grow(crystal, numSteps, p.alpha, p.beta, p.gamma, p.theta, p.kappa, p.mu, p.sigma);
}

function sendFrame(extra = {}) {
  const top = wasm.crystal_frame(crystal);
  // Views of the module's memory, made afresh since it may have grown, then
  // copied so they can be handed to the page.
  const memory = wasm.memory.buffer;
  const cells = size * size;
  const message = {
    type: "frame",
    size,
    steps: wasm.crystal_steps(crystal),
    radius: wasm.crystal_radius(crystal),
    a: new Uint8Array(memory, wasm.crystal_frame_a(crystal), cells).slice(),
    c: new Float32Array(memory, wasm.crystal_frame_c(crystal), cells).slice(),
    d: new Float32Array(memory, wasm.crystal_frame_d(crystal), cells).slice(),
    top,
    ...extra,
  };
  postMessage(message, [message.a.buffer, message.c.buffer, message.d.buffer]);
}

async function run(limit) {
  const pause = () => new Promise((resolve) => setTimeout(resolve, 0));
  let chunk = 1;
  while (running) {
    const start = performance.now();
    let elapsed = 0;
    let steps = 0;
    while (elapsed < BUDGET && wasm.crystal_radius(crystal) < limit) {
      grow(chunk);
      steps += chunk;
      elapsed = performance.now() - start;
      chunk = Math.max(1, Math.round((steps / Math.max(elapsed, 0.1)) * (BUDGET / 4)));
    }
    const done = wasm.crystal_radius(crystal) >= limit;
    sendFrame({ rate: elapsed > 0 ? (1000 * steps) / elapsed : 0 });
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
      if (crystal) wasm.crystal_free(crystal);
      size = message.size;
      params = message.params;
      crystal = wasm.crystal_new(size, message.rho, message.seed >>> 0);
      sendFrame({ rate: 0 });
    } else if (message.type === "params") {
      params = message.params;
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
