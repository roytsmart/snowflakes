"use strict";

// The model's parameters, as the sliders show them. Those spanning orders of
// magnitude get a logarithmic slider, whose far left end is zero.
const PARAMETERS = [
  { key: "rho", symbol: "ρ", name: "vapor density", min: 0.3, max: 0.9, scale: "linear", restart: true },
  { key: "beta", symbol: "β", name: "attachment threshold at tips", min: 1, max: 3.5, scale: "linear" },
  { key: "alpha", symbol: "α", name: "knife-edge threshold", min: 1e-4, max: 1, scale: "log" },
  { key: "theta", symbol: "θ", name: "knife-edge vapor limit", min: 1e-4, max: 0.5, scale: "log" },
  { key: "kappa", symbol: "κ", name: "direct freezing", min: 1e-5, max: 0.5, scale: "log" },
  { key: "mu", symbol: "μ", name: "melting of the quasi-liquid", min: 1e-4, max: 0.5, scale: "log" },
  { key: "gamma", symbol: "γ", name: "melting of the ice", min: 1e-7, max: 1e-2, scale: "log" },
  { key: "sigma", symbol: "σ", name: "noise", min: 1e-7, max: 1e-3, scale: "log" },
];

// Parameter sets from the appendix of Gravner and Griffeath (2008).
const PRESETS = [
  {
    "name": "Fern",
    "note": "Fig. 10 of the paper, with β = 1.9: weak anisotropy lets side branches sprout all along the arms.",
    "params": {
      "rho": 0.8,
      "beta": 1.9,
      "alpha": 0.004,
      "theta": 0.001,
      "kappa": 0.05,
      "mu": 0.015,
      "gamma": 0.0001
    }
  },
  {
    "name": "Stellar dendrite",
    "note": "Fig. 11 of the paper, with κ = 0.005: six feathery arms.",
    "params": {
      "rho": 0.635,
      "beta": 1.6,
      "alpha": 0.4,
      "theta": 0.025,
      "kappa": 0.005,
      "mu": 0.015,
      "gamma": 0.0005
    }
  },
  {
    "name": "Delicate dendrite",
    "note": "Fig. 12 of the paper, with μ = 0.04: thin arms with parabolic tips.",
    "params": {
      "rho": 0.5,
      "beta": 1.4,
      "alpha": 0.1,
      "theta": 0.005,
      "kappa": 0.001,
      "mu": 0.04,
      "gamma": 0.001
    }
  },
  {
    "name": "Simple star",
    "note": "Fig. 13, left: strong direct freezing (κ = 0.15) stops the six arms from branching.",
    "params": {
      "rho": 0.65,
      "beta": 1.75,
      "alpha": 0.2,
      "theta": 0.026,
      "kappa": 0.15,
      "mu": 0.015,
      "gamma": 1e-05
    }
  },
  {
    "name": "Plate with dendrite ends",
    "note": "Fig. 13, right: a central plate keeps filling in between six branching arms.",
    "params": {
      "rho": 0.38,
      "beta": 1.06,
      "alpha": 0.35,
      "theta": 0.112,
      "kappa": 0.001,
      "mu": 0.14,
      "gamma": 0.0006
    }
  },
  {
    "name": "Sectored plate",
    "note": "Fig. 16, right: broad faceted branches, with a little noise in the vapor (σ = 0.00005).",
    "params": {
      "rho": 0.8,
      "beta": 2.6,
      "alpha": 0.006,
      "theta": 0.005,
      "kappa": 0.05,
      "mu": 0.015,
      "gamma": 0.0001,
      "sigma": 5e-05
    }
  },
  {
    "name": "Plate with ribs",
    "note": "Fig. 14, right: the plate stops and restarts, leaving hexagonal ribs. Slow.",
    "params": {
      "rho": 0.37,
      "beta": 1.09,
      "alpha": 0.02,
      "theta": 0.09,
      "kappa": 0.003,
      "mu": 0.12,
      "gamma": 1e-06
    }
  },
  {
    "name": "Hexagonal plate",
    "note": "Fig. 10 of the paper, with β = 2.8: strong anisotropy makes a faceted plate. Slow.",
    "params": {
      "rho": 0.8,
      "beta": 2.8,
      "alpha": 0.004,
      "theta": 0.001,
      "kappa": 0.05,
      "mu": 0.015,
      "gamma": 0.0001
    }
  }
];

const canvas = document.getElementById("crystal");
const context = canvas.getContext("2d");
const overlay = document.getElementById("overlay");
const status = document.getElementById("status");
const playButton = document.getElementById("play");
const restartButton = document.getElementById("restart");
const saveButton = document.getElementById("save");
const presetSelect = document.getElementById("preset");
const presetNote = document.getElementById("preset-note");
const sizeSelect = document.getElementById("size");
const sliders = document.getElementById("sliders");

const worker = new Worker("worker.js", { type: "module" });

let params = {};
let size = 201;
let running = false;
let ready = false;
let frame = null;
let resumeWhenVisible = false;

// ---- Parameters -----------------------------------------------------------

const RESOLUTION = 1000;

function toSlider(parameter, value) {
  if (parameter.scale === "linear") {
    return Math.round(((value - parameter.min) / (parameter.max - parameter.min)) * RESOLUTION);
  }
  if (value <= 0) return 0;
  const position = Math.log(value / parameter.min) / Math.log(parameter.max / parameter.min);
  return Math.min(RESOLUTION, Math.max(1, Math.round(position * RESOLUTION)));
}

function fromSlider(parameter, position) {
  if (parameter.scale === "linear") {
    return parameter.min + ((parameter.max - parameter.min) * position) / RESOLUTION;
  }
  if (position <= 0) return 0;
  return parameter.min * Math.pow(parameter.max / parameter.min, position / RESOLUTION);
}

function format(value) {
  return String(Number(value.toPrecision(3)));
}

const controls = {};

for (const parameter of PARAMETERS) {
  const row = document.createElement("div");
  row.className = "slider";
  const id = `parameter-${parameter.key}`;
  row.innerHTML = `
    <label for="${id}">${parameter.symbol} <small>${parameter.name}</small></label>
    <input type="number" id="${id}" inputmode="decimal" step="any" min="0">
    <input type="range" min="0" max="${RESOLUTION}" step="1" aria-label="${parameter.symbol}, ${parameter.name}">`;
  const [number, range] = row.querySelectorAll("input");
  range.addEventListener("input", () => {
    setParameter(parameter, fromSlider(parameter, Number(range.value)), "range");
  });
  number.addEventListener("change", () => {
    const value = Number(number.value);
    if (Number.isFinite(value) && value >= 0) setParameter(parameter, value, "number");
    else number.value = format(params[parameter.key]);
  });
  controls[parameter.key] = { number, range };
  sliders.append(row);
}

function show(parameter, from = null) {
  const value = params[parameter.key];
  if (from !== "number") controls[parameter.key].number.value = format(value);
  if (from !== "range") controls[parameter.key].range.value = toSlider(parameter, value);
}

function setParameter(parameter, value, from) {
  params[parameter.key] = value;
  show(parameter, from);
  presetSelect.value = "custom";
  presetNote.textContent = "Your own parameters.";
  if (!parameter.restart) worker.postMessage({ type: "params", params: stepParams() });
}

function stepParams() {
  const { rho, ...rest } = params;
  return rest;
}

// ---- Presets --------------------------------------------------------------

PRESETS.forEach((preset, index) => {
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = preset.name;
  presetSelect.append(option);
});
const custom = document.createElement("option");
custom.value = "custom";
custom.textContent = "Custom";
presetSelect.append(custom);

function applyPreset(index) {
  const preset = PRESETS[index];
  params = { sigma: 0, ...preset.params };
  for (const parameter of PARAMETERS) show(parameter);
  presetSelect.value = String(index);
  presetNote.textContent = preset.note;
}

presetSelect.addEventListener("change", () => {
  if (presetSelect.value === "custom") return;
  applyPreset(Number(presetSelect.value));
  restart(true);
});

// A smaller grid on a phone, where Python runs several times slower.
size = window.matchMedia("(max-width: 700px)").matches ? 201 : 301;
sizeSelect.value = String(size);
sizeSelect.addEventListener("change", () => {
  size = Number(sizeSelect.value);
  restart(true);
});

// ---- Running --------------------------------------------------------------

function restart(play) {
  if (!ready) return;
  view = null;
  worker.postMessage({
    type: "reset",
    size,
    rho: params.rho,
    seed: Math.floor(Math.random() * 2 ** 31),
    params: stepParams(),
  });
  setRunning(play);
}

function setRunning(value) {
  running = value;
  playButton.textContent = running ? "Pause" : "Grow";
  if (running) worker.postMessage({ type: "play", limit: Math.floor(0.4 * size) });
  else worker.postMessage({ type: "pause" });
}

playButton.addEventListener("click", () => setRunning(!running));
canvas.addEventListener("click", () => {
  if (ready) setRunning(!running);
});
restartButton.addEventListener("click", () => restart(true));

document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) {
    resumeWhenVisible = true;
    setRunning(false);
  } else if (!document.hidden && resumeWhenVisible) {
    resumeWhenVisible = false;
    setRunning(true);
  }
});

worker.onmessage = (event) => {
  const message = event.data;
  if (message.type === "status") {
    overlay.textContent = message.text;
  } else if (message.type === "ready") {
    ready = true;
    overlay.textContent = "";
    for (const button of [playButton, restartButton, saveButton]) button.disabled = false;
    restart(true);
  } else if (message.type === "frame") {
    frame = message;
    paint();
    report();
  } else if (message.type === "done") {
    setRunning(false);
    status.textContent += " · fully grown";
  } else if (message.type === "error") {
    setRunning(false);
    overlay.textContent = `Something went wrong: ${message.text}`;
  }
};

worker.postMessage({ type: "load" });

function report() {
  const parts = [`Step ${frame.steps.toLocaleString()}`];
  if (frame.rate) parts.push(`${Math.round(frame.rate)} steps/s`);
  parts.push(`${2 * frame.radius + 1} cells across`);
  status.textContent = parts.join(" · ");
}

// ---- Drawing --------------------------------------------------------------

// The grid is a triangular lattice stored on a square array: a cell's
// neighbors are left, right, up, down, and along one diagonal. A shear of
// the image puts every cell back at its place on the lattice.
const SHEAR = Math.sqrt(3) / 2;
const SKY = [8, 16, 38];
const VAPOR = [30, 54, 108];
const ICE_DIM = [96, 146, 214];
const ICE_BRIGHT = [244, 249, 255];

const base = document.createElement("canvas");
const baseContext = base.getContext("2d");
const tiled = document.createElement("canvas");
const tiledContext = tiled.getContext("2d");
let image = null;
let margin = 0;
let view = null;

function paint() {
  const n = frame.size;
  if (!image || image.width !== n) {
    base.width = base.height = n;
    image = baseContext.createImageData(n, n);
    // The grid wraps around, so copies of it on every side fill the corners
    // of the view however far it zooms out.
    margin = Math.ceil(0.3 * n);
    tiled.width = tiled.height = n + 2 * margin;
  }
  const pixels = image.data;
  const { a, c, d } = frame;
  const rho = params.rho;
  const top = frame.top > 0 ? frame.top : 1;
  for (let k = 0, p = 0; k < a.length; k++, p += 4) {
    let color0, color1, t;
    if (a[k]) {
      t = Math.pow(Math.min(c[k] / top, 1), 0.8);
      color0 = ICE_DIM;
      color1 = ICE_BRIGHT;
    } else {
      t = Math.min(Math.max(d[k] / rho, 0), 1);
      color0 = SKY;
      color1 = VAPOR;
    }
    pixels[p] = color0[0] + (color1[0] - color0[0]) * t;
    pixels[p + 1] = color0[1] + (color1[1] - color0[1]) * t;
    pixels[p + 2] = color0[2] + (color1[2] - color0[2]) * t;
    pixels[p + 3] = 255;
  }
  baseContext.putImageData(image, 0, 0);
  for (let u = -1; u <= 1; u++) {
    for (let v = -1; v <= 1; v++) {
      tiledContext.drawImage(base, margin + u * n, margin + v * n);
    }
  }
  draw();
}

function draw() {
  if (!frame) return;
  const n = frame.size;
  const width = canvas.width;
  const height = canvas.height;
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.fillStyle = `rgb(${SKY.join(",")})`;
  context.fillRect(0, 0, width, height);

  // Follow the crystal as it grows, from close up to the whole grid.
  const target = Math.min(Math.max(1.35 * frame.radius + 6, 0.12 * n), 0.48 * n);
  view = view === null ? target : view + (target - view) * 0.2;
  const scale = Math.min(width, height) / 2 / view;

  // Cell (row, column) sits at (column + row / 2, row * sqrt(3) / 2).
  const center = Math.floor(n / 2) + 0.5 + margin;
  context.setTransform(
    scale,
    0,
    scale / 2,
    scale * SHEAR,
    width / 2 - scale * (center + center / 2),
    height / 2 - scale * SHEAR * center,
  );
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(tiled, 0, 0);
}

function resize() {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.round(canvas.clientWidth * ratio);
  const height = Math.round(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    draw();
  }
}
new ResizeObserver(resize).observe(canvas);
resize();

saveButton.addEventListener("click", () => {
  canvas.toBlob((blob) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `snowflake-step-${frame ? frame.steps : 0}.png`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  }, "image/png");
});

applyPreset(0);
