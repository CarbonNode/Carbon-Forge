const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const preview = document.getElementById("preview");
const beforeImg = document.getElementById("before-img");
const afterImg = document.getElementById("after-img");
const wipeLine = document.getElementById("wipe-line");
const particlesCanvas = document.getElementById("particles");
const spinner = document.getElementById("spinner");
const spinnerText = document.getElementById("spinner-text");
const actions = document.getElementById("actions");
const downloadBtn = document.getElementById("download-btn");
const reprocessBtn = document.getElementById("reprocess-btn");
const resetBtn = document.getElementById("reset-btn");
const statusEl = document.getElementById("status");
const settingsToggle = document.getElementById("settings-toggle");
const settingsPanel = document.getElementById("settings");
const alphaMatting = document.getElementById("alpha-matting");
const mattingOptions = document.getElementById("matting-options");
const fgThreshold = document.getElementById("fg-threshold");
const bgThreshold = document.getElementById("bg-threshold");
const erodeSize = document.getElementById("erode-size");
const fgVal = document.getElementById("fg-val");
const bgVal = document.getElementById("bg-val");
const erodeVal = document.getElementById("erode-val");
const modelSelect = document.getElementById("model-select");
const colorRemove = document.getElementById("color-remove");
const colorOptions = document.getElementById("color-options");
const colorPicker = document.getElementById("color-picker");
const colorHex = document.getElementById("color-hex");
const colorTolerance = document.getElementById("color-tolerance");
const tolVal = document.getElementById("tol-val");
const edgeSmooth = document.getElementById("edge-smooth");
const edgeOptions = document.getElementById("edge-options");
const edgeStrength = document.getElementById("edge-strength");
const edgeStrengthVal = document.getElementById("edge-strength-val");
const edgeTrim = document.getElementById("edge-trim");
const edgeTrimVal = document.getElementById("edge-trim-val");

// Watermark removal elements
const watermarkRemove = document.getElementById("watermark-remove");
const watermarkOptions = document.getElementById("watermark-options");
const watermarkPosition = document.getElementById("watermark-position");
const watermarkSize = document.getElementById("watermark-size");
const wmSizeVal = document.getElementById("wm-size-val");
const autoTrimCheck = document.getElementById("auto-trim");

// Batch elements
const batchPanel = document.getElementById("batch-panel");
const batchCounter = document.getElementById("batch-counter");
const batchProgressFill = document.getElementById("batch-progress-fill");
const batchList = document.getElementById("batch-list");
const batchCancelBtn = document.getElementById("batch-cancel-btn");
const batchOpenBtn = document.getElementById("batch-open-btn");
const batchDoneBtn = document.getElementById("batch-done-btn");

// Sprite split elements
const spritePanel = document.getElementById("sprite-panel");
const spriteCount = document.getElementById("sprite-count");
const spriteGrid = document.getElementById("sprite-grid");
const spriteSaveBtn = document.getElementById("sprite-save-btn");
const spriteResetBtn = document.getElementById("sprite-reset-btn");

let resultBlob = null;
let originalName = "image";
let originalBytes = null;
let pickedColors = [];
let editOnlyMode = false;
let splitMode = false;
let batchCancelled = false;
let spriteBlobs = []; // Array of Blobs for split sprites

// Prevent Electron's default file drop behavior
document.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); });
document.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); });

// Backend status
let backendReady = false;
const dropContent = document.getElementById("drop-content");
const savedDropChildren = dropContent ? [...dropContent.childNodes] : [];

function setDropZoneLoading() {
  if (!dropContent) return;
  while (dropContent.firstChild) dropContent.removeChild(dropContent.firstChild);
  const spin = document.createElement("div");
  spin.className = "loading-spinner";
  const p = document.createElement("p");
  p.textContent = "Loading AI engine...";
  const hint = document.createElement("span");
  hint.className = "hint";
  hint.textContent = "First launch takes ~15-30 seconds";
  dropContent.append(spin, p, hint);
  dropZone.classList.add("loading");
}

function setDropZoneReady() {
  if (!dropContent) return;
  while (dropContent.firstChild) dropContent.removeChild(dropContent.firstChild);
  for (const node of savedDropChildren) dropContent.appendChild(node);
  dropZone.classList.remove("loading");
}

setDropZoneLoading();
statusEl.textContent = "Starting AI engine...";
statusEl.className = "";

window.api.onBackendStatus((s) => {
  if (s === "ready") {
    backendReady = true;
    statusEl.textContent = "Ready";
    statusEl.className = "ready";
    setDropZoneReady();
  } else if (s === "error") {
    statusEl.textContent = "Backend failed to start — check backend.log";
    statusEl.className = "error";
  }
});

// Settings panel toggle
settingsToggle.addEventListener("click", () => {
  settingsPanel.classList.toggle("visible");
  settingsToggle.classList.toggle("active");
});

document.getElementById("settings-close").addEventListener("click", () => {
  settingsPanel.classList.remove("visible");
  settingsToggle.classList.remove("active");
});

// Alpha matting toggle
alphaMatting.addEventListener("change", () => {
  if (alphaMatting.checked) {
    mattingOptions.classList.remove("hidden");
  } else {
    mattingOptions.classList.add("hidden");
  }
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});

// Slider value displays
fgThreshold.addEventListener("input", () => {
  fgVal.textContent = fgThreshold.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
bgThreshold.addEventListener("input", () => {
  bgVal.textContent = bgThreshold.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
erodeSize.addEventListener("input", () => {
  erodeVal.textContent = erodeSize.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
modelSelect.addEventListener("change", () => {
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});

// Color removal controls
colorRemove.addEventListener("change", () => {
  if (colorRemove.checked) {
    colorOptions.classList.remove("hidden");
  } else {
    colorOptions.classList.add("hidden");
  }
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
colorPicker.addEventListener("input", () => {
  colorHex.textContent = colorPicker.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
colorTolerance.addEventListener("input", () => {
  tolVal.textContent = colorTolerance.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});

// Edge smoothing controls
edgeSmooth.addEventListener("change", () => {
  edgeOptions.classList.toggle("hidden", !edgeSmooth.checked);
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
edgeStrength.addEventListener("input", () => {
  edgeStrengthVal.textContent = edgeStrength.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
edgeTrim.addEventListener("input", () => {
  edgeTrimVal.textContent = edgeTrim.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});

// Watermark removal controls
watermarkRemove.addEventListener("change", () => {
  watermarkOptions.classList.toggle("hidden", !watermarkRemove.checked);
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
watermarkSize.addEventListener("input", () => {
  wmSizeVal.textContent = watermarkSize.value;
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});
watermarkPosition.addEventListener("change", () => {
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});

// Auto trim control
autoTrimCheck.addEventListener("change", () => {
  if (originalBytes) reprocessBtn.classList.remove("hidden");
});

// Get current settings
function getSettings(opts = {}) {
  const allColors = [...pickedColors];
  if (colorRemove.checked && pickedColors.length === 0) {
    const hex = colorPicker.value;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    allColors.push([r, g, b]);
  }
  return {
    model: modelSelect.value,
    alphaMatting: alphaMatting.checked,
    fgThreshold: parseInt(fgThreshold.value),
    bgThreshold: parseInt(bgThreshold.value),
    erodeSize: parseInt(erodeSize.value),
    colorRemove: colorRemove.checked || pickedColors.length > 0,
    colors: allColors,
    colorTolerance: parseInt(colorTolerance.value),
    edgeSmooth: edgeSmooth.checked,
    edgeStrength: parseInt(edgeStrength.value),
    edgeTrim: parseInt(edgeTrim.value),
    watermarkRemove: watermarkRemove.checked,
    watermarkPosition: watermarkPosition.value,
    watermarkSize: parseInt(watermarkSize.value),
    autoTrim: autoTrimCheck.checked,
    skipBg: opts.skipBg || false,
  };
}

// Helper: check if file is an image
function isImageFile(file) {
  const ext = file.name.toLowerCase().split(".").pop();
  const imageExts = ["png", "jpg", "jpeg", "webp", "bmp", "tiff", "gif"];
  return file.type.startsWith("image/") || imageExts.includes(ext);
}

// Drop zone events
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove("dragover");
  if (!backendReady) {
    statusEl.textContent = "AI engine still loading — wait for Ready";
    statusEl.className = "error";
    return;
  }
  const files = [...e.dataTransfer.files].filter(isImageFile);
  if (files.length === 0) return;

  if (files.length === 1) {
    processFile(files[0]);
  } else {
    startBatch(files);
  }
});

fileInput.addEventListener("change", (e) => {
  if (!backendReady) {
    statusEl.textContent = "AI engine still loading — wait for Ready";
    statusEl.className = "error";
    fileInput.value = "";
    return;
  }
  const files = [...e.target.files].filter(isImageFile);
  if (files.length === 0) return;

  if (files.length === 1) {
    processFile(files[0]);
  } else {
    startBatch(files);
  }
  fileInput.value = "";
});

// Mode toggle
function setMode(mode) {
  editOnlyMode = mode === "edit";
  splitMode = mode === "split";
  document.getElementById("mode-remove").classList.toggle("active", mode === "remove");
  document.getElementById("mode-split").classList.toggle("active", mode === "split");
  document.getElementById("mode-edit").classList.toggle("active", mode === "edit");
}
document.getElementById("mode-remove").addEventListener("click", () => setMode("remove"));
document.getElementById("mode-split").addEventListener("click", () => setMode("split"));
document.getElementById("mode-edit").addEventListener("click", () => setMode("edit"));

// Sparkle particle system
function emitParticles(containerEl, durationMs) {
  const ctx = particlesCanvas.getContext("2d");
  const rect = containerEl.getBoundingClientRect();
  particlesCanvas.width = rect.width;
  particlesCanvas.height = rect.height;

  const particles = [];
  const startTime = performance.now();

  function spawnParticle(x) {
    const count = 3 + Math.floor(Math.random() * 4);
    for (let i = 0; i < count; i++) {
      particles.push({
        x: x,
        y: Math.random() * rect.height,
        vx: (Math.random() - 0.3) * 3,
        vy: (Math.random() - 0.5) * 4,
        size: 1 + Math.random() * 3,
        life: 1,
        decay: 0.01 + Math.random() * 0.03,
        color: Math.random() > 0.5 ? "#e94560" : "#ff8fa3",
      });
    }
  }

  function animate(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / durationMs, 1);

    ctx.clearRect(0, 0, particlesCanvas.width, particlesCanvas.height);

    if (progress < 0.95) {
      spawnParticle(progress * rect.width);
    }

    for (let i = particles.length - 1; i >= 0; i--) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.life -= p.decay;
      p.vy += 0.05;

      if (p.life <= 0) {
        particles.splice(i, 1);
        continue;
      }

      ctx.globalAlpha = p.life;
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;

    if (progress < 1 || particles.length > 0) {
      requestAnimationFrame(animate);
    } else {
      ctx.clearRect(0, 0, particlesCanvas.width, particlesCanvas.height);
    }
  }

  requestAnimationFrame(animate);
}

// Reveal animation
function playReveal() {
  const container = document.getElementById("reveal-container");
  const animDuration = 1200;

  beforeImg.classList.remove("reveal");
  wipeLine.classList.remove("animate");

  void beforeImg.offsetWidth;

  afterImg.style.display = "block";
  beforeImg.classList.add("reveal");
  wipeLine.classList.add("animate");
  emitParticles(container, animDuration);
}

async function processFile(file) {
  originalName = file.name.replace(/\.[^.]+$/, "");

  const buffer = await file.arrayBuffer();
  originalBytes = new Uint8Array(buffer);

  dropZone.classList.add("hidden");

  if (splitMode) {
    // Split sprites mode
    preview.classList.add("hidden");
    actions.classList.add("hidden");
    spritePanel.classList.remove("hidden");
    spriteGrid.innerHTML = "";
    spriteBlobs = [];
    spriteCount.textContent = "...";
    spriteSaveBtn.classList.add("hidden");

    statusEl.textContent = "Splitting sprites...";
    statusEl.className = "";

    try {
      const settings = getSettings();
      const result = await window.api.splitSprites(buffer, settings);
      spriteBlobs = result.sprites.map(b64 => {
        const bin = atob(b64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        return new Blob([arr], { type: "image/png" });
      });

      spriteCount.textContent = spriteBlobs.length;
      spriteGrid.innerHTML = "";
      spriteBlobs.forEach((blob, i) => {
        const url = URL.createObjectURL(blob);
        const card = document.createElement("div");
        card.className = "sprite-card";
        card.innerHTML = `<img src="${url}" /><span>${originalName}_${i + 1}.png</span>`;
        spriteGrid.appendChild(card);
      });

      spriteSaveBtn.classList.remove("hidden");
      statusEl.textContent = `Found ${spriteBlobs.length} sprites`;
      statusEl.className = "ready";
    } catch (err) {
      statusEl.textContent = "Split failed — " + err.message;
      statusEl.className = "error";
    }
    return;
  }

  preview.classList.remove("hidden");
  actions.classList.add("hidden");

  const settings = getSettings();
  const needsBackend = !editOnlyMode || settings.watermarkRemove;

  if (editOnlyMode && !needsBackend) {
    const url = URL.createObjectURL(file);
    beforeImg.src = url;
    afterImg.src = url;
    afterImg.style.display = "block";
    beforeImg.style.clipPath = "inset(0 0% 0 100%)";
    await new Promise((resolve, reject) => {
      afterImg.onload = resolve;
      afterImg.onerror = () => reject(new Error("Image failed to load"));
    });
    resultBlob = new Blob([originalBytes], { type: file.type });
    actions.classList.remove("hidden");
    statusEl.textContent = "Edit mode";
    statusEl.className = "ready";
  } else {
    beforeImg.src = URL.createObjectURL(file);
    await runRemoval(originalBytes.buffer, true);
  }
}

function getSpinnerText(settings) {
  const parts = [];
  if (settings.watermarkRemove) parts.push("watermark");
  if (!settings.skipBg) parts.push("background");
  if (parts.length === 0) return "Processing...";
  return "Removing " + parts.join(" & ") + "...";
}

async function runRemoval(arrayBuffer, animate = true) {
  const settings = getSettings({ skipBg: editOnlyMode });

  // Update spinner text
  spinnerText.textContent = getSpinnerText(settings);
  spinner.classList.add("visible");
  statusEl.textContent = spinnerText.textContent;
  statusEl.className = "";
  reprocessBtn.classList.add("hidden");

  if (animate) {
    afterImg.style.display = "none";
    beforeImg.classList.remove("reveal");
    beforeImg.style.clipPath = "";
    wipeLine.classList.remove("animate");
  }

  try {
    const resultBuffer = await window.api.removeBg(arrayBuffer, settings);
    resultBlob = new Blob([resultBuffer], { type: "image/png" });

    const newUrl = URL.createObjectURL(resultBlob);
    afterImg.src = newUrl;
    await new Promise((resolve, reject) => {
      afterImg.onload = resolve;
      afterImg.onerror = () => reject(new Error("Result image failed to load"));
    });

    spinner.classList.remove("visible");

    if (animate) {
      playReveal();
      setTimeout(() => {
        actions.classList.remove("hidden");
        statusEl.textContent = "Done!";
        statusEl.className = "ready";
      }, 1300);
    } else {
      afterImg.style.display = "block";
      beforeImg.style.clipPath = "inset(0 0% 0 100%)";
      actions.classList.remove("hidden");
      statusEl.textContent = "Done!";
      statusEl.className = "ready";
    }
  } catch (err) {
    spinner.classList.remove("visible");
    statusEl.textContent = "Processing failed — " + err.message;
    statusEl.className = "error";
  }
}

// Reprocess with new settings
reprocessBtn.addEventListener("click", async () => {
  if (!originalBytes) return;
  const copy = originalBytes.slice(0).buffer;
  await runRemoval(copy, false);
});

// ===================== BATCH PROCESSING =====================

async function startBatch(files) {
  // Ask for output directory first
  const outDir = await window.api.selectDirectory();
  if (!outDir) return; // User cancelled

  batchCancelled = false;

  // Enter batch mode UI
  dropZone.classList.add("hidden");
  preview.classList.add("hidden");
  actions.classList.add("hidden");
  batchPanel.classList.remove("hidden");
  batchOpenBtn.classList.add("hidden");
  batchDoneBtn.classList.add("hidden");
  batchCancelBtn.classList.remove("hidden");

  // Build file list UI
  batchList.innerHTML = "";
  const items = files.map((file, i) => {
    const row = document.createElement("div");
    row.className = "batch-item";
    row.innerHTML = `
      <span class="batch-icon pending">&#9675;</span>
      <span class="batch-name">${file.name}</span>
      <span class="batch-status">Pending</span>
    `;
    batchList.appendChild(row);
    return { file, row };
  });

  let completed = 0;
  let failed = 0;

  batchCounter.textContent = `0 / ${files.length}`;
  batchProgressFill.style.width = "0%";

  statusEl.textContent = `Batch: 0 / ${files.length}`;
  statusEl.className = "";

  for (let i = 0; i < items.length; i++) {
    if (batchCancelled) break;

    const { file, row } = items[i];
    const icon = row.querySelector(".batch-icon");
    const status = row.querySelector(".batch-status");

    // Mark as processing
    icon.innerHTML = "&#8635;";
    icon.className = "batch-icon processing";
    status.textContent = "Processing...";
    row.scrollIntoView({ behavior: "smooth", block: "nearest" });

    try {
      const buffer = await file.arrayBuffer();
      const settings = getSettings();
      // Batch always does full pipeline (no skipBg)
      const resultBuffer = await window.api.removeBg(buffer, settings);

      // Save to output directory
      const outName = file.name.replace(/\.[^.]+$/, "") + "-nobg.png";
      const outPath = outDir.replace(/\\/g, "/") + "/" + outName;
      await window.api.saveToPath(resultBuffer, outPath);

      icon.innerHTML = "&#10003;";
      icon.className = "batch-icon done";
      status.textContent = "Done";
      completed++;
    } catch (err) {
      icon.innerHTML = "&#10007;";
      icon.className = "batch-icon error";
      status.textContent = err.message || "Failed";
      failed++;
    }

    const total = completed + failed;
    batchCounter.textContent = `${total} / ${files.length}`;
    batchProgressFill.style.width = `${(total / files.length) * 100}%`;
    statusEl.textContent = `Batch: ${total} / ${files.length}`;
  }

  // Done
  batchCancelBtn.classList.add("hidden");
  batchOpenBtn.classList.remove("hidden");
  batchDoneBtn.classList.remove("hidden");

  if (batchCancelled) {
    statusEl.textContent = `Batch cancelled — ${completed} done, ${failed} failed`;
  } else {
    statusEl.textContent = `Batch complete — ${completed} done${failed ? `, ${failed} failed` : ""}`;
  }
  statusEl.className = failed ? "error" : "ready";

  // Open folder button
  batchOpenBtn.onclick = () => window.api.openPath(outDir);

  // New batch button
  batchDoneBtn.onclick = () => {
    batchPanel.classList.add("hidden");
    dropZone.classList.remove("hidden");
    statusEl.textContent = "Ready";
    statusEl.className = "ready";
  };
}

batchCancelBtn.addEventListener("click", () => {
  batchCancelled = true;
  batchCancelBtn.textContent = "Cancelling...";
  batchCancelBtn.disabled = true;
});

// ===================== SPRITE SPLIT =====================

spriteSaveBtn.addEventListener("click", async () => {
  if (spriteBlobs.length === 0) return;
  const outDir = await window.api.selectDirectory();
  if (!outDir) return;

  for (let i = 0; i < spriteBlobs.length; i++) {
    const buf = await spriteBlobs[i].arrayBuffer();
    const name = `${originalName}_${i + 1}.png`;
    const outPath = outDir.replace(/\\/g, "/") + "/" + name;
    await window.api.saveToPath(buf, outPath);
  }
  statusEl.textContent = `Saved ${spriteBlobs.length} sprites`;
  statusEl.className = "ready";
});

spriteResetBtn.addEventListener("click", () => {
  spritePanel.classList.add("hidden");
  dropZone.classList.remove("hidden");
  spriteBlobs = [];
  spriteGrid.innerHTML = "";
  originalBytes = null;
  statusEl.textContent = "Ready";
  statusEl.className = "ready";
});

// ===================== EYEDROPPER =====================

const eyedropperBtn = document.getElementById("eyedropper-btn");
const revealContainer = document.getElementById("reveal-container");
const tooltip = document.getElementById("color-preview-tooltip");
let eyedropperActive = false;

eyedropperBtn.addEventListener("click", () => {
  eyedropperActive = !eyedropperActive;
  eyedropperBtn.classList.toggle("active", eyedropperActive);
  revealContainer.classList.toggle("eyedropper-mode", eyedropperActive);
  if (!eyedropperActive) tooltip.style.display = "none";
  if (eyedropperActive && eraserActive) {
    eraserActive = false;
    eraserBtn.classList.remove("active");
    eraserCanvas.classList.remove("active");
    revealContainer.classList.remove("eraser-mode");
    brushCursor.style.display = "none";
    commitEraser();
  }
});

function getPixelColor(e) {
  const img = afterImg;
  const rect = img.getBoundingClientRect();
  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const x = Math.floor((e.clientX - rect.left) * scaleX);
  const y = Math.floor((e.clientY - rect.top) * scaleY);

  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const pixel = ctx.getImageData(x, y, 1, 1).data;
  return { r: pixel[0], g: pixel[1], b: pixel[2], a: pixel[3] };
}

function rgbToHex(r, g, b) {
  return "#" + [r, g, b].map(v => v.toString(16).padStart(2, "0")).join("");
}

function renderPickedColors() {
  const list = document.getElementById("picked-colors-list");
  const container = document.getElementById("picked-colors");
  const countEl = document.getElementById("picked-colors-count");
  list.innerHTML = "";
  pickedColors.forEach((c, i) => {
    const hex = rgbToHex(c[0], c[1], c[2]);
    const item = document.createElement("div");
    item.className = "picked-color-item";
    item.innerHTML = `<div class="swatch" style="background:${hex}"></div><span>${hex}</span><button class="remove-color-btn" data-index="${i}">&times;</button>`;
    list.appendChild(item);
  });
  countEl.textContent = pickedColors.length;
  container.classList.toggle("hidden", pickedColors.length === 0);
}

document.getElementById("picked-colors-toggle").addEventListener("click", () => {
  const container = document.getElementById("picked-colors");
  const list = document.getElementById("picked-colors-list");
  container.classList.toggle("expanded");
  list.classList.toggle("collapsed");
});

document.addEventListener("click", async (e) => {
  if (!e.target.classList.contains("remove-color-btn")) return;
  const idx = parseInt(e.target.dataset.index);
  pickedColors.splice(idx, 1);
  renderPickedColors();
  if (originalBytes) {
    const copy = originalBytes.slice(0).buffer;
    await runRemoval(copy, false);
  }
});

afterImg.addEventListener("mousemove", (e) => {
  if (!eyedropperActive) return;
  const color = getPixelColor(e);
  const hex = rgbToHex(color.r, color.g, color.b);
  tooltip.querySelector(".swatch").style.background = hex;
  tooltip.querySelector("span").textContent = hex;
  tooltip.style.display = "flex";
  tooltip.style.left = (e.clientX + 16) + "px";
  tooltip.style.top = (e.clientY - 10) + "px";
});

afterImg.addEventListener("mouseleave", () => {
  tooltip.style.display = "none";
});

afterImg.addEventListener("click", async (e) => {
  if (!eyedropperActive || !originalBytes) return;

  const color = getPixelColor(e);
  pickedColors.push([color.r, color.g, color.b]);
  renderPickedColors();

  const copy = originalBytes.slice(0).buffer;
  await runRemoval(copy, false);
});

// ===================== ERASER =====================

const eraserBtn = document.getElementById("eraser-btn");
const eraserCanvas = document.getElementById("eraser-canvas");
const brushCursor = document.getElementById("brush-cursor");
let eraserActive = false;
let eraserSize = 20;
let isErasing = false;
let eraserCtx = null;

let editCanvas = null;
let editCtx = null;
let undoStack = [];
let redoStack = [];
const MAX_UNDO = 30;

function initEditCanvas() {
  editCanvas = document.createElement("canvas");
  editCanvas.width = afterImg.naturalWidth;
  editCanvas.height = afterImg.naturalHeight;
  editCtx = editCanvas.getContext("2d");
  editCtx.drawImage(afterImg, 0, 0);
  undoStack = [];
  redoStack = [];
}

function saveUndoState() {
  undoStack.push(editCtx.getImageData(0, 0, editCanvas.width, editCanvas.height));
  if (undoStack.length > MAX_UNDO) undoStack.shift();
  redoStack = [];
}

function undo() {
  if (!editCtx || undoStack.length === 0) return;
  redoStack.push(editCtx.getImageData(0, 0, editCanvas.width, editCanvas.height));
  const state = undoStack.pop();
  editCtx.putImageData(state, 0, 0);
  afterImg.src = editCanvas.toDataURL("image/png");
  commitEraser();
  statusEl.textContent = `Undo (${undoStack.length} left)`;
}

function redo() {
  if (!editCtx || redoStack.length === 0) return;
  undoStack.push(editCtx.getImageData(0, 0, editCanvas.width, editCanvas.height));
  const state = redoStack.pop();
  editCtx.putImageData(state, 0, 0);
  afterImg.src = editCanvas.toDataURL("image/png");
  commitEraser();
  statusEl.textContent = `Redo (${redoStack.length} left)`;
}

function updateBrushCursor(e) {
  if (!eraserActive) return;
  const rect = afterImg.getBoundingClientRect();
  const displayRadius = (eraserSize / afterImg.naturalWidth) * rect.width;
  brushCursor.style.width = (displayRadius * 2) + "px";
  brushCursor.style.height = (displayRadius * 2) + "px";
  brushCursor.style.left = e.clientX + "px";
  brushCursor.style.top = e.clientY + "px";
  brushCursor.style.display = "block";
}

function eraseAt(e) {
  if (!editCtx) return;
  const rect = afterImg.getBoundingClientRect();
  const scaleX = afterImg.naturalWidth / rect.width;
  const scaleY = afterImg.naturalHeight / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  editCtx.globalCompositeOperation = "destination-out";
  editCtx.beginPath();
  editCtx.arc(x, y, eraserSize, 0, Math.PI * 2);
  editCtx.fill();
  editCtx.globalCompositeOperation = "source-over";

  afterImg.src = editCanvas.toDataURL("image/png");
}

function commitEraser() {
  if (!editCanvas) return;
  editCanvas.toBlob((blob) => {
    resultBlob = blob;
  }, "image/png");
}

eraserBtn.addEventListener("click", () => {
  eraserActive = !eraserActive;
  eraserBtn.classList.toggle("active", eraserActive);
  eraserCanvas.classList.toggle("active", eraserActive);
  revealContainer.classList.toggle("eraser-mode", eraserActive);

  if (eraserActive && eyedropperActive) {
    eyedropperActive = false;
    eyedropperBtn.classList.remove("active");
    revealContainer.classList.remove("eyedropper-mode");
    tooltip.style.display = "none";
  }

  if (eraserActive) {
    initEditCanvas();
  } else {
    brushCursor.style.display = "none";
    commitEraser();
  }
});

eraserCanvas.addEventListener("mousedown", (e) => {
  if (!eraserActive || e.button !== 0) return;
  saveUndoState();
  isErasing = true;
  eraseAt(e);
});

eraserCanvas.addEventListener("mousemove", (e) => {
  updateBrushCursor(e);
  if (isErasing) eraseAt(e);
});

eraserCanvas.addEventListener("mouseup", () => {
  if (isErasing) {
    isErasing = false;
    commitEraser();
  }
});

eraserCanvas.addEventListener("mouseleave", () => {
  brushCursor.style.display = "none";
  if (isErasing) {
    isErasing = false;
    commitEraser();
  }
});

// Keyboard shortcuts
document.addEventListener("keydown", (e) => {
  if (eraserActive) {
    if (e.key === "[") {
      eraserSize = Math.max(2, eraserSize - 3);
      statusEl.textContent = `Brush: ${eraserSize}px`;
    } else if (e.key === "]") {
      eraserSize = Math.min(200, eraserSize + 3);
      statusEl.textContent = `Brush: ${eraserSize}px`;
    }
  }
  if ((e.ctrlKey || e.metaKey) && e.key === "z" && !e.shiftKey) {
    e.preventDefault();
    undo();
  } else if ((e.ctrlKey || e.metaKey) && (e.key === "y" || (e.key === "z" && e.shiftKey))) {
    e.preventDefault();
    redo();
  }
});

// ===================== ZOOM & PAN =====================

let zoomLevel = 1;
let panX = 0, panY = 0;
let isPanning = false;
let panStartX, panStartY;

function updateTransform() {
  revealContainer.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
  revealContainer.style.transformOrigin = "center center";
}

revealContainer.addEventListener("wheel", (e) => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? -0.15 : 0.15;
  zoomLevel = Math.max(0.5, Math.min(5, zoomLevel + delta));
  updateTransform();
});

revealContainer.addEventListener("mousedown", (e) => {
  if (e.button !== 1) return;
  e.preventDefault();
  isPanning = true;
  panStartX = e.clientX - panX;
  panStartY = e.clientY - panY;
  revealContainer.style.cursor = "grabbing";
});

document.addEventListener("mousemove", (e) => {
  if (!isPanning) return;
  panX = e.clientX - panStartX;
  panY = e.clientY - panStartY;
  updateTransform();
});

document.addEventListener("mouseup", (e) => {
  if (e.button !== 1) return;
  isPanning = false;
  revealContainer.style.cursor = "";
});

revealContainer.addEventListener("dblclick", (e) => {
  if (eyedropperActive) return;
  zoomLevel = 1;
  panX = 0;
  panY = 0;
  revealContainer.style.transform = "";
});

// ===================== TRIM =====================

document.getElementById("trim-btn").addEventListener("click", () => {
  if (!afterImg.naturalWidth) return;

  const canvas = document.createElement("canvas");
  canvas.width = afterImg.naturalWidth;
  canvas.height = afterImg.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(afterImg, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const { data, width, height } = imageData;

  let top = height, left = width, bottom = 0, right = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const alpha = data[(y * width + x) * 4 + 3];
      if (alpha > 0) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }

  if (bottom <= top || right <= left) return;

  top = Math.max(0, top - 1);
  left = Math.max(0, left - 1);
  bottom = Math.min(height - 1, bottom + 1);
  right = Math.min(width - 1, right + 1);

  const w = right - left + 1;
  const h = bottom - top + 1;
  const trimmed = ctx.getImageData(left, top, w, h);

  const outCanvas = document.createElement("canvas");
  outCanvas.width = w;
  outCanvas.height = h;
  outCanvas.getContext("2d").putImageData(trimmed, 0, 0);

  outCanvas.toBlob((blob) => {
    resultBlob = blob;
    const url = URL.createObjectURL(blob);
    afterImg.src = url;
    if (editCanvas) {
      editCanvas.width = w;
      editCanvas.height = h;
      editCtx = editCanvas.getContext("2d");
      editCtx.putImageData(trimmed, 0, 0);
      undoStack = [];
      redoStack = [];
    }
    statusEl.textContent = `Trimmed: ${canvas.width}x${canvas.height} → ${w}x${h}`;
    statusEl.className = "ready";
  }, "image/png");
});

// ===================== RESIZE =====================

const resizeBtn = document.getElementById("resize-btn");
const resizePanel = document.getElementById("resize-panel");
const resizeW = document.getElementById("resize-w");
const resizeH = document.getElementById("resize-h");
const resizeLock = document.getElementById("resize-lock");
const resizeApply = document.getElementById("resize-apply");
const resizeInfo = document.getElementById("resize-info");
let aspectLocked = true;
let originalWidth = 0, originalHeight = 0;

resizeBtn.addEventListener("click", () => {
  const showing = resizePanel.classList.toggle("hidden");
  resizeBtn.classList.toggle("active", !resizePanel.classList.contains("hidden"));
  if (!resizePanel.classList.contains("hidden") && afterImg.naturalWidth) {
    originalWidth = afterImg.naturalWidth;
    originalHeight = afterImg.naturalHeight;
    resizeW.value = originalWidth;
    resizeH.value = originalHeight;
    resizeInfo.textContent = `Original: ${originalWidth} x ${originalHeight}`;
  }
});

resizeLock.addEventListener("click", () => {
  aspectLocked = !aspectLocked;
  resizeLock.classList.toggle("active", aspectLocked);
});

resizeW.addEventListener("input", () => {
  if (aspectLocked && originalWidth) {
    const ratio = originalHeight / originalWidth;
    resizeH.value = Math.round(parseInt(resizeW.value) * ratio) || "";
  }
});

resizeH.addEventListener("input", () => {
  if (aspectLocked && originalHeight) {
    const ratio = originalWidth / originalHeight;
    resizeW.value = Math.round(parseInt(resizeH.value) * ratio) || "";
  }
});

resizeApply.addEventListener("click", () => {
  const w = parseInt(resizeW.value);
  const h = parseInt(resizeH.value);
  if (!w || !h || w < 1 || h < 1) return;

  const srcCanvas = document.createElement("canvas");
  srcCanvas.width = afterImg.naturalWidth;
  srcCanvas.height = afterImg.naturalHeight;
  const srcCtx = srcCanvas.getContext("2d");
  srcCtx.drawImage(afterImg, 0, 0);

  const dstCanvas = document.createElement("canvas");
  dstCanvas.width = w;
  dstCanvas.height = h;
  const dstCtx = dstCanvas.getContext("2d");
  dstCtx.imageSmoothingEnabled = true;
  dstCtx.imageSmoothingQuality = "high";
  dstCtx.drawImage(srcCanvas, 0, 0, w, h);

  dstCanvas.toBlob((blob) => {
    resultBlob = blob;
    const url = URL.createObjectURL(blob);
    afterImg.src = url;
    if (editCanvas) {
      editCanvas.width = w;
      editCanvas.height = h;
      editCtx = editCanvas.getContext("2d");
      editCtx.drawImage(dstCanvas, 0, 0);
      undoStack = [];
      redoStack = [];
    }
    originalWidth = w;
    originalHeight = h;
    resizeInfo.textContent = `Resized to ${w} x ${h}`;
    statusEl.textContent = `Resized to ${w} x ${h}`;
    statusEl.className = "ready";
  }, "image/png");
});

// ===================== DOWNLOAD & RESET =====================

downloadBtn.addEventListener("click", async () => {
  if (!resultBlob) return;
  const buffer = await resultBlob.arrayBuffer();
  await window.api.saveFile(buffer, `${originalName}-nobg.png`);
});

resetBtn.addEventListener("click", () => {
  preview.classList.add("hidden");
  actions.classList.add("hidden");
  dropZone.classList.remove("hidden");
  beforeImg.src = "";
  afterImg.src = "";
  afterImg.style.display = "none";
  beforeImg.classList.remove("reveal");
  beforeImg.style.clipPath = "";
  wipeLine.classList.remove("animate");
  resultBlob = null;
  originalBytes = null;
  pickedColors = [];
  renderPickedColors();
  reprocessBtn.classList.add("hidden");
  eraserActive = false;
  eraserBtn.classList.remove("active");
  eraserCanvas.classList.remove("active");
  revealContainer.classList.remove("eraser-mode");
  brushCursor.style.display = "none";
  editCanvas = null;
  editCtx = null;
  zoomLevel = 1;
  panX = 0;
  panY = 0;
  revealContainer.style.transform = "";
  spritePanel.classList.add("hidden");
  spriteBlobs = [];
  spriteGrid.innerHTML = "";
  statusEl.textContent = "Ready";
  statusEl.className = "ready";
});
