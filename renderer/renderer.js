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
let trimMode = false;
let batchCancelled = false;
let spriteBlobs = []; // Array of Blobs for split sprites

// Prevent Electron's default file drop behavior
document.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); });
document.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); });

// Backend status
let backendReady = false;
const loadingFloater = document.getElementById("loading-floater");
const loadingFloaterTitle = document.getElementById("loading-floater-title");
const loadingFloaterHint = document.getElementById("loading-floater-hint");
const loadingFloaterClose = document.getElementById("loading-floater-close");

function showLoaderFloater(state, title, hint) {
  loadingFloater.classList.remove("hidden", "ready", "error");
  if (state === "ready") loadingFloater.classList.add("ready");
  else if (state === "error") loadingFloater.classList.add("error");
  loadingFloaterTitle.textContent = title;
  loadingFloaterHint.textContent = hint;
}

function hideLoaderFloater() {
  loadingFloater.classList.add("hidden");
}

loadingFloaterClose.addEventListener("click", hideLoaderFloater);

showLoaderFloater("loading", "Loading AI engine…", "You can set up Generate, add references, or paste an API key while it warms up.");
statusEl.textContent = "Starting AI engine...";
statusEl.className = "";

window.api.onBackendStatus((s) => {
  if (s === "ready") {
    backendReady = true;
    statusEl.textContent = "Ready";
    statusEl.className = "ready";
    showLoaderFloater("ready", "AI engine ready", "Drop an image or pick a mode.");
    setTimeout(hideLoaderFloater, 2200);
  } else if (s === "error") {
    statusEl.textContent = "Backend failed to start — check backend.log";
    statusEl.className = "error";
    showLoaderFloater("error", "AI engine failed to start", "Check backend.log via the gear menu. Generate still works without it.");
  }
});

// Background-mode setting (Start with Windows)
const startWithWindows = document.getElementById("start-with-windows");
if (startWithWindows && window.api.getSettings) {
  window.api.getSettings().then((s) => {
    startWithWindows.checked = !!(s && s.startWithWindows);
  }).catch(() => {});
  startWithWindows.addEventListener("change", () => {
    window.api.setSetting("startWithWindows", startWithWindows.checked).catch(() => {});
  });
}

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
  if (!backendReady && !trimMode) {
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
  if (!backendReady && !trimMode) {
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
  trimMode = mode === "trim";
  document.getElementById("mode-remove").classList.toggle("active", mode === "remove");
  document.getElementById("mode-split").classList.toggle("active", mode === "split");
  document.getElementById("mode-edit").classList.toggle("active", mode === "edit");
  document.getElementById("mode-trim").classList.toggle("active", mode === "trim");
  document.getElementById("mode-generate").classList.toggle("active", mode === "generate");
}
document.getElementById("mode-remove").addEventListener("click", () => setMode("remove"));
document.getElementById("mode-split").addEventListener("click", () => setMode("split"));
document.getElementById("mode-edit").addEventListener("click", () => setMode("edit"));
document.getElementById("mode-trim").addEventListener("click", () => setMode("trim"));

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

async function trimTransparentBlob(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res;
      img.onerror = () => rej(new Error("Image failed to decode"));
      img.src = url;
    });
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const data = ctx.getImageData(0, 0, w, h).data;
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (data[(y * w + x) * 4 + 3] > 0) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    if (maxX < 0) throw new Error("Image is fully transparent");
    minX = Math.max(0, minX - 1);
    minY = Math.max(0, minY - 1);
    maxX = Math.min(w - 1, maxX + 1);
    maxY = Math.min(h - 1, maxY + 1);
    const cw = maxX - minX + 1;
    const ch = maxY - minY + 1;
    const crop = document.createElement("canvas");
    crop.width = cw;
    crop.height = ch;
    crop.getContext("2d").drawImage(canvas, -minX, -minY);
    const blob = await new Promise((res, rej) =>
      crop.toBlob(b => b ? res(b) : rej(new Error("toBlob returned null")), "image/png")
    );
    return { blob, originalSize: [w, h], trimmedSize: [cw, ch] };
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function processFile(file) {
  originalName = file.name.replace(/\.[^.]+$/, "");

  const buffer = await file.arrayBuffer();
  originalBytes = new Uint8Array(buffer);

  dropZone.classList.add("hidden");

  if (trimMode) {
    preview.classList.remove("hidden");
    actions.classList.add("hidden");
    spinner.classList.add("visible");
    spinnerText.textContent = "Trimming...";
    statusEl.textContent = "Trimming transparent edges...";
    statusEl.className = "";
    beforeImg.src = URL.createObjectURL(file);
    try {
      const { blob, originalSize, trimmedSize } = await trimTransparentBlob(file);
      resultBlob = blob;
      const newUrl = URL.createObjectURL(blob);
      afterImg.src = newUrl;
      await new Promise((resolve, reject) => {
        afterImg.onload = resolve;
        afterImg.onerror = () => reject(new Error("Trimmed image failed to load"));
      });
      spinner.classList.remove("visible");
      afterImg.style.display = "block";
      beforeImg.style.clipPath = "inset(0 0% 0 100%)";
      actions.classList.remove("hidden");
      statusEl.textContent = `Trimmed ${originalSize[0]}x${originalSize[1]} -> ${trimmedSize[0]}x${trimmedSize[1]}`;
      statusEl.className = "ready";
    } catch (err) {
      spinner.classList.remove("visible");
      statusEl.textContent = "Trim failed - " + err.message;
      statusEl.className = "error";
    }
    return;
  }

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

    const doneText = `Done — ${afterImg.naturalWidth}x${afterImg.naturalHeight}`;
    if (animate) {
      playReveal();
      setTimeout(() => {
        actions.classList.remove("hidden");
        // After the wipe, show the result at its natural size so trim is visible.
        // Must remove .reveal class to clear the animation's forwards-fill clip-path.
        beforeImg.src = afterImg.src;
        beforeImg.classList.remove("reveal");
        beforeImg.style.clipPath = "none";
        afterImg.style.display = "none";
        wipeLine.classList.remove("animate");
        statusEl.textContent = doneText;
        statusEl.className = "ready";
      }, 1300);
    } else {
      beforeImg.src = afterImg.src;
      beforeImg.classList.remove("reveal");
      beforeImg.style.clipPath = "none";
      afterImg.style.display = "none";
      wipeLine.classList.remove("animate");
      actions.classList.remove("hidden");
      statusEl.textContent = doneText;
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
      let resultBuffer;
      let outSuffix = "-nobg.png";
      if (trimMode) {
        const { blob } = await trimTransparentBlob(file);
        resultBuffer = await blob.arrayBuffer();
        outSuffix = "-trim.png";
      } else {
        const buffer = await file.arrayBuffer();
        const settings = getSettings();
        resultBuffer = await window.api.removeBg(buffer, settings);
      }

      const outName = file.name.replace(/\.[^.]+$/, "") + outSuffix;
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
  const genSectionEl = document.getElementById("gen-section");
  if (genSectionEl) genSectionEl.classList.add("hidden");
  document.getElementById("gen-panel").classList.add("hidden");
  document.getElementById("gen-results-panel").classList.add("hidden");
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

// ===================== GENERATE (tabs + image + video) =====================

const genSection = document.getElementById("gen-section");
const genPanel = document.getElementById("gen-panel");
const genResultsPanel = document.getElementById("gen-results-panel");
const genTabsEl = document.getElementById("gen-tabs");
const genNewTabBtn = document.getElementById("gen-new-tab-btn");
const genPrompt = document.getElementById("gen-prompt");
const genCount = document.getElementById("gen-count");
const genCutoutMode = document.getElementById("gen-cutout-mode");
const genCutoutWrap = document.getElementById("gen-cutout-wrap");
const genModelSelect = document.getElementById("gen-model-select");
const genModelNote = document.getElementById("gen-model-note");
const genVideoControls = document.getElementById("gen-video-controls");
const genLengthEl = document.getElementById("gen-length");
const genAspectEl = document.getElementById("gen-aspect");
const genAudioEl = document.getElementById("gen-audio");
const genCostEl = document.getElementById("gen-cost");
const genRefsThumbs = document.getElementById("gen-refs-thumbs");
const genLibBtn = document.getElementById("gen-lib-btn");
const genQuickRefBtn = document.getElementById("gen-quick-ref-btn");
const genQuickRefInput = document.getElementById("gen-quick-ref-input");
const genGoBtn = document.getElementById("gen-go-btn");
const genCancelBtn = document.getElementById("gen-cancel-btn");
const genBackBtn = document.getElementById("gen-back-btn");
const genStatusEl = document.getElementById("gen-status");
const genResultsGrid = document.getElementById("gen-results-grid");
const genResultsCount = document.getElementById("gen-results-count");
const genResultsTitle = document.getElementById("gen-results-title");
const genSaveAllBtn = document.getElementById("gen-save-all-btn");
const genAgainBtn = document.getElementById("gen-again-btn");
const genResetBtn = document.getElementById("gen-reset-btn");
const modeGenerateBtn = document.getElementById("mode-generate");
const geminiKeyInput = document.getElementById("gemini-key");

// --- Model catalog + pricing (per-image or per-second-of-video) ---
const MODEL_DEFS = [
  { id: "gemini-2.5-flash-image",         label: "Nano Banana (Gemini 2.5 Flash Image)", kind: "image", note: "Default — best for multi-ref edits & consistency" },
  { id: "imagen-4.0-generate-001",        label: "Imagen 4",                kind: "image", note: "Higher fidelity, weaker refs" },
  { id: "imagen-4.0-ultra-generate-001",  label: "Imagen 4 Ultra",          kind: "image", note: "Highest fidelity, 1 per call" },
  { id: "imagen-4.0-fast-generate-001",   label: "Imagen 4 Fast",           kind: "image", note: "Cheapest image option" },
  { id: "veo-3.0-generate-001",           label: "Veo 3",                   kind: "video", note: "Latest, with audio" },
  { id: "veo-3.0-fast-generate-001",      label: "Veo 3 Fast",              kind: "video", note: "Cheaper, faster" },
  { id: "veo-2.0-generate-001",           label: "Veo 2",                   kind: "video", note: "Older, silent" },
];
const MODELS_BY_ID = new Map(MODEL_DEFS.map(m => [m.id, m]));
const PRICING = {
  "gemini-2.5-flash-image":          { per: "image", cost: 0.04 },
  "imagen-4.0-generate-001":         { per: "image", cost: 0.04 },
  "imagen-4.0-ultra-generate-001":   { per: "image", cost: 0.06 },
  "imagen-4.0-fast-generate-001":    { per: "image", cost: 0.02 },
  "veo-3.0-generate-001":            { per: "sec",   cost: 0.40 },
  "veo-3.0-fast-generate-001":       { per: "sec",   cost: 0.10 },
  "veo-2.0-generate-001":            { per: "sec",   cost: 0.50 },
};
function modelKind(id) { return (MODELS_BY_ID.get(id) || {}).kind || "image"; }
function isVideoModel(id) { return modelKind(id) === "video"; }

// --- Tab state ---
// tab: { id, model, prompt, count, cutoutMode, length, aspect, audio, refs[],
//        startFrameIdx, status, jobId, view, results[], hasUnviewedResults, errorMsg }
let tabs = [];
const tabsById = new Map();
let activeTabId = null;
const jobIdToTab = new Map();

function newTab(seed = {}) {
  const tab = {
    id: `tab-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    model: "gemini-2.5-flash-image",
    prompt: "",
    count: 4,
    cutoutMode: "figure",
    length: 8,
    aspect: "16:9",
    audio: true,
    refs: [],
    startFrameIdx: 0,
    status: "idle",        // idle | generating | ready | error
    jobId: null,
    view: "form",          // form | results
    results: [],
    hasUnviewedResults: false,
    errorMsg: "",
    ...seed,
  };
  tabs.push(tab);
  tabsById.set(tab.id, tab);
  return tab;
}

function tabIndex(tab) { return tabs.indexOf(tab); }

function tabDisplayName(tab) {
  const p = (tab.prompt || "").trim();
  if (!p) return `Untitled ${tabIndex(tab) + 1}`;
  const trimmed = p.replace(/\s+/g, " ").slice(0, 28);
  return trimmed + (p.length > 28 ? "…" : "");
}

function activeTab() { return tabsById.get(activeTabId) || null; }

// --- Persistence ---
function persistTabs() {
  const serializable = tabs.map(t => ({
    id: t.id,
    model: t.model,
    prompt: t.prompt,
    count: t.count,
    cutoutMode: t.cutoutMode,
    length: t.length,
    aspect: t.aspect,
    audio: t.audio,
    libraryRefIds: t.refs.filter(r => r.source === "library" && r.id).map(r => r.id),
    startFrameIdx: t.startFrameIdx,
  }));
  window.api.setSetting("genTabs", { activeTabId, tabs: serializable }).catch(() => {});
}

async function restoreTabs() {
  let saved;
  try {
    const s = await window.api.getSettings();
    saved = s && s.genTabs;
  } catch {}
  if (saved && Array.isArray(saved.tabs) && saved.tabs.length) {
    let libRefs = [];
    try { libRefs = await window.api.referencesList(); } catch {}
    const refsById = new Map(libRefs.map(r => [r.id, r]));
    for (const persisted of saved.tabs) {
      const refs = [];
      for (const id of persisted.libraryRefIds || []) {
        const item = refsById.get(id);
        if (!item) continue;
        const bytes = item.data instanceof Uint8Array ? item.data : new Uint8Array(item.data || []);
        refs.push({
          source: "library",
          id: item.id,
          name: item.name,
          mimeType: item.mimeType,
          base64: bytesToBase64(bytes),
        });
      }
      newTab({ ...persisted, refs });
    }
    activeTabId = saved.activeTabId && tabsById.has(saved.activeTabId) ? saved.activeTabId : tabs[0].id;
  } else {
    const first = newTab();
    activeTabId = first.id;
  }
  renderActiveTab();
}

// --- Render: tab strip ---
function renderTabs() {
  genTabsEl.textContent = "";
  for (const tab of tabs) {
    const el = document.createElement("button");
    el.type = "button";
    el.className = "gen-tab";
    if (tab.id === activeTabId) el.classList.add("active");
    if (tab.status === "generating") el.classList.add("generating");
    else if (tab.status === "error") el.classList.add("has-error");
    else if (tab.hasUnviewedResults) el.classList.add("has-results");

    const dot = document.createElement("span");
    dot.className = "gen-tab-dot";
    el.appendChild(dot);

    const label = document.createElement("span");
    label.className = "gen-tab-label";
    label.textContent = tabDisplayName(tab);
    el.appendChild(label);

    if (tabs.length > 1) {
      const close = document.createElement("button");
      close.type = "button";
      close.className = "gen-tab-close";
      close.textContent = "×";
      close.title = "Close tab";
      close.addEventListener("click", (e) => {
        e.stopPropagation();
        closeTab(tab.id);
      });
      el.appendChild(close);
    }

    el.addEventListener("click", () => switchTab(tab.id));
    genTabsEl.appendChild(el);
  }
}

function switchTab(id) {
  if (id === activeTabId) return;
  saveFormIntoActiveTab();
  activeTabId = id;
  const tab = activeTab();
  if (tab) tab.hasUnviewedResults = false;
  renderActiveTab();
  persistTabs();
}

function closeTab(id) {
  const tab = tabsById.get(id);
  if (!tab) return;
  if (tab.status === "generating") {
    if (!confirm(`Cancel running job in "${tabDisplayName(tab)}"?`)) return;
    if (tab.jobId) {
      window.api.geminiJobCancel(tab.jobId).catch(() => {});
      jobIdToTab.delete(tab.jobId);
    }
  }
  const idx = tabs.indexOf(tab);
  tabs.splice(idx, 1);
  tabsById.delete(id);
  if (id === activeTabId) {
    if (tabs.length === 0) {
      const fresh = newTab();
      activeTabId = fresh.id;
    } else {
      activeTabId = tabs[Math.min(idx, tabs.length - 1)].id;
    }
  }
  renderActiveTab();
  persistTabs();
}

function saveFormIntoActiveTab() {
  const tab = activeTab();
  if (!tab) return;
  tab.prompt = genPrompt.value;
  tab.count = Math.max(1, Math.min(12, parseInt(genCount.value) || 1));
  tab.cutoutMode = genCutoutMode.value;
  tab.model = genModelSelect.value;
  tab.length = parseInt(genLengthEl.value) || 8;
  tab.aspect = genAspectEl.value;
  tab.audio = genAudioEl.checked;
}

// --- Render: form + results panels ---
function renderActiveTab() {
  renderTabs();
  const tab = activeTab();
  if (!tab) return;
  renderFormForTab(tab);
  if (tab.view === "results") {
    genPanel.classList.add("hidden");
    genResultsPanel.classList.remove("hidden");
    renderResultsForTab(tab);
  } else {
    genResultsPanel.classList.add("hidden");
    genPanel.classList.remove("hidden");
  }
}

function renderFormForTab(tab) {
  genPrompt.value = tab.prompt;
  genCount.value = tab.count;
  genCutoutMode.value = tab.cutoutMode;
  genModelSelect.value = tab.model;
  genLengthEl.value = String(tab.length);
  genAspectEl.value = tab.aspect;
  genAudioEl.checked = tab.audio;

  const isVideo = isVideoModel(tab.model);
  genVideoControls.classList.toggle("hidden", !isVideo);
  genCutoutWrap.classList.toggle("hidden", isVideo);

  const def = MODELS_BY_ID.get(tab.model);
  if (genModelNote) genModelNote.textContent = def ? def.note : "";

  // Veo 2 doesn't generate audio — disable the audio checkbox visually
  const isVeo2 = tab.model === "veo-2.0-generate-001";
  if (genAudioEl) {
    genAudioEl.disabled = isVeo2;
    if (isVeo2) genAudioEl.checked = false;
  }

  renderRefsForTab(tab);
  updateCostEstimate();

  if (tab.status === "generating") {
    genGoBtn.disabled = true;
    genGoBtn.textContent = "Generating…";
    genStatusEl.textContent = "Running — switch tabs freely";
    genStatusEl.className = "";
  } else if (tab.status === "error") {
    genGoBtn.disabled = false;
    genGoBtn.textContent = "Generate";
    genStatusEl.textContent = tab.errorMsg;
    genStatusEl.className = "error";
  } else {
    genGoBtn.disabled = false;
    genGoBtn.textContent = "Generate";
    genStatusEl.textContent = "";
    genStatusEl.className = "";
  }
}

function renderRefsForTab(tab) {
  genRefsThumbs.textContent = "";
  const isVideo = isVideoModel(tab.model);
  tab.refs.forEach((ref, idx) => {
    const pill = document.createElement("div");
    pill.className = "gen-ref-pill";
    if (ref.source === "library") pill.classList.add("is-saved");
    if (isVideo) {
      pill.classList.add("video-mode");
      if (idx === tab.startFrameIdx) pill.classList.add("start-frame");
      else pill.classList.add("inactive-ref");
      pill.title = idx === tab.startFrameIdx
        ? "Start frame — used as first frame of the video"
        : "Click to use as start frame";
    } else {
      pill.title = ref.name || (ref.source === "library" ? "Library reference" : "Quick reference");
    }
    const img = document.createElement("img");
    img.src = `data:${ref.mimeType};base64,${ref.base64}`;
    pill.appendChild(img);

    if (isVideo && idx === tab.startFrameIdx) {
      const badge = document.createElement("span");
      badge.className = "gen-ref-start-badge";
      badge.textContent = "START";
      pill.appendChild(badge);
    }

    const remove = document.createElement("button");
    remove.className = "gen-ref-pill-remove";
    remove.textContent = "×";
    remove.title = "Remove from this tab";
    remove.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      tab.refs.splice(idx, 1);
      if (tab.startFrameIdx >= tab.refs.length) tab.startFrameIdx = Math.max(0, tab.refs.length - 1);
      renderRefsForTab(tab);
      persistTabs();
    });
    pill.appendChild(remove);

    if (isVideo) {
      pill.addEventListener("click", (e) => {
        if (e.target === remove) return;
        tab.startFrameIdx = idx;
        renderRefsForTab(tab);
        persistTabs();
      });
    }

    genRefsThumbs.appendChild(pill);
  });
}

function updateCostEstimate() {
  const model = genModelSelect.value;
  const count = Math.max(1, Math.min(12, parseInt(genCount.value) || 1));
  const length = parseInt(genLengthEl.value) || 8;
  const p = PRICING[model];
  if (!p) { genCostEl.textContent = ""; genCostEl.classList.remove("expensive"); return; }
  let cost;
  if (p.per === "image") cost = count * p.cost;
  else cost = count * length * p.cost;
  const formatted = cost < 0.01 ? "<$0.01" : `~$${cost.toFixed(2)}`;
  genCostEl.textContent = formatted;
  genCostEl.classList.toggle("expensive", cost >= 1);
}

function renderResultsForTab(tab) {
  const isVideo = isVideoModel(tab.model);
  const total = Math.max(tab.results.length, tab.count);
  let okCount = 0;
  for (const r of tab.results) if (r && r.ok && r.blob) okCount++;
  genResultsCount.textContent = `${okCount} / ${total}`;
  if (genResultsTitle) {
    genResultsTitle.textContent = isVideo ? "Generated Videos" : "Generated";
  }

  // Ensure correct number of cards in the grid
  while (genResultsGrid.children.length < total) {
    const idx = genResultsGrid.children.length;
    const card = makeGenCard(idx, total, isVideo);
    genResultsGrid.appendChild(card);
  }
  while (genResultsGrid.children.length > total) {
    const last = genResultsGrid.lastChild;
    const oldUrl = last.querySelector("img,video")?._currentUrl;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
    last.remove();
  }

  for (let i = 0; i < total; i++) {
    updateCardFromResult(genResultsGrid.children[i], tab.results[i], tab, i);
  }

  // Cancel button visibility — only when actively generating this tab
  if (genCancelBtn) {
    genCancelBtn.classList.toggle("hidden", tab.status !== "generating");
  }
}

function makeGenCard(idx, total, isVideo) {
  const card = document.createElement("div");
  card.className = "gen-card" + (isVideo ? " is-video" : "");
  card.dataset.idx = String(idx);

  const thumbWrap = document.createElement("div");
  thumbWrap.className = "gen-thumb-wrap";
  if (isVideo) {
    const video = document.createElement("video");
    video.className = "gen-thumb hidden";
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    thumbWrap.appendChild(video);
  } else {
    const img = document.createElement("img");
    img.className = "gen-thumb hidden";
    thumbWrap.appendChild(img);
  }
  const spinnerWrap = document.createElement("div");
  spinnerWrap.className = "gen-card-spinner";
  const spin = document.createElement("div");
  spin.className = "spin";
  spinnerWrap.appendChild(spin);
  thumbWrap.appendChild(spinnerWrap);
  card.appendChild(thumbWrap);

  const status = document.createElement("span");
  status.className = "gen-card-status";
  status.textContent = `#${idx + 1} · Queued…`;
  card.appendChild(status);

  const actions = document.createElement("div");
  actions.className = "gen-card-actions hidden";
  const saveBtn = document.createElement("button");
  saveBtn.className = "gen-card-save";
  saveBtn.textContent = "Save";
  actions.appendChild(saveBtn);
  card.appendChild(actions);

  if (!isVideo) {
    const recutBtn = document.createElement("button");
    recutBtn.className = "gen-card-recut hidden";
    recutBtn.setAttribute("aria-label", "Re-run cutout");
    recutBtn.setAttribute("title", "Re-run cutout");
    card.appendChild(recutBtn);
  }

  return card;
}

function updateCardFromResult(card, result, tab, idx) {
  const isVideo = isVideoModel(tab.model);
  const media = card.querySelector(isVideo ? "video.gen-thumb" : "img.gen-thumb");
  const spinner = card.querySelector(".gen-card-spinner");
  const statusEl = card.querySelector(".gen-card-status");
  const actions = card.querySelector(".gen-card-actions");
  const saveBtn = actions ? actions.querySelector(".gen-card-save") : null;
  const recutBtn = card.querySelector(".gen-card-recut");

  // No result yet — queued or in-progress
  if (!result || (!result.ok && !result.error)) {
    card.classList.remove("error");
    media.classList.add("hidden");
    spinner.style.display = "flex";
    if (result && result.progress) {
      statusEl.textContent = `#${idx + 1} · ${result.progress}`;
    } else {
      statusEl.textContent = `#${idx + 1} · ${isVideo ? "Queued — Veo takes 30s–2min" : "Generating…"}`;
    }
    actions.classList.add("hidden");
    if (recutBtn) recutBtn.classList.add("hidden");
    return;
  }

  // Error state
  if (result.error) {
    card.classList.add("error");
    media.classList.add("hidden");
    spinner.style.display = "none";
    statusEl.textContent = `#${idx + 1} · ${result.error}`;
    actions.classList.add("hidden");
    if (recutBtn) recutBtn.classList.add("hidden");
    return;
  }

  // Success — show media
  card.classList.remove("error");
  if (result.blob) {
    if (media._currentUrl && media._currentBlob !== result.blob) {
      URL.revokeObjectURL(media._currentUrl);
      media._currentUrl = null;
    }
    if (!media._currentUrl || media._currentBlob !== result.blob) {
      const url = URL.createObjectURL(result.blob);
      media._currentUrl = url;
      media._currentBlob = result.blob;
      media.src = url;
    }
    media.classList.remove("hidden");
    spinner.style.display = result.cutting ? "flex" : "none";
    statusEl.textContent = `#${idx + 1} · ${result.cutting ? "Cutting out…" : (result.label || (isVideo ? "Generated" : "Generated"))}`;
    if (!result.cutting) {
      actions.classList.remove("hidden");
      if (recutBtn) recutBtn.classList.remove("hidden");
    } else {
      actions.classList.add("hidden");
      if (recutBtn) recutBtn.classList.add("hidden");
    }

    if (saveBtn) {
      saveBtn.onclick = async () => {
        const buf = await result.blob.arrayBuffer();
        const ext = isVideo ? "mp4" : "png";
        const name = `${slugify(tab.prompt)}-${idx + 1}.${ext}`;
        if (isVideo) await window.api.saveVideo(buf, name);
        else await window.api.saveFile(buf, name);
      };
    }
    if (recutBtn && result.raw) {
      recutBtn.onclick = async () => {
        result.cutting = true;
        updateCardFromResult(card, result, tab, idx);
        await runCutoutOnResult(tab, idx, true);
        if (tab.id === activeTabId) updateCardFromResult(card, tab.results[idx], tab, idx);
      };
    }
  }
}

// --- Cutout post-processing (image only) ---
function cutoutSettingsFor(mode) {
  if (mode === "none") return null;
  if (mode === "object") {
    return {
      model: "isnet-general-use",
      alphaMatting: false,
      colorRemove: true,
      colors: [],
      colorTolerance: 25,
      edgeSmooth: true,
      edgeStrength: 60,
      edgeTrim: 1,
      autoTrim: true,
    };
  }
  return {
    model: "isnet-general-use",
    alphaMatting: true,
    fgThreshold: 240,
    bgThreshold: 10,
    erodeSize: 10,
    colorRemove: false,
    colors: [],
    colorTolerance: 20,
    edgeSmooth: true,
    edgeStrength: 50,
    edgeTrim: 1,
    autoTrim: true,
  };
}

async function runCutoutOnResult(tab, idx, force = false) {
  const result = tab.results[idx];
  if (!result || !result.ok || !result.blob) return;
  if (!result.raw) result.raw = result.blob;

  if (tab.cutoutMode === "none" && !force) {
    result.label = "Raw";
    result.cutting = false;
    return;
  }

  result.cutting = true;
  if (tab.id === activeTabId) {
    const card = genResultsGrid.children[idx];
    if (card) updateCardFromResult(card, result, tab, idx);
  }

  try {
    const settings = cutoutSettingsFor(tab.cutoutMode === "none" ? "figure" : tab.cutoutMode);
    if (!settings) {
      result.label = "Raw";
      return;
    }
    const rawBytes = new Uint8Array(await result.raw.arrayBuffer());
    const slice = rawBytes.buffer.slice(rawBytes.byteOffset, rawBytes.byteOffset + rawBytes.byteLength);
    const cutBuf = await window.api.removeBg(slice, settings);
    result.blob = new Blob([cutBuf], { type: "image/png" });
    result.label = "Cut + trimmed";
  } catch (err) {
    result.label = "Raw (cutout failed)";
    console.error("cutout failed", err);
  } finally {
    result.cutting = false;
  }
}

// --- Helpers (kept from previous version) ---
function readFileAsBase64(file) {
  return new Promise((res, rej) => {
    const fr = new FileReader();
    fr.onload = () => {
      const result = fr.result;
      const comma = result.indexOf(",");
      res({ mimeType: file.type || "image/png", base64: result.slice(comma + 1) });
    };
    fr.onerror = () => rej(new Error("Failed to read reference image"));
    fr.readAsDataURL(file);
  });
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function bufferFromIpc(data) {
  if (data instanceof Uint8Array) return data;
  if (data && data.type === "Buffer" && Array.isArray(data.data)) return new Uint8Array(data.data);
  if (data instanceof ArrayBuffer) return new Uint8Array(data);
  return new Uint8Array(data);
}

function slugify(s) {
  return (s || "").slice(0, 40).replace(/[^a-z0-9]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "generated";
}

// --- Populate model picker ---
function populateModelSelect() {
  genModelSelect.textContent = "";
  let lastKind = null;
  for (const m of MODEL_DEFS) {
    if (lastKind && lastKind !== m.kind) {
      const sep = document.createElement("option");
      sep.disabled = true;
      sep.textContent = "──────────";
      genModelSelect.appendChild(sep);
    }
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.label;
    genModelSelect.appendChild(opt);
    lastKind = m.kind;
  }
}
populateModelSelect();

// --- API key field wiring (unchanged) ---
if (geminiKeyInput && window.api.getSettings) {
  window.api.getSettings().then((s) => {
    if (s && s.geminiApiKey) geminiKeyInput.value = s.geminiApiKey;
  }).catch(() => {});
  let keyTimer = null;
  geminiKeyInput.addEventListener("input", () => {
    clearTimeout(keyTimer);
    keyTimer = setTimeout(() => {
      window.api.setSetting("geminiApiKey", geminiKeyInput.value.trim()).catch(() => {});
    }, 400);
  });
}

// --- Quick ref add (active tab) ---
async function addQuickRefFromFile(file) {
  if (!isImageFile(file)) return;
  const tab = activeTab();
  if (!tab) return;
  try {
    const ref = await readFileAsBase64(file);
    tab.refs.push({ source: "quick", mimeType: ref.mimeType, base64: ref.base64, name: file.name });
    renderRefsForTab(tab);
    persistTabs();
  } catch (err) {
    genStatusEl.textContent = err.message;
    genStatusEl.className = "error";
  }
}

genQuickRefBtn.addEventListener("click", () => genQuickRefInput.click());
genQuickRefInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) addQuickRefFromFile(file);
  genQuickRefInput.value = "";
});

// --- Form input → active tab state ---
genPrompt.addEventListener("input", () => {
  const tab = activeTab();
  if (!tab) return;
  tab.prompt = genPrompt.value;
  // Update tab label in place
  const idx = tabIndex(tab);
  const tabBtn = genTabsEl.children[idx];
  if (tabBtn) {
    const lbl = tabBtn.querySelector(".gen-tab-label");
    if (lbl) lbl.textContent = tabDisplayName(tab);
  }
  persistTabs();
});
genCount.addEventListener("input", () => {
  const tab = activeTab();
  if (tab) tab.count = Math.max(1, Math.min(12, parseInt(genCount.value) || 1));
  updateCostEstimate();
  persistTabs();
});
genCutoutMode.addEventListener("change", () => {
  const tab = activeTab();
  if (tab) tab.cutoutMode = genCutoutMode.value;
  persistTabs();
});
genModelSelect.addEventListener("change", () => {
  const tab = activeTab();
  if (!tab) return;
  tab.model = genModelSelect.value;
  renderFormForTab(tab);
  persistTabs();
});
genLengthEl.addEventListener("change", () => {
  const tab = activeTab();
  if (tab) tab.length = parseInt(genLengthEl.value) || 8;
  updateCostEstimate();
  persistTabs();
});
genAspectEl.addEventListener("change", () => {
  const tab = activeTab();
  if (tab) tab.aspect = genAspectEl.value;
  persistTabs();
});
genAudioEl.addEventListener("change", () => {
  const tab = activeTab();
  if (tab) tab.audio = genAudioEl.checked;
  persistTabs();
});

// --- Tab management buttons ---
genNewTabBtn.addEventListener("click", () => {
  saveFormIntoActiveTab();
  const prev = activeTab();
  const fresh = newTab({
    model: prev ? prev.model : "gemini-2.5-flash-image",
    aspect: prev ? prev.aspect : "16:9",
  });
  activeTabId = fresh.id;
  fresh.view = "form";
  renderActiveTab();
  persistTabs();
  setTimeout(() => genPrompt.focus(), 0);
});

// --- Mode switching ---
modeGenerateBtn.addEventListener("click", () => {
  setMode("generate");
  dropZone.classList.add("hidden");
  preview.classList.add("hidden");
  actions.classList.add("hidden");
  batchPanel.classList.add("hidden");
  spritePanel.classList.add("hidden");
  if (genSection) genSection.classList.remove("hidden");
  statusEl.textContent = "Generate mode";
  statusEl.className = "";
  renderActiveTab();
});

genBackBtn.addEventListener("click", () => {
  if (genSection) genSection.classList.add("hidden");
  genPanel.classList.add("hidden");
  genResultsPanel.classList.add("hidden");
  setMode("remove");
  dropZone.classList.remove("hidden");
  statusEl.textContent = "Ready";
  statusEl.className = "ready";
});

// --- Generate / Cancel / Edit / Reset / Save All ---
async function runGenerate() {
  const tab = activeTab();
  if (!tab) return;
  saveFormIntoActiveTab();

  if (!tab.prompt.trim()) {
    genStatusEl.textContent = "Prompt is empty";
    genStatusEl.className = "error";
    return;
  }

  const isVideo = isVideoModel(tab.model);

  if (!isVideo && tab.cutoutMode !== "none" && !backendReady) {
    genStatusEl.textContent = "AI engine still loading — wait for Ready, or pick Cutout: None";
    genStatusEl.className = "error";
    return;
  }

  // Reset the tab's results buffer to the new count
  tab.results = new Array(tab.count).fill(null);
  tab.status = "generating";
  tab.errorMsg = "";
  tab.view = "results";

  const refs = tab.refs.map(r => ({ mimeType: r.mimeType, base64: r.base64 }));
  const opts = {
    model: tab.model,
    prompt: tab.prompt,
    count: tab.count,
  };
  if (isVideo) {
    opts.length = tab.length;
    opts.aspect = tab.aspect;
    opts.audio = tab.audio;
    opts.startFrameIdx = tab.startFrameIdx;
    opts.referenceImages = refs;
  } else {
    opts.referenceImages = refs;
    opts.aspectRatio = "1:1";
  }

  renderActiveTab();

  let res;
  try {
    res = await window.api.geminiJobStart(opts);
  } catch (err) {
    tab.status = "error";
    tab.errorMsg = err.message;
    tab.view = "form";
    renderActiveTab();
    return;
  }

  tab.jobId = res.jobId;
  jobIdToTab.set(res.jobId, tab.id);
  if (tab.id === activeTabId) renderResultsForTab(tab);
}

genGoBtn.addEventListener("click", runGenerate);

if (genCancelBtn) {
  genCancelBtn.addEventListener("click", () => {
    const tab = activeTab();
    if (!tab || tab.status !== "generating" || !tab.jobId) return;
    window.api.geminiJobCancel(tab.jobId).catch(() => {});
  });
}

genAgainBtn.addEventListener("click", () => {
  const tab = activeTab();
  if (!tab) return;
  tab.view = "form";
  renderActiveTab();
});

genResetBtn.addEventListener("click", () => {
  const tab = activeTab();
  if (!tab) return;
  if (tab.status === "generating") {
    if (!confirm("Cancel running job and clear?")) return;
    if (tab.jobId) {
      window.api.geminiJobCancel(tab.jobId).catch(() => {});
      jobIdToTab.delete(tab.jobId);
    }
  }
  // Release any object URLs to free memory
  for (const card of genResultsGrid.children) {
    const m = card.querySelector("img,video");
    if (m && m._currentUrl) URL.revokeObjectURL(m._currentUrl);
  }
  tab.results = [];
  tab.hasUnviewedResults = false;
  tab.status = "idle";
  tab.errorMsg = "";
  tab.view = "form";
  tab.prompt = "";
  tab.refs = [];
  tab.startFrameIdx = 0;
  renderActiveTab();
  persistTabs();
});

genSaveAllBtn.addEventListener("click", async () => {
  const tab = activeTab();
  if (!tab) return;
  const ok = tab.results.filter(r => r && r.ok && r.blob);
  if (!ok.length) return;
  const outDir = await window.api.selectDirectory();
  if (!outDir) return;
  const base = slugify(tab.prompt);
  const isVideo = isVideoModel(tab.model);
  const ext = isVideo ? "mp4" : "png";
  let saved = 0;
  for (let i = 0; i < tab.results.length; i++) {
    const r = tab.results[i];
    if (!r || !r.ok || !r.blob) continue;
    const buf = await r.blob.arrayBuffer();
    const outPath = outDir.replace(/\\/g, "/") + "/" + `${base}-${i + 1}.${ext}`;
    const okSave = await window.api.saveToPath(buf, outPath);
    if (okSave) saved++;
  }
  statusEl.textContent = `Saved ${saved} of ${ok.length} to ${outDir}`;
  statusEl.className = "ready";
});

// --- Job event handling — routes to correct tab regardless of which is active ---
window.api.onGeminiJobEvent((e) => {
  const tabId = jobIdToTab.get(e.jobId);
  if (!tabId) return;
  const tab = tabsById.get(tabId);
  if (!tab) return;

  if (e.kind === "image" || e.kind === "video") {
    const isVideo = e.kind === "video";
    const mimeType = e.mimeType || (isVideo ? "video/mp4" : "image/png");
    const blob = e.ok && e.data
      ? new Blob([bufferFromIpc(e.data)], { type: mimeType })
      : null;
    const result = {
      ok: e.ok,
      kind: e.kind,
      blob,
      raw: blob,
      error: e.error,
      label: e.ok ? (isVideo ? "Generated" : (tab.cutoutMode === "none" ? "Raw" : "")) : "",
      cutting: false,
    };
    tab.results[e.idx] = result;

    if (tab.id === activeTabId) {
      const card = genResultsGrid.children[e.idx];
      if (card) updateCardFromResult(card, result, tab, e.idx);
    } else {
      tab.hasUnviewedResults = true;
      renderTabs();
    }

    // Async cutout for image results
    if (!isVideo && e.ok && tab.cutoutMode !== "none") {
      runCutoutOnResult(tab, e.idx).then(() => {
        if (tab.id === activeTabId) {
          const card = genResultsGrid.children[e.idx];
          if (card) updateCardFromResult(card, tab.results[e.idx], tab, e.idx);
        }
      });
    }
    return;
  }

  if (e.kind === "progress") {
    if (e.idx != null) {
      const existing = tab.results[e.idx] || {};
      tab.results[e.idx] = { ...existing, progress: e.message };
      if (tab.id === activeTabId) {
        const card = genResultsGrid.children[e.idx];
        if (card) updateCardFromResult(card, tab.results[e.idx], tab, e.idx);
      }
    }
    return;
  }

  if (e.kind === "done") {
    tab.status = "ready";
    tab.jobId = null;
    jobIdToTab.delete(e.jobId);
    if (tab.id !== activeTabId) tab.hasUnviewedResults = true;
    renderTabs();
    if (tab.id === activeTabId) {
      // Re-render form to re-enable Generate button + update status
      renderFormForTab(tab);
      renderResultsForTab(tab);
    }
    return;
  }

  if (e.kind === "error") {
    tab.status = "error";
    tab.errorMsg = e.message;
    tab.jobId = null;
    jobIdToTab.delete(e.jobId);
    if (tab.id === activeTabId) {
      tab.view = "form";
      renderActiveTab();
    } else {
      renderTabs();
    }
    return;
  }
});

// Restore tabs on startup
restoreTabs();

// ===================== REFERENCE LIBRARY =====================

const libModal = document.getElementById("lib-modal");
const libCard = document.getElementById("lib-card");
const libBackdrop = document.getElementById("lib-backdrop");
const libCloseBtn = document.getElementById("lib-close");
const libAddBtn = document.getElementById("lib-add-btn");
const libAddInput = document.getElementById("lib-add-input");
const libCategoryChips = document.getElementById("lib-category-chips");
const libGrid = document.getElementById("lib-grid");
const libEmpty = document.getElementById("lib-empty");
const libCancelBtn = document.getElementById("lib-cancel-btn");
const libUseBtn = document.getElementById("lib-use-btn");
const libSelectedCount = document.getElementById("lib-selected-count");
const libCategoriesList = document.getElementById("lib-categories-list");

let libItems = [];
let libCategoryFilter = null;
let libSelectedIds = new Set();
let libDragDepth = 0;
let libSessionCategories = new Set();

async function libRefresh() {
  let raw;
  try {
    raw = await window.api.referencesList();
  } catch (err) {
    console.error("Failed to load references", err);
    raw = [];
  }
  libItems = raw.map((item) => {
    const bytes = item.data instanceof Uint8Array ? item.data : new Uint8Array(item.data || []);
    return { ...item, base64: bytesToBase64(bytes) };
  });
  libRender();
}

function libCategoryCounts() {
  const counts = new Map();
  for (const item of libItems) {
    counts.set(item.category, (counts.get(item.category) || 0) + 1);
  }
  return counts;
}

function libRenderChips() {
  libCategoryChips.textContent = "";
  const counts = libCategoryCounts();
  for (const cat of libSessionCategories) {
    if (!counts.has(cat)) counts.set(cat, 0);
  }
  const chips = [{ label: "All", value: null, count: libItems.length }];
  const sortedCats = Array.from(counts.keys()).sort((a, b) => a.localeCompare(b));
  for (const cat of sortedCats) chips.push({ label: cat, value: cat, count: counts.get(cat) });

  for (const chip of chips) {
    const el = document.createElement("button");
    el.className = "lib-chip" + (chip.value === libCategoryFilter ? " active" : "");
    el.textContent = chip.label + " ";
    const c = document.createElement("span");
    c.className = "lib-chip-count";
    c.textContent = chip.count;
    el.appendChild(c);
    el.addEventListener("click", () => {
      libCategoryFilter = chip.value;
      libRender();
    });
    libCategoryChips.appendChild(el);
  }

  const newChip = document.createElement("button");
  newChip.className = "lib-chip lib-chip-new";
  newChip.textContent = "+ New category";
  newChip.title = "Create a new category — uploads while it's active go straight to it";
  newChip.addEventListener("click", () => libStartNewCategory(newChip));
  libCategoryChips.appendChild(newChip);
}

function libStartNewCategory(chipEl) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "lib-chip-new-input";
  input.placeholder = "Category name…";
  input.maxLength = 40;
  chipEl.replaceWith(input);
  input.focus();

  let finished = false;
  const finish = async (commit) => {
    if (finished) return;
    finished = true;
    const name = input.value.trim();
    if (commit && name) {
      await libRegisterCategory(name);
      libCategoryFilter = name;
    }
    libRender();
  };
  input.addEventListener("blur", () => finish(true));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); finish(true); }
    if (e.key === "Escape") { e.preventDefault(); input.value = ""; finish(false); }
  });
}

function libRenderCategoriesList() {
  libCategoriesList.textContent = "";
  const counts = libCategoryCounts();
  for (const cat of counts.keys()) {
    const opt = document.createElement("option");
    opt.value = cat;
    libCategoriesList.appendChild(opt);
  }
}

function libVisibleItems() {
  if (libCategoryFilter == null) return libItems;
  return libItems.filter((x) => x.category === libCategoryFilter);
}

function libMakeItemCard(item) {
  const card = document.createElement("div");
  card.className = "lib-item" + (libSelectedIds.has(item.id) ? " selected" : "");

  const thumbWrap = document.createElement("div");
  thumbWrap.className = "lib-thumb-wrap";
  const img = document.createElement("img");
  img.src = `data:${item.mimeType};base64,${item.base64}`;
  img.alt = item.name || "";
  thumbWrap.appendChild(img);

  const check = document.createElement("div");
  check.className = "lib-check";
  const svgNs = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNs, "svg");
  svg.setAttribute("width", "12");
  svg.setAttribute("height", "12");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "3");
  const poly = document.createElementNS(svgNs, "polyline");
  poly.setAttribute("points", "20 6 9 17 4 12");
  svg.appendChild(poly);
  check.appendChild(svg);
  thumbWrap.appendChild(check);

  const del = document.createElement("button");
  del.className = "lib-delete";
  del.textContent = "×";
  del.title = "Delete from library";
  del.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!window.confirm(`Delete "${item.name}" from the library?`)) return;
    await window.api.referencesDelete(item.id);
    libSelectedIds.delete(item.id);
    await libRefresh();
    libUpdateFooter();
  });
  thumbWrap.appendChild(del);

  thumbWrap.addEventListener("click", () => {
    if (libSelectedIds.has(item.id)) libSelectedIds.delete(item.id);
    else libSelectedIds.add(item.id);
    card.classList.toggle("selected");
    libUpdateFooter();
  });

  card.appendChild(thumbWrap);

  const svgNs2 = "http://www.w3.org/2000/svg";
  function makeFieldIcon(d) {
    const s = document.createElementNS(svgNs2, "svg");
    s.setAttribute("width", "11");
    s.setAttribute("height", "11");
    s.setAttribute("viewBox", "0 0 24 24");
    s.setAttribute("fill", "none");
    s.setAttribute("stroke", "currentColor");
    s.setAttribute("stroke-width", "2");
    s.setAttribute("stroke-linecap", "round");
    s.setAttribute("stroke-linejoin", "round");
    const p = document.createElementNS(svgNs2, "path");
    p.setAttribute("d", d);
    s.appendChild(p);
    return s;
  }

  const nameField = document.createElement("div");
  nameField.className = "lib-edit-field";
  nameField.appendChild(makeFieldIcon("M12 20h9 M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4z"));
  const nameInput = document.createElement("input");
  nameInput.className = "lib-name-input";
  nameInput.value = item.name || "";
  nameInput.placeholder = "Click to name…";
  nameInput.title = "Click to rename";
  nameInput.addEventListener("click", (e) => e.stopPropagation());
  nameInput.addEventListener("blur", async () => {
    const newName = nameInput.value.trim() || "Untitled";
    if (newName === item.name) return;
    await window.api.referencesUpdate(item.id, { name: newName });
    item.name = newName;
  });
  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") nameInput.blur();
    if (e.key === "Escape") { nameInput.value = item.name || ""; nameInput.blur(); }
  });
  nameField.appendChild(nameInput);
  card.appendChild(nameField);

  const catChip = document.createElement("button");
  catChip.type = "button";
  catChip.className = "lib-category-chip";
  catChip.title = "Click to set category or create a new one";
  const tagIcon = makeFieldIcon("M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z M7 7h.01");
  tagIcon.classList.add("lib-category-chip-tag");
  catChip.appendChild(tagIcon);
  const catLabel = document.createElement("span");
  catLabel.className = "lib-category-chip-label";
  catLabel.textContent = item.category || "Uncategorized";
  catChip.appendChild(catLabel);
  const chevIcon = makeFieldIcon("M6 9l6 6 6-6");
  chevIcon.classList.add("lib-category-chip-chevron");
  catChip.appendChild(chevIcon);
  catChip.addEventListener("click", (e) => {
    e.stopPropagation();
    openCategoryPopover(item, catChip);
  });
  card.appendChild(catChip);

  return card;
}

let _activeCategoryPopover = null;
function closeCategoryPopover() {
  if (_activeCategoryPopover) {
    _activeCategoryPopover.remove();
    _activeCategoryPopover = null;
  }
  document.removeEventListener("mousedown", _categoryPopoverOutsideHandler, true);
  document.removeEventListener("keydown", _categoryPopoverKeyHandler, true);
}
function _categoryPopoverOutsideHandler(e) {
  if (_activeCategoryPopover && !_activeCategoryPopover.contains(e.target)) {
    closeCategoryPopover();
  }
}
function _categoryPopoverKeyHandler(e) {
  if (e.key === "Escape") closeCategoryPopover();
}

function openCategoryPopover(item, anchorEl) {
  closeCategoryPopover();

  const popover = document.createElement("div");
  popover.className = "lib-category-popover";

  const counts = libCategoryCounts();
  for (const cat of libSessionCategories) {
    if (!counts.has(cat)) counts.set(cat, 0);
  }
  const cats = Array.from(counts.keys()).sort((a, b) => a.localeCompare(b));

  const header = document.createElement("div");
  header.className = "lib-category-popover-header";
  header.textContent = "Set category";
  popover.appendChild(header);

  for (const cat of cats) {
    const opt = document.createElement("button");
    opt.type = "button";
    opt.className = "lib-category-option" + (cat === item.category ? " active" : "");
    const label = document.createElement("span");
    label.className = "lib-category-option-label";
    label.textContent = cat;
    opt.appendChild(label);
    const count = document.createElement("span");
    count.className = "lib-category-option-count";
    count.textContent = counts.get(cat) || 0;
    opt.appendChild(count);
    opt.addEventListener("click", async (e) => {
      e.stopPropagation();
      closeCategoryPopover();
      await window.api.referencesUpdate(item.id, { category: cat });
      await libRefresh();
    });
    popover.appendChild(opt);
  }

  if (cats.length > 0) {
    const divider = document.createElement("div");
    divider.className = "lib-category-divider";
    popover.appendChild(divider);
  }

  const createWrap = document.createElement("div");
  createWrap.className = "lib-category-create";
  const createInput = document.createElement("input");
  createInput.type = "text";
  createInput.placeholder = "+ New category…";
  createInput.maxLength = 40;
  createInput.className = "lib-category-create-input";
  createInput.addEventListener("click", (e) => e.stopPropagation());
  createInput.addEventListener("mousedown", (e) => e.stopPropagation());
  createInput.addEventListener("keydown", async (e) => {
    e.stopPropagation();
    if (e.key === "Enter") {
      e.preventDefault();
      const name = createInput.value.trim();
      if (!name) return;
      closeCategoryPopover();
      await libRegisterCategory(name);
      await window.api.referencesUpdate(item.id, { category: name });
      await libRefresh();
    } else if (e.key === "Escape") {
      closeCategoryPopover();
    }
  });
  createWrap.appendChild(createInput);
  popover.appendChild(createWrap);

  document.body.appendChild(popover);

  const rect = anchorEl.getBoundingClientRect();
  const popoverHeight = popover.offsetHeight;
  const popoverWidth = popover.offsetWidth;
  const viewportH = window.innerHeight;
  const viewportW = window.innerWidth;

  let top = rect.bottom + 6;
  if (top + popoverHeight > viewportH - 8) {
    top = Math.max(8, rect.top - popoverHeight - 6);
  }
  let left = rect.left;
  if (left + popoverWidth > viewportW - 8) {
    left = Math.max(8, viewportW - popoverWidth - 8);
  }
  popover.style.top = `${top}px`;
  popover.style.left = `${left}px`;

  _activeCategoryPopover = popover;
  setTimeout(() => {
    document.addEventListener("mousedown", _categoryPopoverOutsideHandler, true);
    document.addEventListener("keydown", _categoryPopoverKeyHandler, true);
    createInput.focus();
  }, 0);
}

function libRender() {
  libRenderChips();
  libRenderCategoriesList();
  libGrid.textContent = "";
  const items = libVisibleItems();
  libEmpty.classList.toggle("hidden", items.length > 0);
  for (const item of items) {
    libGrid.appendChild(libMakeItemCard(item));
  }
}

function libUpdateFooter() {
  const n = libSelectedIds.size;
  libSelectedCount.textContent = n === 0 ? "0 selected" : `${n} selected`;
  libUseBtn.disabled = n === 0;
}

async function libAddFiles(files) {
  let added = 0;
  for (const file of files) {
    if (!isImageFile(file)) continue;
    try {
      const buf = await file.arrayBuffer();
      const cleanName = file.name.replace(/\.[^.]+$/, "");
      await window.api.referencesAdd({
        name: cleanName,
        category: libCategoryFilter || "Uncategorized",
        mimeType: file.type || "image/png",
        data: new Uint8Array(buf),
      });
      added++;
    } catch (err) {
      console.error("Failed to add reference", err);
    }
  }
  if (added > 0) await libRefresh();
}

async function libRegisterCategory(name) {
  const trimmed = String(name || "").trim();
  if (!trimmed) return;
  libSessionCategories.add(trimmed);
  try {
    const s = await window.api.getSettings();
    const stored = new Set((s && s.userCategories) || []);
    if (!stored.has(trimmed)) {
      stored.add(trimmed);
      await window.api.setSetting("userCategories", Array.from(stored));
    }
  } catch (err) {
    console.error("Failed to persist category", err);
  }
}

async function libUnregisterCategory(name) {
  const trimmed = String(name || "").trim();
  if (!trimmed) return;
  libSessionCategories.delete(trimmed);
  try {
    const s = await window.api.getSettings();
    const stored = new Set((s && s.userCategories) || []);
    if (stored.has(trimmed)) {
      stored.delete(trimmed);
      await window.api.setSetting("userCategories", Array.from(stored));
    }
  } catch (err) {
    console.error("Failed to unregister category", err);
  }
}

async function libOpen() {
  libModal.classList.remove("hidden");
  libSelectedIds = new Set();
  libCategoryFilter = null;
  try {
    const s = await window.api.getSettings();
    libSessionCategories = new Set((s && s.userCategories) || []);
  } catch {
    libSessionCategories = new Set();
  }
  libUpdateFooter();
  libRefresh();
}

function libClose() {
  libModal.classList.add("hidden");
  libCard.classList.remove("dragover");
  libDragDepth = 0;
}

genLibBtn.addEventListener("click", libOpen);
libCloseBtn.addEventListener("click", libClose);
libBackdrop.addEventListener("click", libClose);
libCancelBtn.addEventListener("click", libClose);
libAddBtn.addEventListener("click", () => libAddInput.click());
libAddInput.addEventListener("change", async (e) => {
  const files = [...(e.target.files || [])];
  await libAddFiles(files);
  libAddInput.value = "";
});

libUseBtn.addEventListener("click", () => {
  const tab = activeTab();
  if (!tab) { libClose(); return; }
  for (const id of libSelectedIds) {
    const item = libItems.find((x) => x.id === id);
    if (!item) continue;
    if (tab.refs.some((r) => r.source === "library" && r.id === id)) continue;
    tab.refs.push({
      source: "library",
      id: item.id,
      name: item.name,
      mimeType: item.mimeType,
      base64: item.base64,
    });
  }
  renderRefsForTab(tab);
  persistTabs();
  libClose();
});

libCard.addEventListener("dragenter", (e) => {
  e.preventDefault();
  e.stopPropagation();
  libDragDepth++;
  libCard.classList.add("dragover");
});
libCard.addEventListener("dragover", (e) => {
  e.preventDefault();
  e.stopPropagation();
});
libCard.addEventListener("dragleave", (e) => {
  e.preventDefault();
  e.stopPropagation();
  libDragDepth = Math.max(0, libDragDepth - 1);
  if (libDragDepth === 0) libCard.classList.remove("dragover");
});
libCard.addEventListener("drop", async (e) => {
  e.preventDefault();
  e.stopPropagation();
  libDragDepth = 0;
  libCard.classList.remove("dragover");
  const files = [...(e.dataTransfer.files || [])];
  await libAddFiles(files);
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !libModal.classList.contains("hidden")) libClose();
});

// ===================== PROMPT THEMES =====================

const genThemesChips = document.getElementById("gen-themes-chips");
const genThemeAddBtn = document.getElementById("gen-theme-add-btn");

let themes = [];

async function loadThemes() {
  try {
    const s = await window.api.getSettings();
    themes = Array.isArray(s && s.themes) ? s.themes : [];
  } catch {
    themes = [];
  }
  renderThemes();
}

async function persistThemes() {
  try {
    await window.api.setSetting("themes", themes);
  } catch (err) {
    console.error("Failed to persist themes", err);
  }
}

function insertThemeIntoPrompt(content) {
  const cur = genPrompt.value;
  let next;
  if (!cur.trim()) {
    next = content;
  } else {
    const trail = cur.slice(-1);
    const needsComma = !/[,;:.]/.test(trail) && !/\n/.test(trail);
    const sep = needsComma ? ", " : (/\s$/.test(cur) ? "" : " ");
    next = cur + sep + content;
  }
  genPrompt.value = next;
  genPrompt.dispatchEvent(new Event("input"));
  genPrompt.focus();
  // Move cursor to end
  genPrompt.setSelectionRange(genPrompt.value.length, genPrompt.value.length);
}

function renderThemes() {
  genThemesChips.textContent = "";
  for (const theme of themes) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "gen-theme-chip";
    chip.title = theme.content;
    const label = document.createElement("span");
    label.className = "gen-theme-chip-label";
    label.textContent = theme.name;
    chip.appendChild(label);
    const del = document.createElement("button");
    del.type = "button";
    del.className = "gen-theme-chip-delete";
    del.textContent = "×";
    del.title = "Delete theme";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (!window.confirm(`Delete theme "${theme.name}"?`)) return;
      themes = themes.filter((t) => t.id !== theme.id);
      await persistThemes();
      renderThemes();
    });
    chip.appendChild(del);
    chip.addEventListener("click", () => insertThemeIntoPrompt(theme.content));
    genThemesChips.appendChild(chip);
  }
}

let _activeThemePopover = null;
function closeThemeEditPopover() {
  if (_activeThemePopover) {
    _activeThemePopover.remove();
    _activeThemePopover = null;
  }
  document.removeEventListener("mousedown", _themePopoverOutsideHandler, true);
}
function _themePopoverOutsideHandler(e) {
  if (_activeThemePopover && !_activeThemePopover.contains(e.target) && e.target !== genThemeAddBtn) {
    closeThemeEditPopover();
  }
}

function openThemeEditPopover() {
  closeThemeEditPopover();

  const pop = document.createElement("div");
  pop.className = "theme-edit-popover";

  const header = document.createElement("div");
  header.className = "theme-edit-header";
  header.textContent = "Save theme";
  pop.appendChild(header);

  const nameInput = document.createElement("input");
  nameInput.type = "text";
  nameInput.className = "theme-edit-name";
  nameInput.placeholder = "Short label (e.g. Flat illustration)";
  nameInput.maxLength = 30;
  pop.appendChild(nameInput);

  const contentInput = document.createElement("textarea");
  contentInput.className = "theme-edit-content";
  contentInput.placeholder = "Text to insert into the prompt when you click this theme";
  pop.appendChild(contentInput);

  const actions = document.createElement("div");
  actions.className = "theme-edit-actions";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "theme-edit-cancel";
  cancel.textContent = "Cancel";
  cancel.addEventListener("click", closeThemeEditPopover);
  actions.appendChild(cancel);
  const save = document.createElement("button");
  save.type = "button";
  save.className = "theme-edit-save";
  save.textContent = "Save";
  const trySave = async () => {
    const name = nameInput.value.trim();
    const content = contentInput.value.trim();
    nameInput.style.borderColor = name ? "" : "#e94560";
    contentInput.style.borderColor = content ? "" : "#e94560";
    if (!name || !content) return;
    themes.push({
      id: `theme-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      name,
      content,
    });
    await persistThemes();
    renderThemes();
    closeThemeEditPopover();
  };
  save.addEventListener("click", trySave);
  actions.appendChild(save);
  pop.appendChild(actions);

  contentInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      trySave();
    }
  });
  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      contentInput.focus();
    } else if (e.key === "Escape") {
      closeThemeEditPopover();
    }
  });

  document.body.appendChild(pop);

  const rect = genThemeAddBtn.getBoundingClientRect();
  const popH = pop.offsetHeight;
  const popW = pop.offsetWidth;
  const viewportH = window.innerHeight;
  const viewportW = window.innerWidth;

  let top = rect.bottom + 6;
  if (top + popH > viewportH - 8) top = Math.max(8, rect.top - popH - 6);
  let left = rect.left;
  if (left + popW > viewportW - 8) left = Math.max(8, viewportW - popW - 8);
  pop.style.top = `${top}px`;
  pop.style.left = `${left}px`;

  _activeThemePopover = pop;
  setTimeout(() => {
    document.addEventListener("mousedown", _themePopoverOutsideHandler, true);
    nameInput.focus();
  }, 0);
}

genThemeAddBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (_activeThemePopover) closeThemeEditPopover();
  else openThemeEditPopover();
});

loadThemes();
