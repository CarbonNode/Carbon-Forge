const { app, BrowserWindow, ipcMain, dialog, shell, Tray, Menu, nativeImage } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");

let pythonProcess = null;
let mainWindow = null;
let tray = null;
let isQuitting = false;
let backendLogStream = null;
let backendReady = false;
let backendFailed = false;
const startedHidden = process.argv.includes("--hidden");

function getSettingsPath() {
  const dir = app.getPath("userData");
  try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  return path.join(dir, "settings.json");
}

function loadSettings() {
  try {
    return JSON.parse(fs.readFileSync(getSettingsPath(), "utf8"));
  } catch {
    return {};
  }
}

function saveSettings(s) {
  try {
    fs.writeFileSync(getSettingsPath(), JSON.stringify(s, null, 2));
  } catch (e) {
    if (typeof logBackend === "function") logBackend(`[settings] save failed: ${e.message}`);
  }
}

function showMainWindow() {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  if (!mainWindow.isVisible()) mainWindow.show();
  mainWindow.focus();
}

function createTray() {
  if (tray) return;
  try {
    const iconPath = path.join(__dirname, "icon.ico");
    const image = nativeImage.createFromPath(iconPath);
    tray = new Tray(image.isEmpty() ? nativeImage.createEmpty() : image);
    tray.setToolTip("Carbon Isolate");
    const menu = Menu.buildFromTemplate([
      { label: "Show Carbon Isolate", click: () => showMainWindow() },
      { type: "separator" },
      { label: "Quit", click: () => { isQuitting = true; app.quit(); } },
    ]);
    tray.setContextMenu(menu);
    tray.on("click", () => showMainWindow());
  } catch (e) {
    if (typeof logBackend === "function") logBackend(`[tray] failed to create: ${e.message}`);
  }
}
const PORT = 5123;

function getBackendLogPath() {
  const dir = app.getPath("userData");
  try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  return path.join(dir, "backend.log");
}

function logBackend(line) {
  if (!backendLogStream) {
    try { backendLogStream = fs.createWriteStream(getBackendLogPath(), { flags: "a" }); } catch {}
  }
  const stamped = `[${new Date().toISOString()}] ${line}\n`;
  if (backendLogStream) backendLogStream.write(stamped);
  console.log(stamped.trimEnd());
}

function startBackend() {
  let cmd, args, opts;
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    PYTHONIOENCODING: "utf-8",
  };
  if (app.isPackaged) {
    const backendExe = path.join(process.resourcesPath, "backend", "carbon-isolate-backend.exe");
    cmd = backendExe;
    args = [String(PORT)];
    opts = { stdio: ["ignore", "pipe", "pipe"], windowsHide: true, env };
  } else {
    const serverPath = path.join(__dirname, "backend", "server.py");
    const pythonCmd = process.platform === "win32" ? "python3.exe" : "python3";
    cmd = pythonCmd;
    args = [serverPath, String(PORT)];
    opts = { shell: true, stdio: ["ignore", "pipe", "pipe"], windowsHide: true, env };
  }
  logBackend(`--- backend spawn: ${cmd} ${args.join(" ")} ---`);
  pythonProcess = spawn(cmd, args, opts);

  pythonProcess.stdout.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) logBackend(`[out] ${msg}`);
    if (msg.includes("MODEL_READY") && mainWindow && !backendReady) {
      backendReady = true;
      mainWindow.webContents.send("backend-status", "ready");
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    const msg = data.toString().trim();
    if (msg) logBackend(`[err] ${msg}`);
  });

  pythonProcess.on("error", (err) => {
    logBackend(`[spawn-error] ${err.message}`);
  });

  pythonProcess.on("exit", (code, signal) => {
    logBackend(`[exit] code=${code} signal=${signal}`);
  });
}

ipcMain.handle("open-backend-log", async () => {
  const p = getBackendLogPath();
  if (fs.existsSync(p)) shell.openPath(p);
  return p;
});

// Renderer can poll on startup to avoid the IPC race where the
// "backend-status" push fires before the renderer's listener is registered.
ipcMain.handle("get-backend-status", () => {
  if (backendReady) return "ready";
  if (backendFailed) return "error";
  return "loading";
});

function waitForBackend(retries = 300) {
  return new Promise((resolve, reject) => {
    const check = (attempt) => {
      if (attempt >= retries) return reject(new Error("Backend timeout"));
      const req = http.get(`http://127.0.0.1:${PORT}/health`, (res) => {
        if (res.statusCode === 200) return resolve();
        setTimeout(() => check(attempt + 1), 200);
      });
      req.on("error", () => setTimeout(() => check(attempt + 1), 200));
      req.setTimeout(1000, () => {
        req.destroy();
        setTimeout(() => check(attempt + 1), 200);
      });
    };
    check(0);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 720,
    backgroundColor: "#0d0d0d",
    autoHideMenuBar: true,
    icon: path.join(__dirname, "icon.ico"),
    title: "Carbon Isolate",
    show: !startedHidden,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));

  if (!app.isPackaged) {
    mainWindow.webContents.openDevTools({ mode: "detach" });
  }

  // Close button hides to tray instead of quitting
  mainWindow.on("close", (e) => {
    if (!isQuitting) {
      e.preventDefault();
      mainWindow.hide();
    }
  });
}

ipcMain.handle("remove-bg", async (_event, buffer, settings = {}) => {
  if (!backendReady) {
    throw new Error("AI engine is still loading — wait for status to say Ready");
  }
  const headers = {
    "Content-Type": "application/octet-stream",
  };
  if (settings.model) headers["X-Model"] = settings.model;
  if (settings.alphaMatting) headers["X-Alpha-Matting"] = "true";
  if (settings.fgThreshold != null)
    headers["X-FG-Threshold"] = String(settings.fgThreshold);
  if (settings.bgThreshold != null)
    headers["X-BG-Threshold"] = String(settings.bgThreshold);
  if (settings.erodeSize != null)
    headers["X-Erode-Size"] = String(settings.erodeSize);
  if (settings.colorRemove) headers["X-Color-Remove"] = "true";
  if (settings.colors) headers["X-Colors"] = JSON.stringify(settings.colors);
  if (settings.colorTolerance != null)
    headers["X-Color-Tolerance"] = String(settings.colorTolerance);
  if (settings.edgeSmooth) headers["X-Edge-Smooth"] = "true";
  if (settings.edgeStrength != null)
    headers["X-Edge-Strength"] = String(settings.edgeStrength);
  if (settings.edgeTrim != null)
    headers["X-Edge-Trim"] = String(settings.edgeTrim);

  // Watermark removal
  if (settings.watermarkRemove) headers["X-Watermark-Remove"] = "true";
  if (settings.watermarkPosition)
    headers["X-Watermark-Position"] = settings.watermarkPosition;
  if (settings.watermarkSize != null)
    headers["X-Watermark-Size"] = String(settings.watermarkSize);

  // Pipeline control
  if (settings.skipBg) headers["X-Skip-Bg"] = "true";
  if (settings.autoTrim) headers["X-Auto-Trim"] = "true";

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: PORT,
        path: "/remove-bg",
        method: "POST",
        headers,
        timeout: 300000, // 5 min for large images / model download
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const body = Buffer.concat(chunks);
          if (res.statusCode !== 200) {
            let msg = `Backend error ${res.statusCode}`;
            try {
              const parsed = JSON.parse(body.toString());
              if (parsed.error) msg += `: ${parsed.error}`;
            } catch {
              const preview = body.toString().slice(0, 300);
              if (preview) msg += `: ${preview}`;
            }
            logBackend(`[remove-bg] ${msg}`);
            return reject(new Error(msg));
          }
          if (body.length === 0) {
            return reject(new Error("Backend returned empty response"));
          }
          if (body[0] !== 0x89 || body[1] !== 0x50 || body[2] !== 0x4e || body[3] !== 0x47) {
            logBackend(`[remove-bg] non-PNG response (${body.length} bytes): ${body.toString().slice(0, 200)}`);
            return reject(new Error("Backend returned non-PNG data"));
          }
          resolve(body);
        });
      }
    );
    req.on("error", (err) => {
      logBackend(`[remove-bg] req error: ${err.message}`);
      reject(err);
    });
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out after 5 minutes"));
    });
    req.write(Buffer.from(buffer));
    req.end();
  });
});

ipcMain.handle("split-sprites", async (_event, buffer, settings = {}) => {
  if (!backendReady) {
    throw new Error("AI engine is still loading — wait for status to say Ready");
  }
  const headers = {
    "Content-Type": "application/octet-stream",
  };
  if (settings.model) headers["X-Model"] = settings.model;
  if (settings.alphaMatting) headers["X-Alpha-Matting"] = "true";
  if (settings.fgThreshold != null)
    headers["X-FG-Threshold"] = String(settings.fgThreshold);
  if (settings.bgThreshold != null)
    headers["X-BG-Threshold"] = String(settings.bgThreshold);
  if (settings.erodeSize != null)
    headers["X-Erode-Size"] = String(settings.erodeSize);
  if (settings.colorRemove) headers["X-Color-Remove"] = "true";
  if (settings.colors) headers["X-Colors"] = JSON.stringify(settings.colors);
  if (settings.colorTolerance != null)
    headers["X-Color-Tolerance"] = String(settings.colorTolerance);
  if (settings.edgeSmooth) headers["X-Edge-Smooth"] = "true";
  if (settings.edgeStrength != null)
    headers["X-Edge-Strength"] = String(settings.edgeStrength);
  if (settings.edgeTrim != null)
    headers["X-Edge-Trim"] = String(settings.edgeTrim);
  if (settings.watermarkRemove) headers["X-Watermark-Remove"] = "true";
  if (settings.watermarkPosition)
    headers["X-Watermark-Position"] = settings.watermarkPosition;
  if (settings.watermarkSize != null)
    headers["X-Watermark-Size"] = String(settings.watermarkSize);

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: PORT,
        path: "/split-sprites",
        method: "POST",
        headers,
        timeout: 300000,
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const body = Buffer.concat(chunks).toString();
          if (res.statusCode !== 200) {
            try {
              const err = JSON.parse(body);
              reject(new Error(err.error || `Backend error ${res.statusCode}`));
            } catch {
              reject(new Error(`Backend error ${res.statusCode}: ${body.slice(0, 200)}`));
            }
            return;
          }
          try {
            const json = JSON.parse(body);
            resolve(json);
          } catch (e) {
            reject(new Error("Invalid response from split-sprites"));
          }
        });
      }
    );
    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timed out"));
    });
    req.write(Buffer.from(buffer));
    req.end();
  });
});

ipcMain.handle("save-file", async (_event, buffer, defaultName) => {
  const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultName,
    filters: [{ name: "PNG Image", extensions: ["png"] }],
  });
  if (canceled || !filePath) return false;
  fs.writeFileSync(filePath, Buffer.from(buffer));
  return true;
});

ipcMain.handle("save-video", async (_event, buffer, defaultName) => {
  const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
    defaultPath: defaultName,
    filters: [{ name: "MP4 Video", extensions: ["mp4"] }],
  });
  if (canceled || !filePath) return false;
  fs.writeFileSync(filePath, Buffer.from(buffer));
  return true;
});

// Batch processing support
ipcMain.handle("select-directory", async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ["openDirectory", "createDirectory"],
    title: "Select output folder for processed images",
  });
  if (canceled || !filePaths.length) return null;
  return filePaths[0];
});

ipcMain.handle("save-to-path", async (_event, buffer, filePath) => {
  try {
    fs.writeFileSync(filePath, Buffer.from(buffer));
    return true;
  } catch (err) {
    console.error("save-to-path error:", err);
    return false;
  }
});

ipcMain.handle("open-path", async (_event, dirPath) => {
  shell.openPath(dirPath);
});

// Single instance lock — quit if another instance is already running
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    showMainWindow();
  });
}

ipcMain.handle("get-settings", () => loadSettings());
ipcMain.handle("set-setting", (_e, key, value) => {
  const s = loadSettings();
  s[key] = value;
  saveSettings(s);
  if (key === "startWithWindows") {
    try {
      app.setLoginItemSettings({
        openAtLogin: !!value,
        openAsHidden: true,
        args: ["--hidden"],
      });
    } catch (e) {
      logBackend(`[settings] setLoginItemSettings failed: ${e.message}`);
    }
  }
  return s;
});

// ===== Gemini / Imagen / Veo generation =====
const GEMINI_API = "https://generativelanguage.googleapis.com/v1beta";
const DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-image";

const MODEL_KIND = {
  "gemini-2.5-flash-image":         "gemini-image",
  "imagen-4.0-generate-001":        "imagen",
  "imagen-4.0-ultra-generate-001":  "imagen",
  "imagen-4.0-fast-generate-001":   "imagen",
  "veo-3.0-generate-001":           "veo",
  "veo-3.0-fast-generate-001":      "veo",
  "veo-2.0-generate-001":           "veo",
};
const IMAGEN_MAX_BATCH = {
  "imagen-4.0-ultra-generate-001": 1, // Ultra returns one sample per call
};

function modelKind(model) { return MODEL_KIND[model] || "gemini-image"; }

async function fetchJson(url, init, { maxRetries = 3, retryDelayMs = 1000, isCancelled } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    if (isCancelled && isCancelled()) throw new Error("Cancelled");
    let res;
    try {
      res = await fetch(url, init);
    } catch (err) {
      lastErr = err;
      await new Promise(r => setTimeout(r, retryDelayMs * (attempt + 1)));
      continue;
    }
    if (res.status === 429 || res.status >= 500) {
      let body = "";
      try { body = await res.text(); } catch {}
      lastErr = new Error(`HTTP ${res.status}: ${body.slice(0, 500)}`);
      await new Promise(r => setTimeout(r, retryDelayMs * (attempt + 1)));
      continue;
    }
    if (!res.ok) {
      let body = "";
      try { body = await res.text(); } catch {}
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 500)}`);
    }
    try {
      return await res.json();
    } catch (err) {
      throw new Error(`Invalid JSON from ${url}: ${err.message}`);
    }
  }
  throw lastErr || new Error("Request failed after retries");
}

async function callGeminiImage({ apiKey, model, prompt, referenceImages = [] }) {
  const url = `${GEMINI_API}/models/${model}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const parts = [
    ...referenceImages.map(r => ({ inlineData: { mimeType: r.mimeType, data: r.base64 } })),
    { text: prompt },
  ];
  const body = JSON.stringify({ contents: [{ parts }] });
  const json = await fetchJson(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
  });
  const outParts = json?.candidates?.[0]?.content?.parts || [];
  return outParts
    .filter(p => p.inlineData?.data)
    .map(p => Buffer.from(p.inlineData.data, "base64"));
}

async function callImagen({ apiKey, model, prompt, sampleCount = 1, aspectRatio = "1:1" }) {
  const url = `${GEMINI_API}/models/${model}:predict`;
  const body = JSON.stringify({
    instances: [{ prompt }],
    parameters: {
      sampleCount: Math.max(1, Math.min(4, sampleCount)),
      aspectRatio,
      personGeneration: "allow_adult",
    },
  });
  const json = await fetchJson(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body,
  });
  const preds = json.predictions || json.generatedImages || [];
  const out = [];
  for (const p of preds) {
    const b64 =
      p.bytesBase64Encoded ||
      (p.image && (p.image.imageBytes || p.image.bytesBase64Encoded)) ||
      p.imageBytes ||
      null;
    if (b64) out.push(Buffer.from(b64, "base64"));
  }
  return out;
}

async function startVeo({ apiKey, model, prompt, startImage, aspectRatio = "16:9", durationSeconds = 8, generateAudio = true }) {
  const url = `${GEMINI_API}/models/${model}:predictLongRunning`;
  const instance = { prompt };
  if (startImage) {
    instance.image = { inlineData: { mimeType: startImage.mimeType, data: startImage.base64 } };
  }
  const parameters = {
    aspectRatio,
    durationSeconds: String(durationSeconds),
    sampleCount: 1,
    personGeneration: "allow_adult",
  };
  // generateAudio is a Veo 3 affordance; Veo 2 ignores it
  if (generateAudio != null && /^veo-3\./.test(model)) {
    parameters.generateAudio = !!generateAudio;
  }
  const body = JSON.stringify({ instances: [instance], parameters });
  const json = await fetchJson(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-goog-api-key": apiKey,
    },
    body,
  });
  if (!json.name) throw new Error("Veo: no operation name returned");
  return json.name;
}

async function pollVeoOperation({ apiKey, operationName, isCancelled, onProgress }) {
  const url = `${GEMINI_API}/${operationName}`;
  const start = Date.now();
  let delay = 4000;
  while (true) {
    if (isCancelled && isCancelled()) throw new Error("Cancelled");
    await new Promise(r => setTimeout(r, delay));
    delay = 8000;
    if (isCancelled && isCancelled()) throw new Error("Cancelled");
    let json;
    try {
      const res = await fetch(url, { headers: { "x-goog-api-key": apiKey } });
      json = await res.json();
    } catch (err) {
      logBackend(`[veo] poll transient: ${err.message}`);
      continue;
    }
    if (json.error) {
      const msg = json.error.message || JSON.stringify(json.error);
      throw new Error(`Veo error: ${msg}`);
    }
    const elapsed = Math.round((Date.now() - start) / 1000);
    if (json.done) {
      const resp = json.response || {};
      const samples =
        resp.generateVideoResponse?.generatedSamples ||
        resp.generatedSamples ||
        [];
      if (!samples.length) {
        throw new Error("Veo finished with no video samples (likely safety-filtered)");
      }
      const sample = samples[0];
      const uri =
        sample.video?.uri ||
        sample.uri ||
        sample.video?.url ||
        null;
      const inlineB64 =
        sample.video?.bytesBase64Encoded ||
        sample.bytesBase64Encoded ||
        null;
      if (onProgress) onProgress(`Downloading video…`);
      return { uri, inlineB64 };
    }
    if (onProgress) onProgress(`Generating video… ${elapsed}s elapsed`);
  }
}

async function downloadVeoVideo({ uri, inlineB64 }, apiKey) {
  if (inlineB64) return Buffer.from(inlineB64, "base64");
  if (!uri) throw new Error("Veo response missing both video URI and inline bytes");
  const attempts = [
    { url: uri, headers: { "x-goog-api-key": apiKey } },
    { url: uri + (uri.includes("?") ? "&" : "?") + `key=${encodeURIComponent(apiKey)}`, headers: {} },
    { url: uri, headers: {} },
  ];
  let lastErr;
  for (const { url, headers } of attempts) {
    try {
      const res = await fetch(url, { headers });
      if (!res.ok) {
        lastErr = new Error(`Download HTTP ${res.status}`);
        continue;
      }
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length > 0) return buf;
      lastErr = new Error("Empty MP4 response");
    } catch (e) {
      lastErr = e;
    }
  }
  throw lastErr || new Error("Veo MP4 download failed");
}

// ===== Reference image library =====
function getReferencesDir() {
  const dir = path.join(app.getPath("userData"), "references");
  try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  return dir;
}
function getReferencesIndexPath() {
  return path.join(getReferencesDir(), "index.json");
}
function loadReferencesIndex() {
  try {
    const raw = fs.readFileSync(getReferencesIndexPath(), "utf8");
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch { return []; }
}
function saveReferencesIndex(items) {
  try {
    fs.writeFileSync(getReferencesIndexPath(), JSON.stringify(items, null, 2));
  } catch (e) {
    logBackend(`[references] save index failed: ${e.message}`);
  }
}
const EXT_BY_MIME = {
  "image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
  "image/gif": ".gif", "image/bmp": ".bmp",
};

ipcMain.handle("references-list", async () => {
  const items = loadReferencesIndex();
  const dir = getReferencesDir();
  const out = [];
  for (const item of items) {
    try {
      const buf = fs.readFileSync(path.join(dir, item.file));
      out.push({ ...item, data: buf });
    } catch (e) {
      logBackend(`[references] skipping missing file ${item.file}: ${e.message}`);
    }
  }
  return out;
});

ipcMain.handle("references-add", async (_e, { name, category, mimeType, data }) => {
  if (!data) throw new Error("Reference data is empty");
  const items = loadReferencesIndex();
  const id = `ref-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const ext = EXT_BY_MIME[mimeType] || ".png";
  const file = `${id}${ext}`;
  fs.writeFileSync(path.join(getReferencesDir(), file), Buffer.from(data));
  const record = {
    id,
    name: (name || "Untitled").slice(0, 80),
    category: (category || "Uncategorized").slice(0, 40),
    mimeType: mimeType || "image/png",
    file,
    addedAt: Date.now(),
  };
  items.unshift(record);
  saveReferencesIndex(items);
  logBackend(`[references] added ${record.id} category="${record.category}"`);
  return record;
});

ipcMain.handle("references-delete", async (_e, id) => {
  const items = loadReferencesIndex();
  const idx = items.findIndex((x) => x.id === id);
  if (idx < 0) return false;
  const removed = items.splice(idx, 1)[0];
  try { fs.unlinkSync(path.join(getReferencesDir(), removed.file)); } catch {}
  saveReferencesIndex(items);
  logBackend(`[references] deleted ${id}`);
  return true;
});

ipcMain.handle("references-update", async (_e, id, patch = {}) => {
  const items = loadReferencesIndex();
  const idx = items.findIndex((x) => x.id === id);
  if (idx < 0) return null;
  for (const key of ["name", "category"]) {
    if (key in patch) items[idx][key] = String(patch[key]).slice(0, 80);
  }
  saveReferencesIndex(items);
  return items[idx];
});

// ===== Job registry — background-running multi-tab generation =====
const generationJobs = new Map();

function emitJobEvent(jobId, kind, payload = {}) {
  if (!mainWindow || mainWindow.isDestroyed()) return;
  try {
    mainWindow.webContents.send("gemini-job-event", { jobId, kind, ...payload });
  } catch (e) {
    logBackend(`[gen] emit failed: ${e.message}`);
  }
}

ipcMain.handle("gemini-job-start", async (_event, opts = {}) => {
  const settings = loadSettings();
  const apiKey = (settings && settings.geminiApiKey) || process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Gemini API key not set — open Settings to add one");
  const prompt = String(opts.prompt || "").trim();
  if (!prompt) throw new Error("Prompt is empty");
  const model = String(opts.model || DEFAULT_GEMINI_MODEL);
  if (!MODEL_KIND[model]) throw new Error(`Unknown model: ${model}`);
  const count = Math.max(1, Math.min(12, parseInt(opts.count) || 1));

  const jobId = `job-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const job = {
    id: jobId,
    apiKey,
    opts: { ...opts, prompt, model, count },
    cancelled: false,
    startedAt: Date.now(),
  };
  generationJobs.set(jobId, job);
  logBackend(`[gen] start ${jobId} model=${model} count=${count} kind=${modelKind(model)}`);

  // Fire-and-forget — events emit results back to the renderer
  runJob(job).catch(err => {
    logBackend(`[gen] ${jobId} crashed: ${err.stack || err.message}`);
    emitJobEvent(jobId, "error", { message: err.message || String(err) });
    generationJobs.delete(jobId);
  });

  return { jobId };
});

ipcMain.handle("gemini-job-cancel", async (_event, jobId) => {
  const job = generationJobs.get(jobId);
  if (!job) return false;
  job.cancelled = true;
  logBackend(`[gen] cancel requested ${jobId}`);
  return true;
});

async function runJob(job) {
  try {
    const kind = modelKind(job.opts.model);
    if (kind === "veo") await runVideoJob(job);
    else if (kind === "imagen") await runImagenJob(job);
    else await runGeminiImageJob(job);
  } finally {
    generationJobs.delete(job.id);
  }
}

async function runGeminiImageJob(job) {
  const { prompt, model, count, referenceImages = [] } = job.opts;
  const apiKey = job.apiKey;
  let okN = 0, failN = 0;

  await Promise.all(Array.from({ length: count }, async (_, idx) => {
    if (job.cancelled) {
      emitJobEvent(job.id, "image", { idx, ok: false, error: "Cancelled" });
      failN++;
      return;
    }
    try {
      const bufs = await callGeminiImage({ apiKey, model, prompt, referenceImages });
      if (job.cancelled) return;
      if (!bufs.length) {
        emitJobEvent(job.id, "image", { idx, ok: false, error: "No image returned (likely safety-filtered)" });
        failN++;
        return;
      }
      emitJobEvent(job.id, "image", { idx, ok: true, data: bufs[0], mimeType: "image/png" });
      okN++;
    } catch (err) {
      emitJobEvent(job.id, "image", { idx, ok: false, error: err.message || String(err) });
      failN++;
    }
  }));

  emitJobEvent(job.id, "done", { ok: okN, failed: failN, total: count });
}

async function runImagenJob(job) {
  const { prompt, model, count } = job.opts;
  const apiKey = job.apiKey;
  const aspectRatio = job.opts.aspectRatio || "1:1";
  const maxBatch = IMAGEN_MAX_BATCH[model] || 4;

  let cursor = 0;
  let okN = 0, failN = 0;
  while (cursor < count) {
    if (job.cancelled) {
      for (let i = cursor; i < count; i++) {
        emitJobEvent(job.id, "image", { idx: i, ok: false, error: "Cancelled" });
        failN++;
      }
      break;
    }
    const batchSize = Math.min(maxBatch, count - cursor);
    try {
      const bufs = await callImagen({ apiKey, model, prompt, sampleCount: batchSize, aspectRatio });
      for (let i = 0; i < batchSize; i++) {
        const idx = cursor + i;
        const buf = bufs[i];
        if (buf) {
          emitJobEvent(job.id, "image", { idx, ok: true, data: buf, mimeType: "image/png" });
          okN++;
        } else {
          emitJobEvent(job.id, "image", { idx, ok: false, error: "No image returned (safety filter or empty batch)" });
          failN++;
        }
      }
    } catch (err) {
      const msg = err.message || String(err);
      for (let i = 0; i < batchSize; i++) {
        emitJobEvent(job.id, "image", { idx: cursor + i, ok: false, error: msg });
        failN++;
      }
    }
    cursor += batchSize;
  }

  emitJobEvent(job.id, "done", { ok: okN, failed: failN, total: count });
}

async function runVideoJob(job) {
  const { prompt, model, count, length, aspect, audio, referenceImages = [], startFrameIdx = 0 } = job.opts;
  const apiKey = job.apiKey;
  const startImage = referenceImages.length
    ? (referenceImages[startFrameIdx] || referenceImages[0])
    : null;
  const isCancelled = () => job.cancelled;

  let okN = 0, failN = 0;
  await Promise.all(Array.from({ length: count }, async (_, idx) => {
    if (job.cancelled) {
      emitJobEvent(job.id, "video", { idx, ok: false, error: "Cancelled" });
      failN++;
      return;
    }
    try {
      emitJobEvent(job.id, "progress", { idx, message: "Submitting to Veo…" });
      const operationName = await startVeo({
        apiKey, model, prompt,
        startImage,
        aspectRatio: aspect,
        durationSeconds: length,
        generateAudio: audio,
      });
      if (job.cancelled) {
        emitJobEvent(job.id, "video", { idx, ok: false, error: "Cancelled" });
        failN++;
        return;
      }
      emitJobEvent(job.id, "progress", { idx, message: `Generating video… 0s elapsed` });
      const sample = await pollVeoOperation({
        apiKey, operationName, isCancelled,
        onProgress: (msg) => emitJobEvent(job.id, "progress", { idx, message: msg }),
      });
      if (job.cancelled) return;
      const mp4 = await downloadVeoVideo(sample, apiKey);
      if (job.cancelled) return;
      emitJobEvent(job.id, "video", { idx, ok: true, data: mp4, mimeType: "video/mp4" });
      okN++;
    } catch (err) {
      const msg = err.message || String(err);
      logBackend(`[veo] ${job.id} idx=${idx} failed: ${msg}`);
      emitJobEvent(job.id, "video", { idx, ok: false, error: msg });
      failN++;
    }
  }));

  emitJobEvent(job.id, "done", { ok: okN, failed: failN, total: count });
}

app.whenReady().then(async () => {
  startBackend();
  createWindow();
  createTray();
  try {
    await waitForBackend();
    backendReady = true;
    logBackend("[ready] /health responded — backend ready");
    if (mainWindow) mainWindow.webContents.send("backend-status", "ready");
  } catch (e) {
    logBackend(`[ready] timed out waiting for /health: ${e.message}`);
    backendFailed = true;
    if (mainWindow) mainWindow.webContents.send("backend-status", "error");
  }
});

app.on("will-quit", () => {
  if (pythonProcess) {
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(pythonProcess.pid), "/f", "/t"]);
    } else {
      pythonProcess.kill();
    }
    pythonProcess = null;
  }
});

// Don't quit when window closes — tray keeps backend warm. Quit explicitly via tray.
app.on("window-all-closed", () => {
  if (process.platform !== "win32") app.quit();
});
