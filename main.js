const { app, BrowserWindow, ipcMain, dialog, shell } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");

let pythonProcess = null;
let mainWindow = null;
let backendLogStream = null;
let backendReady = false;
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
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
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
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
}

app.whenReady().then(async () => {
  startBackend();
  createWindow();
  try {
    await waitForBackend();
    backendReady = true;
    logBackend("[ready] /health responded — backend ready");
    mainWindow.webContents.send("backend-status", "ready");
  } catch (e) {
    logBackend(`[ready] timed out waiting for /health: ${e.message}`);
    mainWindow.webContents.send("backend-status", "error");
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

app.on("window-all-closed", () => app.quit());
