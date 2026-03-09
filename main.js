const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");

let pythonProcess = null;
let mainWindow = null;
const PORT = 5123;

function startBackend() {
  let cmd, args, opts;
  if (app.isPackaged) {
    // Use bundled PyInstaller exe
    const backendExe = path.join(process.resourcesPath, "backend", "carbon-isolate-backend.exe");
    cmd = backendExe;
    args = [String(PORT)];
    opts = { stdio: ["ignore", "pipe", "pipe"] };
  } else {
    // Dev mode: run Python directly
    const serverPath = path.join(__dirname, "backend", "server.py");
    const pythonCmd = process.platform === "win32" ? "python3.exe" : "python3";
    cmd = pythonCmd;
    args = [serverPath, String(PORT)];
    opts = { shell: true, stdio: ["ignore", "pipe", "pipe"] };
  }
  pythonProcess = spawn(cmd, args, opts);

  pythonProcess.stdout.on("data", (data) => {
    const msg = data.toString().trim();
    console.log(`[backend] ${msg}`);
    if (msg.includes("MODEL_READY") && mainWindow) {
      mainWindow.webContents.send("backend-status", "ready");
    }
  });

  pythonProcess.stderr.on("data", (data) => {
    console.error(`[backend] ${data.toString().trim()}`);
  });

  pythonProcess.on("error", (err) => {
    console.error("Failed to start backend:", err);
  });
}

function waitForBackend(retries = 150) {
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

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: "127.0.0.1",
        port: PORT,
        path: "/remove-bg",
        method: "POST",
        headers,
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => resolve(Buffer.concat(chunks)));
      }
    );
    req.on("error", reject);
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

app.whenReady().then(async () => {
  startBackend();
  createWindow();
  try {
    await waitForBackend();
    mainWindow.webContents.send("backend-status", "ready");
  } catch {
    mainWindow.webContents.send("backend-status", "error");
  }
});

app.on("will-quit", () => {
  if (pythonProcess) {
    // shell: true spawns via cmd.exe, so kill the process tree
    if (process.platform === "win32") {
      spawn("taskkill", ["/pid", String(pythonProcess.pid), "/f", "/t"]);
    } else {
      pythonProcess.kill();
    }
    pythonProcess = null;
  }
});

app.on("window-all-closed", () => app.quit());
