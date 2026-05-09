const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  saveFile: (buffer, defaultName) =>
    ipcRenderer.invoke("save-file", buffer, defaultName),
  removeBg: (buffer, settings) =>
    ipcRenderer.invoke("remove-bg", buffer, settings),
  splitSprites: (buffer, settings) =>
    ipcRenderer.invoke("split-sprites", buffer, settings),
  onBackendStatus: (callback) =>
    ipcRenderer.on("backend-status", (_event, status) => callback(status)),

  // Batch processing
  selectDirectory: () => ipcRenderer.invoke("select-directory"),
  saveToPath: (buffer, filePath) =>
    ipcRenderer.invoke("save-to-path", buffer, filePath),
  openPath: (dirPath) => ipcRenderer.invoke("open-path", dirPath),
  openBackendLog: () => ipcRenderer.invoke("open-backend-log"),

  // App settings
  getSettings: () => ipcRenderer.invoke("get-settings"),
  setSetting: (key, value) => ipcRenderer.invoke("set-setting", key, value),
});
