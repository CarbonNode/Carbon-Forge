const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  saveFile: (buffer, defaultName) =>
    ipcRenderer.invoke("save-file", buffer, defaultName),
  removeBg: (buffer, settings) =>
    ipcRenderer.invoke("remove-bg", buffer, settings),
  onBackendStatus: (callback) =>
    ipcRenderer.on("backend-status", (_event, status) => callback(status)),
});
