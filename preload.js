const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("api", {
  saveFile: (buffer, defaultName) =>
    ipcRenderer.invoke("save-file", buffer, defaultName),
  saveVideo: (buffer, defaultName) =>
    ipcRenderer.invoke("save-video", buffer, defaultName),
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

  // Gemini / Imagen / Veo job-based generation
  geminiJobStart: (opts) => ipcRenderer.invoke("gemini-job-start", opts),
  geminiJobCancel: (jobId) => ipcRenderer.invoke("gemini-job-cancel", jobId),
  onGeminiJobEvent: (callback) => {
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on("gemini-job-event", handler);
    return () => ipcRenderer.removeListener("gemini-job-event", handler);
  },

  // Reference library
  referencesList: () => ipcRenderer.invoke("references-list"),
  referencesAdd: (item) => ipcRenderer.invoke("references-add", item),
  referencesDelete: (id) => ipcRenderer.invoke("references-delete", id),
  referencesUpdate: (id, patch) => ipcRenderer.invoke("references-update", id, patch),
});
