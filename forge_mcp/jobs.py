"""Veo job registry — in-memory dict mirrored to a JSON file (atomic writes)."""
import json
import os
import secrets
import threading
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._jobs = {}
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._jobs = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._jobs = {}

    def _flush(self):
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._jobs, f, indent=1)
        os.replace(tmp, self.path)

    def create(self, *, kind, model, prompt, project, subpath, filename) -> dict:
        job = {
            "id": "job_" + secrets.token_urlsafe(8),
            "kind": kind, "model": model, "prompt": prompt,
            "project": project, "subpath": subpath, "filename": filename,
            "status": "running", "message": "submitted",
            "operation_name": None, "results": [], "error": None,
            "created_at": _now(), "updated_at": _now(),
        }
        with self._lock:
            self._jobs[job["id"]] = job
            self._flush()
        return dict(job)

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(fields)
            job["updated_at"] = _now()
            self._flush()

    def get(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def running_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j["status"] == "running")

    def mark_interrupted(self) -> list:
        """Call on startup. Jobs mid-flight without a Google operation are dead;
        jobs WITH an operation name are returned for the caller to resume polling."""
        resumable = []
        with self._lock:
            for job in self._jobs.values():
                if job["status"] != "running":
                    continue
                if job.get("operation_name"):
                    resumable.append(dict(job))
                else:
                    job["status"] = "failed"
                    job["error"] = "interrupted by service restart before submission completed"
                    job["updated_at"] = _now()
            self._flush()
        return resumable
