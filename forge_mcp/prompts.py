"""Saved image-prompt registry (favorites) — named, reusable prompts.

In-memory dict mirrored to JSON (atomic writes) under results_root/prompts/prompts.json,
on the persistent /results volume so favorites survive container rebuilds. Prompts are pure
text (optionally carrying a {subject} placeholder) plus light metadata — no binary payload,
so this is a simpler cousin of CharacterStore.
"""
import json
import os
import threading
from datetime import datetime, timezone

_PUBLIC_FIELDS = ("name", "kind", "model", "notes", "refs", "created_at", "updated_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(name: str) -> str:
    return (name or "").strip().lower()


class PromptError(Exception):
    """Readable, user-facing prompt-registry failure."""


class PromptStore:
    def __init__(self, results_root: str):
        self.dir = os.path.join(results_root, "prompts")
        self.path = os.path.join(self.dir, "prompts.json")
        self._lock = threading.Lock()
        self._prompts = {}
        os.makedirs(self.dir, exist_ok=True)
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._prompts = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._prompts = {}

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._prompts, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _public(e: dict) -> dict:
        out = {k: e[k] for k in _PUBLIC_FIELDS if k in e and e[k] is not None}
        out["has_subject"] = "{subject}" in (e.get("prompt") or "")
        return out

    def save(self, *, name, prompt, kind="gemini", refs=None, model=None, notes=None) -> dict:
        """Create or replace a named prompt favorite."""
        k = _key(name)
        if not k:
            raise PromptError("Prompt name is required")
        if not (prompt or "").strip():
            raise PromptError("Prompt text is required")
        with self._lock:
            prev = self._prompts.get(k)
            entry = {
                "name": name.strip(), "key": k, "prompt": prompt,
                "kind": kind or (prev or {}).get("kind") or "gemini",
                "refs": refs if refs is not None else (prev or {}).get("refs"),
                "model": model if model is not None else (prev or {}).get("model"),
                "notes": notes if notes is not None else (prev or {}).get("notes"),
                "created_at": (prev or {}).get("created_at") or _now(),
                "updated_at": _now(),
            }
            self._prompts[k] = entry
            self._flush()
        return self._public(entry)

    def get(self, name: str):
        with self._lock:
            e = self._prompts.get(_key(name))
        return dict(e) if e else None

    def render(self, name: str, subject: str | None = None) -> dict:
        """Public entry with `prompt` filled — {subject} substituted when `subject` is given."""
        e = self.get(name)
        if not e:
            avail = ", ".join(sorted(p["name"] for p in self._prompts.values())) or "(none)"
            raise PromptError(f"No saved prompt named '{name}'. Saved: {avail}")
        text = e.get("prompt") or ""
        has_sub = "{subject}" in text
        if subject:
            text = text.replace("{subject}", subject)
        out = self._public(e)
        out["prompt"] = text
        out["needs_subject"] = has_sub and not subject
        return out

    def list(self) -> list:
        with self._lock:
            return [self._public(e) for e in self._prompts.values()]

    def delete(self, name: str) -> dict:
        with self._lock:
            e = self._prompts.pop(_key(name), None)
            if not e:
                raise PromptError(f"No saved prompt named '{name}'")
            self._flush()
        return {"deleted": e.get("name")}
