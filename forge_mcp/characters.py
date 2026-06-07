"""Saved-character registry — named reference images for IPAdapter consistency.

In-memory dict mirrored to JSON (atomic writes); reference images stored as files under
results_root/characters/ — OUTSIDE results_root/files/, so the cache janitor never prunes them,
and on the persistent /results volume so they survive container rebuilds.
"""
import json
import os
import secrets
import threading
from datetime import datetime, timezone

_PUBLIC_FIELDS = ("name", "description", "mode", "weight", "ref_bytes", "dims",
                  "source", "created_at", "updated_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(name: str) -> str:
    return (name or "").strip().lower()


class CharacterError(Exception):
    """Readable, user-facing character-registry failure."""


class CharacterStore:
    def __init__(self, results_root: str):
        self.dir = os.path.join(results_root, "characters")
        self.path = os.path.join(self.dir, "characters.json")
        self._lock = threading.Lock()
        self._chars = {}
        os.makedirs(self.dir, exist_ok=True)
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._chars = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._chars = {}

    def _flush(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._chars, f, indent=1)
        os.replace(tmp, self.path)

    @staticmethod
    def _public(e: dict) -> dict:
        return {k: e[k] for k in _PUBLIC_FIELDS if k in e}

    def save(self, *, name, data, ext="png", description=None, mode="character",
             weight=0.8, source=None, dims=None) -> dict:
        """Create/replace a character; writes the ref image and returns the public entry."""
        k = _key(name)
        if not k:
            raise CharacterError("Character name is required")
        ref_id = secrets.token_urlsafe(8)
        ref_file = os.path.join(self.dir, f"{ref_id}.{ext or 'png'}")
        with open(ref_file, "wb") as f:
            f.write(data)
        with self._lock:
            prev = self._chars.get(k)
            if prev and prev.get("ref_file") and prev["ref_file"] != ref_file and os.path.isfile(prev["ref_file"]):
                try:
                    os.remove(prev["ref_file"])  # replace the old ref image
                except OSError:
                    pass
            entry = {
                "name": name.strip(), "key": k, "ref_file": ref_file, "ref_bytes": len(data),
                "description": description if description is not None else (prev or {}).get("description"),
                "mode": mode or (prev or {}).get("mode") or "character",
                "weight": weight if weight is not None else (prev or {}).get("weight", 0.8),
                "source": source, "dims": dims,
                "created_at": (prev or {}).get("created_at") or _now(), "updated_at": _now(),
            }
            self._chars[k] = entry
            self._flush()
        return self._public(entry)

    def get(self, name: str):
        """Internal entry (includes ref_file) or None."""
        with self._lock:
            e = self._chars.get(_key(name))
        return dict(e) if e else None

    def read_reference(self, name: str):
        """Return (ref_bytes, entry) for a saved character. Raises CharacterError if missing."""
        e = self.get(name)
        if not e:
            avail = ", ".join(sorted(c["name"] for c in self._chars.values())) or "(none)"
            raise CharacterError(f"No saved character named '{name}'. Saved: {avail}")
        rf = e.get("ref_file")
        if not rf or not os.path.isfile(rf):
            raise CharacterError(f"Character '{name}' reference file is missing on disk")
        with open(rf, "rb") as f:
            return f.read(), e

    def list(self) -> list:
        with self._lock:
            return [self._public(e) for e in self._chars.values()]

    def delete(self, name: str) -> dict:
        with self._lock:
            e = self._chars.pop(_key(name), None)
            if not e:
                raise CharacterError(f"No saved character named '{name}'")
            if e.get("ref_file") and os.path.isfile(e["ref_file"]):
                try:
                    os.remove(e["ref_file"])
                except OSError:
                    pass
            self._flush()
        return {"deleted": e.get("name")}
