"""Saved-character registry — named reference images for IPAdapter consistency.

In-memory dict mirrored to JSON (atomic writes); reference images stored as files under
results_root/characters/ — OUTSIDE results_root/files/, so the cache janitor never prunes them,
and on the persistent /results volume so they survive container rebuilds.

A character may hold MULTIPLE reference images (e.g. front / side / 3-4 angles); the IPAdapter
path averages their embeds for a stronger, more robust likeness.
"""
import json
import os
import secrets
import threading
from datetime import datetime, timezone

_PUBLIC_FIELDS = ("name", "description", "mode", "weight", "ref_count", "ref_bytes", "dims",
                  "source", "created_at", "updated_at")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _key(name: str) -> str:
    return (name or "").strip().lower()


class CharacterError(Exception):
    """Readable, user-facing character-registry failure."""


def _ref_files(entry: dict) -> list:
    """All reference file paths for an entry, tolerating the legacy single-`ref_file` schema."""
    if entry.get("ref_files"):
        return list(entry["ref_files"])
    return [entry["ref_file"]] if entry.get("ref_file") else []


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

    def _write_ref(self, data: bytes, ext: str) -> str:
        ref_file = os.path.join(self.dir, f"{secrets.token_urlsafe(8)}.{ext or 'png'}")
        with open(ref_file, "wb") as f:
            f.write(data)
        return ref_file

    @staticmethod
    def _public(e: dict) -> dict:
        out = {k: e[k] for k in _PUBLIC_FIELDS if k in e}
        out["ref_count"] = len(_ref_files(e))
        return out

    def save(self, *, name, data, ext="png", description=None, mode="character",
             weight=0.8, source=None, dims=None) -> dict:
        """Create/replace a character with a single primary reference. Replacing drops old refs."""
        k = _key(name)
        if not k:
            raise CharacterError("Character name is required")
        ref_file = self._write_ref(data, ext)
        with self._lock:
            prev = self._chars.get(k)
            for old in (_ref_files(prev) if prev else []):
                if old != ref_file and os.path.isfile(old):
                    try:
                        os.remove(old)  # replace: clear prior references
                    except OSError:
                        pass
            entry = {
                "name": name.strip(), "key": k, "ref_files": [ref_file], "ref_bytes": len(data),
                "description": description if description is not None else (prev or {}).get("description"),
                "mode": mode or (prev or {}).get("mode") or "character",
                "weight": weight if weight is not None else (prev or {}).get("weight", 0.8),
                "source": source, "dims": dims,
                "created_at": (prev or {}).get("created_at") or _now(), "updated_at": _now(),
            }
            self._chars[k] = entry
            self._flush()
        return self._public(entry)

    def add_reference(self, *, name, data, ext="png") -> dict:
        """Append another reference image to an existing character (more angles = stronger likeness)."""
        k = _key(name)
        with self._lock:
            e = self._chars.get(k)
            if not e:
                raise CharacterError(f"No saved character named '{name}' — save_character it first")
            ref_file = self._write_ref(data, ext)
            e["ref_files"] = _ref_files(e) + [ref_file]  # also migrates legacy ref_file
            e.pop("ref_file", None)
            e["updated_at"] = _now()
            self._flush()
            return self._public(e)

    def get(self, name: str):
        """Internal entry (includes ref_files) or None."""
        with self._lock:
            e = self._chars.get(_key(name))
        return dict(e) if e else None

    def read_references(self, name: str):
        """Return ([ref_bytes, ...], entry) for a saved character. Raises if missing/empty."""
        e = self.get(name)
        if not e:
            avail = ", ".join(sorted(c["name"] for c in self._chars.values())) or "(none)"
            raise CharacterError(f"No saved character named '{name}'. Saved: {avail}")
        blobs = []
        for rf in _ref_files(e):
            if os.path.isfile(rf):
                with open(rf, "rb") as f:
                    blobs.append(f.read())
        if not blobs:
            raise CharacterError(f"Character '{name}' has no reference images on disk")
        return blobs, e

    def list(self) -> list:
        with self._lock:
            return [self._public(e) for e in self._chars.values()]

    def update(self, *, name, new_name=None, description=None, mode=None, weight=None) -> dict:
        """Edit a character's metadata in place (rename / description / mode / weight). References untouched."""
        k = _key(name)
        with self._lock:
            e = self._chars.get(k)
            if not e:
                raise CharacterError(f"No saved character named '{name}'")
            if description is not None:
                e["description"] = description
            if mode is not None:
                e["mode"] = mode
            if weight is not None:
                e["weight"] = weight
            if new_name is not None and new_name.strip():
                nk = _key(new_name)
                if nk != k and nk in self._chars:
                    raise CharacterError(f"A character named '{new_name}' already exists")
                e["name"] = new_name.strip()
                e["key"] = nk
                if nk != k:
                    del self._chars[k]
                    self._chars[nk] = e
            e["updated_at"] = _now()
            self._flush()
            return self._public(e)

    def delete(self, name: str) -> dict:
        with self._lock:
            e = self._chars.pop(_key(name), None)
            if not e:
                raise CharacterError(f"No saved character named '{name}'")
            for rf in _ref_files(e):
                if os.path.isfile(rf):
                    try:
                        os.remove(rf)
                    except OSError:
                        pass
            self._flush()
        return {"deleted": e.get("name")}
