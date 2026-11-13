"""Snapshot storage: create / list / resolve / load, all local and atomic.

Default store: per-user app-data directory (see util.default_store_dir),
overridable with --store or COMFY_DETECTIVE_STORE. Never inside a ComfyUI
installation unless the user chooses that path.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Tuple

from . import SNAPSHOT_SCHEMA_VERSION, TOOL_VERSION
from .models import ScanResult, to_dict
from .util import ToolError, atomic_write_json, read_text

MAX_SUPPORTED_SCHEMA = SNAPSHOT_SCHEMA_VERSION


def new_snapshot_id() -> str:
    return "%s-%s" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()),
                      secrets.token_hex(2))


def create_snapshot(store_dir: str, scan: ScanResult,
                    name: Optional[str] = None) -> Tuple[str, str]:
    store_dir = os.path.abspath(os.path.expanduser(store_dir))
    snapshot_id = new_snapshot_id()
    path = os.path.join(store_dir, snapshot_id + ".json")
    if os.path.exists(path):  # random suffix collision; refuse, never overwrite
        raise ToolError("snapshot id collision (%s); retry" % path, code=3)
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "name": name or None,
        "created_at": scan.scan_time,
        "tool_version": TOOL_VERSION,
        "scan": to_dict(scan),
    }
    try:
        atomic_write_json(path, payload)
    except OSError as exc:
        raise ToolError("could not write snapshot to %s: %s" % (store_dir, exc),
                        code=3)
    return snapshot_id, path


def list_snapshots(store_dir: str) -> List[Dict[str, Any]]:
    store_dir = os.path.abspath(os.path.expanduser(store_dir))
    entries: List[Dict[str, Any]] = []
    if not os.path.isdir(store_dir):
        return entries
    try:
        names = sorted(n for n in os.listdir(store_dir) if n.endswith(".json"))
    except OSError as exc:
        raise ToolError("snapshot store unreadable (%s): %s" % (store_dir, exc),
                        code=3)
    for n in names:
        path = os.path.join(store_dir, n)
        entry: Dict[str, Any] = {
            "file": path,
            "snapshot_id": os.path.splitext(n)[0],
            "name": None, "created_at": None, "comfyui_root": None,
            "valid": True, "error": None,
        }
        data, err = _read_snapshot_file(path)
        if data is None:
            entry["valid"] = False
            entry["error"] = err
        else:
            entry["snapshot_id"] = data.get("snapshot_id", entry["snapshot_id"])
            entry["name"] = data.get("name")
            entry["created_at"] = data.get("created_at")
            entry["comfyui_root"] = (data.get("scan") or {}).get("comfyui_root")
        entries.append(entry)
    entries.sort(key=lambda e: (e.get("created_at") or "", e["snapshot_id"]))
    return entries


def _read_snapshot_file(path: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text, err = read_text(path, 64 * 1024 * 1024)
    if text is None:
        return None, err or "unreadable"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "invalid JSON (line %d, column %d): %s" % (
            exc.lineno, exc.colno, exc.msg)
    if not isinstance(data, dict):
        return None, "snapshot must be a JSON object"
    sv = data.get("schema_version")
    if not isinstance(sv, int):
        return None, "missing or non-integer schema_version"
    if sv > MAX_SUPPORTED_SCHEMA:
        return None, ("snapshot schema_version %d is newer than this tool "
                      "supports (%d); upgrade comfy-detective"
                      % (sv, MAX_SUPPORTED_SCHEMA))
    if not isinstance(data.get("scan"), dict):
        return None, "missing 'scan' object"
    return data, None


def load_snapshot_file(path: str) -> Dict[str, Any]:
    data, err = _read_snapshot_file(path)
    if data is None:
        raise ToolError("invalid snapshot %s: %s" % (path, err), code=3)
    return data


def resolve_snapshot(store_dir: str, ident: str) -> Dict[str, Any]:
    """Resolve by exact id, exact name, or unique id prefix."""
    store_dir = os.path.abspath(os.path.expanduser(store_dir))
    if not os.path.isdir(store_dir):
        raise ToolError("snapshot store does not exist: %s\n"
                        "Create a snapshot first: comfy-detective snapshot "
                        "create --comfyui <path>" % store_dir, code=3)
    entries = list_snapshots(store_dir)
    valid = [e for e in entries if e["valid"]]
    stem = os.path.splitext(os.path.basename(ident))[0]

    def by_id(e):
        return e["snapshot_id"] == stem
    matches = [e for e in valid if by_id(e)]
    if not matches:
        matches = [e for e in valid
                   if e.get("name") and e["name"].casefold() == ident.casefold()]
    if not matches:
        matches = [e for e in valid if e["snapshot_id"].startswith(stem)]
    if not matches:
        available = ", ".join(sorted(e["snapshot_id"] for e in valid)) or "none"
        raise ToolError("snapshot not found: %s\navailable: %s" % (ident, available),
                        code=3)
    if len(matches) > 1:
        raise ToolError("ambiguous snapshot reference '%s': matches %s"
                        % (ident, ", ".join(e["snapshot_id"] for e in matches)),
                        code=3)
    return load_snapshot_file(matches[0]["file"])
