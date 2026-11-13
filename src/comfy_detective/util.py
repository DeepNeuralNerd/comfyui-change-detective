"""Shared helpers: paths, atomic JSON writes, file-based Git info, versions.

Everything here is read-only with respect to the scanned installation. The only
processes ever started are ``git status --porcelain`` (for dirty-worktree
detection) and nothing else; when git is unavailable the affected value is
reported as unknown, never guessed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, Optional


class ToolError(Exception):
    """User-facing error carrying a meaningful process exit code.

    Exit codes: 2 usage (argparse), 3 snapshot problems, 4 invalid ComfyUI
    root, 6 unreadable/unparsable user file (workflow/log), 1 unexpected.
    """

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# paths / files
# --------------------------------------------------------------------------

def norm_key(path: str) -> str:
    """Normalized, comparable path key (case- and separator-insensitive)."""
    try:
        return os.path.normcase(os.path.normpath(os.path.abspath(path)))
    except (OSError, ValueError):
        return os.path.normcase(os.path.normpath(path))


def read_text(path: str, max_bytes: int = 2_000_000) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, error). Never raises for missing/unreadable files."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(max_bytes + 1)
    except FileNotFoundError:
        return None, "file not found"
    except NotADirectoryError:
        return None, "path is not a directory"
    except OSError as exc:
        return None, "could not read: %s" % exc
    truncated = len(data) > max_bytes
    text = data[:max_bytes].decode("utf-8", errors="replace")
    if truncated:
        text += "\n...[truncated at %d bytes]" % max_bytes
    return text, None


def file_identity(path: str) -> Optional[Dict[str, Any]]:
    """Path, size, mtime (whole seconds). No hashing."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"path": path, "size": st.st_size, "mtime": int(st.st_mtime)}


def sha256_file(path: str) -> Optional[str]:
    import hashlib

    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def atomic_write_json(path: str, obj: Any) -> None:
    """Write JSON atomically: temp file in the same directory, then replace."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# git, read from .git files first (no subprocess, no network); `git status`
# only for dirty state and only when the binary cooperates.
# --------------------------------------------------------------------------

_HEX_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _first_line(path: str) -> Optional[str]:
    text, err = read_text(path, 4096)
    if text is None:
        return None
    line = text.splitlines()[0].strip() if text.splitlines() else ""
    return line or None


def _resolve_ref(gitdir: str, ref: str) -> Optional[str]:
    candidate = os.path.join(gitdir, *ref.split("/"))
    if os.path.isfile(candidate):
        sha = _first_line(candidate)
        if sha and _HEX_RE.match(sha):
            return sha
    packed = os.path.join(gitdir, "packed-refs")
    if os.path.isfile(packed):
        text, _ = read_text(packed, 4 * 1024 * 1024)
        if text:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[1].strip() == ref and _HEX_RE.match(parts[0]):
                    return parts[0]
    return None


def git_info(repo_path: str, allow_git_cmd: bool = True) -> Dict[str, Any]:
    """File-based Git facts for a directory. Never raises.

    Keys: repo_path, is_git, commit, branch, detached, dirty (None=unknown),
    notes (why something is unknown).
    """
    info: Dict[str, Any] = {
        "repo_path": repo_path,
        "is_git": False,
        "commit": None,
        "branch": None,
        "detached": False,
        "dirty": None,
        "notes": [],
    }
    if not os.path.isdir(repo_path):
        info["notes"].append("directory does not exist")
        return info
    gitdir = os.path.join(repo_path, ".git")
    if os.path.isfile(gitdir):
        text, _ = read_text(gitdir, 4096)
        if text and text.lstrip().startswith("gitdir:"):
            target = text.split(":", 1)[1].strip()
            gitdir = target if os.path.isabs(target) else os.path.normpath(
                os.path.join(repo_path, target))
        else:
            info["notes"].append(".git file present but unreadable/unsupported")
            return info
    if not os.path.isdir(gitdir):
        info["notes"].append("no .git directory (not a git checkout)")
        return info
    info["is_git"] = True

    head_line = _first_line(os.path.join(gitdir, "HEAD"))
    if head_line is None:
        info["notes"].append("HEAD missing or unreadable")
    elif head_line.startswith("ref:"):
        ref = head_line[4:].strip()
        if ref.startswith("refs/heads/"):
            info["branch"] = ref[len("refs/heads/"):]
        else:
            info["branch"] = ref
        sha = _resolve_ref(gitdir, ref)
        if sha:
            info["commit"] = sha
        else:
            info["notes"].append(
                "branch ref '%s' not found (unborn branch or missing refs)" % ref)
    elif _HEX_RE.match(head_line):
        info["detached"] = True
        info["commit"] = head_line
        info["notes"].append("detached HEAD")
    else:
        info["notes"].append("malformed HEAD file")

    if allow_git_cmd and info["commit"] is not None:
        dirty = _git_dirty(repo_path)
        if dirty is None:
            info["notes"].append(
                "dirty state unknown: could not run 'git status' (git missing, "
                "sandbox, or timeout)")
        else:
            info["dirty"] = dirty
    return info


def _git_dirty(repo_path: str) -> Optional[bool]:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True, text=True, timeout=20,
            stdin=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001 - any failure means "unknown", never "clean"
        return None
    if proc.returncode != 0:
        return None
    return bool(proc.stdout.strip())


# --------------------------------------------------------------------------
# versions / package names
# --------------------------------------------------------------------------

def canonical_pkg_name(name: str) -> str:
    """PEP 503 style canonical name for comparing declared vs installed."""
    return re.sub(r"[-_.]+", "-", name).lower()


_INF = float("inf")

_PRE_ALIASES = {"a": 0, "alpha": 0, "b": 1, "beta": 1,
                "c": 2, "rc": 2, "pre": 2, "preview": 2}

_VERSION_RE = re.compile(
    r"^(?:(?P<epoch>\d+)!)?v?"
    r"(?P<release>\d+(?:\.\d+)*)"
    r"(?:[-_.]?(?P<pre>a|b|c|rc|alpha|beta|pre|preview)"
    r"[-_.]?(?P<pre_n>\d+)?)?"
    r"(?:[-_.]?(?P<post>post|rev|r)[-_.]?(?P<post_n>\d+)?"
    r"|[-_.](?P<post_d>\d+))?"
    r"(?:[-_.]?(?P<dev>dev)[-_.]?(?P<dev_n>\d+)?)?"
    r"(?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?$",
    re.IGNORECASE)


def version_tuple(v: str) -> Optional[tuple]:
    """PEP 440 sort key: (epoch, release, pre_key, post_key, dev_key).

    Follows the canonical PEP 440 ordering: dev sorts BEFORE the release it
    attaches to (1.0.dev1 < 1.0, 1.0.post1.dev1 < 1.0.post1,
    1.0rc1.dev1 < 1.0rc1), aN/bN/rcN < final < postN, and missing parts use
    +/- infinity sentinels. None (not comparable) when unparseable. A local
    "+label" is accepted but ignored (it refines equality only).
    """
    if not v:
        return None
    m = _VERSION_RE.match(v.strip())
    if not m:
        return None
    epoch = int(m.group("epoch") or 0)
    release = tuple(int(x) for x in m.group("release").split("."))
    pre = m.group("pre")
    has_post = m.group("post") is not None or m.group("post_d") is not None
    dev = m.group("dev") is not None
    if pre:
        pre_key = (_PRE_ALIASES[pre.lower()], int(m.group("pre_n") or 0))
    elif dev and not has_post:
        # canonical PEP 440: a pure dev release precedes every pre-release
        pre_key = (-_INF, 0)
    else:
        pre_key = (_INF, 0)
    post_key = ((int(m.group("post_n") or m.group("post_d") or 0), 0)
                if has_post else (-_INF, 0))
    dev_key = ((int(m.group("dev_n") or 0), 0) if dev else (_INF, 0))
    return (epoch, release, pre_key, post_key, dev_key)


def compare_versions(a: str, b: str) -> Optional[int]:
    """-1/0/1 for a<b, a==b, a>b; None when not comparable."""
    ta, tb = version_tuple(a), version_tuple(b)
    if ta is None or tb is None:
        return None
    width = max(len(ta[1]), len(tb[1]))

    def pad(key: tuple) -> tuple:
        rel = list(key[1])
        rel += [0] * (width - len(rel))
        return (key[0], tuple(rel)) + key[2:]

    ka, kb = pad(ta), pad(tb)
    return (ka > kb) - (ka < kb)


def default_store_dir() -> str:
    """Snapshot store: env override, else per-OS app-data location."""
    env = os.environ.get("COMFY_DETECTIVE_STORE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
        return os.path.join(base, "ComfyUIChangeDetective", "snapshots")
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(xdg, "comfyui-change-detective", "snapshots")
