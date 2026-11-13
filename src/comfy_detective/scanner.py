"""Read-only scan of a ComfyUI installation.

Collects only what can be observed from the filesystem (plus `git status`
when available). Partial data is recorded honestly: anything skipped or
unreadable lands in `unavailable`/`warnings`, never silently as "unchanged".
"""

from __future__ import annotations

import datetime as _dt
import glob as _glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .models import (CustomNode, DeclaredFile, Environment, GitInfo,
                     PackageInfo, ScanResult, Unavailable, to_dict)
from .util import ToolError, git_info, read_text
from .workflow import parse_workflow, resolve_workflow_models

VENV_CANDIDATES = ("venv", ".venv", "python-venv", "env")
_VERSION_PY_RE = re.compile(r"""__version__\s*=\s*["']([^"']+)["']""")
_VERSION_TOML_RE = re.compile(r"""^version\s*=\s*["']([^"']+)["']""", re.M)

_NOT_CATEGORY_KEYS = {
    "base_path", "is_default", "custom_nodes", "output", "input", "temp",
    "user", "download_model_base", "downloader_enabled", "localhost",
    "listen", "port", "base_url", "auth_token",
}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def scan_install(comfyui_root: str, workflow_paths: Tuple[str, ...] = (),
                 hash_models: bool = False, allow_git_cmd: bool = True
                 ) -> ScanResult:
    root = os.path.abspath(os.path.expanduser(comfyui_root))
    if not os.path.isdir(root):
        raise ToolError("ComfyUI root is not a directory: %s" % root, code=4)
    result = ScanResult(comfyui_root=root, scan_time=_utc_now())

    _scan_version(result, root)
    result.comfy_git = _git_to_model(git_info(root, allow_git_cmd))

    _scan_environment(result, root)
    _scan_packages(result)
    _scan_declared_files(result, root)
    _scan_custom_nodes(result, root, allow_git_cmd)
    result.model_dirs = collect_model_dirs(root, result)

    for wf_path in workflow_paths:
        wf = parse_workflow(wf_path)
        resolve_workflow_models(wf, result.model_dirs, hash_models=hash_models)
        result.workflows.append(wf)
    return result


# --------------------------------------------------------------------------

def _add_unavailable(result: ScanResult, field: str, reason: str) -> None:
    result.unavailable.append(Unavailable(field=field, reason=reason))


def _git_to_model(info: Dict[str, Any]) -> GitInfo:
    return GitInfo(repo_path=info["repo_path"], is_git=info["is_git"],
                   commit=info["commit"], branch=info["branch"],
                   detached=info["detached"], dirty=info["dirty"],
                   notes=list(info["notes"]))


def _scan_version(result: ScanResult, root: str) -> None:
    ver_path = os.path.join(root, "comfyui_version.py")
    text, err = read_text(ver_path, 4096)
    if text:
        m = _VERSION_PY_RE.search(text)
        if m:
            result.comfy_version = m.group(1)
            return
    pj = os.path.join(root, "pyproject.toml")
    text, _ = read_text(pj, 256 * 1024)
    if text:
        m = _VERSION_TOML_RE.search(text)
        if m:
            result.comfy_version = m.group(1)
            result.warnings.append(
                "version taken from pyproject.toml (no comfyui_version.py)")
            return
    _add_unavailable(result, "comfyui.version",
                     "no comfyui_version.py/pyproject version found (%s)"
                     % (err or "file missing"))


def _scan_environment(result: ScanResult, root: str) -> None:
    env = Environment()
    for cand in VENV_CANDIDATES:
        venv = os.path.join(root, cand)
        exe = _find_venv_python(venv)
        if exe:
            env.venv_dir = venv
            env.python_exe = exe
            break
    if env.venv_dir is None:
        _add_unavailable(
            result, "environment",
            "no python venv found under the ComfyUI root (looked for: %s). "
            "Package list therefore unavailable." % ", ".join(VENV_CANDIDATES))
        result.environment = env
        return
    cfg_path = os.path.join(env.venv_dir, "pyvenv.cfg")
    text, err = read_text(cfg_path, 8192)
    if text:
        for line in text.splitlines():
            # pyvenv.cfg is a "key = value" file (per CPython venv docs)
            if "=" in line:
                k, _, v = line.partition("=")
                env.pyvenv[k.strip().lower()] = v.strip()
    else:
        env.pyvenv = {"_note": "pyvenv.cfg: %s" % (err or "missing")}
    sp = os.path.join(env.venv_dir, "Lib", "site-packages")
    if not os.path.isdir(sp):  # posix layout
        cands = sorted(_glob.glob(
            os.path.join(env.venv_dir, "lib", "python*", "site-packages")))
        sp = cands[0] if cands else sp
    env.site_packages = sp if os.path.isdir(sp) else None
    if env.site_packages is None:
        _add_unavailable(result, "packages",
                         "venv found but no site-packages directory")
    result.environment = env


def _find_venv_python(venv_dir: str) -> Optional[str]:
    for exe in (os.path.join(venv_dir, "Scripts", "python.exe"),
                os.path.join(venv_dir, "bin", "python"),
                os.path.join(venv_dir, "bin", "python3")):
        if os.path.isfile(exe):
            return exe
    return None


def _scan_packages(result: ScanResult) -> None:
    sp = result.environment.site_packages
    if not sp:
        if not any(u.field == "environment" for u in result.unavailable):
            _add_unavailable(result, "packages", "no site-packages location")
        return
    packages: List[PackageInfo] = []
    unreadable = 0
    try:
        entries = sorted(os.scandir(sp), key=lambda e: e.name.lower())
    except OSError as exc:
        _add_unavailable(result, "packages",
                         "site-packages unreadable: %s" % exc)
        return
    for ent in entries:
        base = ent.name
        if base.endswith(".dist-info") or base.endswith(".egg-info"):
            pkg = _dist_from_dir(ent.path, base)
            if pkg:
                packages.append(pkg)
            else:
                unreadable += 1
    packages.sort(key=lambda p: (p.name.lower(), p.version.lower()))
    result.packages = packages
    if unreadable:
        result.warnings.append(
            "%d distribution metadata dir(s) could not be parsed" % unreadable)


def _dist_from_dir(directory: str, base: str) -> Optional[PackageInfo]:
    for meta_name in ("METADATA", "PKG-INFO"):
        text, _ = read_text(os.path.join(directory, meta_name), 64 * 1024)
        if text is None:
            continue
        name = version = None
        for line in text.splitlines():
            if line == "":
                break  # metadata headers end at first blank line
            if name is None and line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif version is None and line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
            if name and version:
                break
        if name and version:
            return PackageInfo(name=name, version=version, source="dist-info")
    stem = re.sub(r"\.(dist|egg)-info$", "", base)
    m = re.match(r"^(.+?)-(\d[^-]*)(?:-.*)?$", stem)
    if m:
        return PackageInfo(name=m.group(1), version=m.group(2),
                           source="dist-info-dirname")
    return None


def _scan_declared_files(result: ScanResult, root: str) -> None:
    for fname, role in (("requirements.txt", "requirements"),
                        ("pyproject.toml", "pyproject"),
                        ("extra_model_paths.yaml", "extra_model_paths")):
        p = os.path.join(root, fname)
        result.declared_files.append(
            DeclaredFile(path=p, exists=os.path.isfile(p), role=role))


def _scan_custom_nodes(result: ScanResult, root: str,
                       allow_git_cmd: bool) -> None:
    cn = os.path.join(root, "custom_nodes")
    if not os.path.isdir(cn):
        _add_unavailable(result, "custom_nodes",
                         "no custom_nodes directory: %s" % cn)
        return
    try:
        entries = sorted(os.scandir(cn), key=lambda e: e.name.lower())
    except OSError as exc:
        _add_unavailable(result, "custom_nodes",
                         "custom_nodes unreadable: %s" % exc)
        return
    hidden = 0
    nodes: List[CustomNode] = []
    for ent in entries:
        name = ent.name
        if name.startswith(".") or name.startswith("__"):
            hidden += 1
            continue
        try:
            is_dir = ent.is_dir(follow_symlinks=False)
            is_file = ent.is_file(follow_symlinks=False)
        except OSError:
            continue
        if is_dir:
            node = CustomNode(name=name, path=ent.path, kind="dir")
            node.git = _git_to_model(git_info(ent.path, allow_git_cmd))
            if not node.git.is_git:
                node.notes.append("not a git checkout")
            req = os.path.join(ent.path, "requirements.txt")
            if os.path.isfile(req):
                node.requirements = req
            nodes.append(node)
        elif is_file and name.endswith(".py"):
            nodes.append(CustomNode(name=name[:-3], path=ent.path, kind="file",
                                    notes=["single-file node (not git-tracked)"]))
    result.custom_nodes = nodes
    if hidden:
        result.warnings.append(
            "%d hidden entry(ies) skipped under custom_nodes (ComfyUI does "
            "not load dot-directories such as .disabled)" % hidden)


# --------------------------------------------------------------------------
# model directories: standard layout + extra_model_paths.yaml
# --------------------------------------------------------------------------

def collect_model_dirs(root: str,
                       result: Optional[ScanResult] = None
                       ) -> Dict[str, List[str]]:
    dirs: Dict[str, List[str]] = {}
    models = os.path.join(root, "models")
    if os.path.isdir(models):
        try:
            for ent in os.scandir(models):
                if ent.is_dir(follow_symlinks=False) and not ent.name.startswith("."):
                    dirs.setdefault(ent.name.lower(), []).append(ent.path)
        except OSError:
            if result is not None:
                result.warnings.append("models directory unreadable")
    extra = os.path.join(root, "extra_model_paths.yaml")
    if os.path.isfile(extra):
        parsed, notes = parse_extra_model_paths(extra, root)
        for cat, paths in parsed.items():
            dirs.setdefault(cat.lower(), []).extend(paths)
        for note in notes:
            if result is not None:
                result.warnings.append(note)
    elif result is not None:
        result.warnings.append(
            "no extra_model_paths.yaml; only standard <root>/models "
            "directories are searched")
    # dedupe, keep order
    for cat in dirs:
        seen = set()
        kept = []
        for d in dirs[cat]:
            if d.lower() not in seen:
                seen.add(d.lower())
                if os.path.isdir(d):
                    kept.append(d)
        if kept:
            dirs[cat] = kept
    return {k: v for k, v in dirs.items() if v}


_FLOW_SECTION_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*\{([^{}]*)\}')
_PAIR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*"([^"]*)"')


def parse_extra_model_paths(path: str,
                            comfy_root: Optional[str] = None
                            ) -> Tuple[Dict[str, List[str]], List[str]]:
    """Conservative parser for both flow-mapping (Stability Matrix style)
    and block-style extra_model_paths.yaml. Returns (category->dirs, notes).
    Unparseable lines are skipped with a note, not an error."""
    notes: List[str] = []
    result: Dict[str, List[str]] = {}
    text, err = read_text(path, 256 * 1024)
    if text is None:
        return {}, ["extra_model_paths.yaml unreadable: %s" % err]

    def add(cat: str, raw: str, base: Optional[str]) -> None:
        for piece in raw.replace("\\n", "\n").split("\n"):
            piece = piece.strip()
            if not piece:
                continue
            resolved = piece.replace("/", os.sep)
            if not os.path.isabs(resolved):
                anchor = base or comfy_root or os.path.dirname(path)
                resolved = os.path.normpath(os.path.join(anchor, resolved))
            result.setdefault(cat.strip().lower(), []).append(resolved)

    sections = _FLOW_SECTION_RE.findall(text)
    if sections:
        for _name, body in sections:
            base = None
            bp = re.search(r'base_path\s*:\s*"([^"]*)"', body)
            if bp:
                base = bp.group(1)
            for key, val in _PAIR_RE.findall(body):
                if key.lower() in _NOT_CATEGORY_KEYS:
                    continue
                add(key, val, base)
        return result, notes

    # block style
    current_base: Optional[str] = None
    in_section = False
    unparsed = 0
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        indented = raw_line[:1] in (" ", "\t")
        line = raw_line.strip()
        if ":" not in line:
            unparsed += 1
            continue
        key, _, val = line.partition(":")
        key = key.strip().strip('"\'')
        val = val.strip().strip('"\'')
        if not indented:
            in_section = bool(key) and not val
            current_base = None
            if not in_section:
                unparsed += 1
            continue
        if not in_section:
            unparsed += 1
            continue
        if key.lower() == "base_path":
            current_base = val
            continue
        if key.lower() in _NOT_CATEGORY_KEYS or not val:
            continue
        add(key, val, current_base)
    if unparsed:
        notes.append("extra_model_paths.yaml: %d line(s) not understood; "
                     "model search locations may be incomplete" % unparsed)
    return result, notes
