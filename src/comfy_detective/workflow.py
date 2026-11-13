"""Workflow parsing, static node inventory, and conservative model resolution.

Conservatism rules (they define the module's trustworthiness):
- Workflow content is untrusted input: never executed, malformed JSON yields
  a description, never a traceback.
- A model reference is reported "missing" only when the guessed category has
  readable candidate directories that were searched exhaustively within a
  bounded budget. Anything else is "unknown", with the searched locations.
- Node classes are checked only against a statically built inventory; files
  whose NODE_CLASS_MAPPINGS cannot be read statically lower coverage and are
  reported as unknown coverage, never as missing nodes.
"""

from __future__ import annotations

import ast
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Finding, ModelRef, WorkflowInfo
from .util import file_identity, norm_key, read_text, sha256_file

MODEL_EXTS = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".onnx", ".gguf", ".sks",
    ".pt2", ".pkl",
)

# High-confidence input-name -> category mappings used by common loaders.
MODEL_INPUT_EXACT = {
    "ckpt_name": "checkpoints",
    "ckpt_dir_name": "checkpoints",
    "unet_name": "unet",
    "unet_name_input": "unet",
    "diffusion_model": "unet",
    "vae_name": "vae",
    "vae_name_opt": "vae",
    "clip_name": "clip",
    "clip_name1": "clip",
    "clip_name2": "clip",
    "control_net_name": "controlnet",
    "style_model_name": "style_models",
    "style_models_name": "style_models",
    "vae_approx_name": "vae_approx",
    "gligen_name": "gligen",
    "lora_name": "loras",
    "lora_a_name": "loras",
    "lora_b_name": "loras",
    "embedding_name": "embeddings",
    "hypernetwork_name": "hypernetworks",
    "upscale_name": "upscale_models",
    "sams_name": "sams",
}

# Keyword heuristic: name hint -> category (only applied to values that look
# like model file names, i.e. end with MODEL_EXTS).
MODEL_KEYWORD_HINTS = [
    ("lora", "loras"), ("loras", "loras"),
    ("ckpt", "checkpoints"), ("checkpoint", "checkpoints"),
    ("unet", "unet"), ("diffusion_model", "unet"), ("diffusion", "unet"),
    ("vae", "vae"), ("clip_vision", "clip_vision"), ("clip", "clip"),
    ("controlnet", "controlnet"), ("control_net", "controlnet"),
    ("embedding", "embeddings"), ("gligen", "gligen"),
    ("upscale", "upscale_models"), ("esrgan", "upscale_models"),
    ("hypernetwork", "hypernetworks"),
]

# Category -> categories to search (folder_paths aliases differ across versions).
CATEGORY_ALIASES = {
    "checkpoints": ["checkpoints", "diffusion_models"],
    "unet": ["unet", "diffusion_models"],
    "diffusion_models": ["diffusion_models", "unet"],
    "loras": ["loras"],
    "vae": ["vae"],
    "clip": ["clip", "clip_vision"],
    "clip_vision": ["clip_vision", "clip"],
    "style_models": ["style_models", "clip_vision", "clip"],
    "controlnet": ["controlnet"],
    "embeddings": ["embeddings"],
    "upscale_models": ["upscale_models"],
    "vae_approx": ["vae_approx"],
    "gligen": ["gligen"],
    "hypernetworks": ["hypernetworks"],
    "sams": ["sams"],
}

# litegraph pseudo-nodes that are never execution node classes.
PSEUDO_NODE_TYPES = {"Group", "Note", "MarkdownNote"}

# search budgets (avoid unbounded traversal on slow/large model roots)
_DIR_ENTRY_CAP = 600
_MAX_SEARCH_DIRS = 12
_TOTAL_SCAN_CAP = 4000

_WILDCARD_CHARS = ("*", "?", "[")

# file-name suffixes typical for model weights (used for unlabeled
# subgraph widget values; never claims "missing" on its own)
_MODEL_FILE_RE = re.compile(
    r"\.(safetensors|ckpt|pt|pth|bin|onnx|gguf|torchscript|lora|safetensors)$",
    re.IGNORECASE)


# --------------------------------------------------------------------------
# workflow parsing
# --------------------------------------------------------------------------

def parse_workflow(path: str) -> WorkflowInfo:
    wf = WorkflowInfo(path=os.path.abspath(path))
    text, err = read_text(path, 8 * 1024 * 1024)
    if text is None:
        wf.parse_error = "could not read workflow: %s" % err
        return wf
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        wf.parse_error = ("invalid JSON: %s (line %d, column %d)"
                          % (exc.msg, exc.lineno, exc.colno))
        return wf

    fmt = _detect_format(data)
    wf.format = fmt
    if fmt == "ui":
        nodes = data if isinstance(data, list) else data.get("nodes", [])
        seen: Set[str] = set()
        sub_map: Dict[str, dict] = {}
        if isinstance(data, dict) and isinstance(data.get("definitions"), dict):
            for s in data["definitions"].get("subgraphs") or []:
                if isinstance(s, dict) and isinstance(s.get("id"), str):
                    sub_map[s["id"]] = s
                    wf.subgraphs.append({"id": s["id"],
                                         "name": str(s.get("name") or "")})
        file_like: Set[str] = set()
        for node in nodes:
            if not isinstance(node, dict):
                continue
            wf.node_count += 1
            ntype = node.get("type")
            is_sub = isinstance(ntype, str) and ntype in sub_map
            if isinstance(ntype, str) and ntype and not is_sub:
                seen.add(ntype)
            inputs = node.get("inputs")
            if isinstance(inputs, list):
                for ent in inputs:
                    if isinstance(ent, dict) and ent.get("link") is not None:
                        wf.edge_count += 1
                    if isinstance(ent, dict):
                        _collect_ref(wf, ntype or "", ent.get("name"),
                                     ent.get("value"))
            if is_sub:
                # instance widget values are the real (overridden) settings
                for v in node.get("widgets_values") or []:
                    if isinstance(v, str):
                        t = v.strip()
                        if 0 < len(t) <= 500 and _MODEL_FILE_RE.search(t):
                            file_like.add(t)
        for sub in sub_map.values():
            for node in sub.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                ntype = node.get("type")
                if isinstance(ntype, str) and ntype and ntype not in sub_map:
                    seen.add(ntype)
                inputs = node.get("inputs")
                if isinstance(inputs, list):
                    for ent in inputs:
                        if isinstance(ent, dict):
                            _collect_ref(wf, ntype or "", ent.get("name"),
                                         ent.get("value"))
        wf.node_classes = sorted(seen)
        known_vals = {r.value for r in wf.model_refs}
        for val in sorted(file_like):
            if val in known_vals:
                continue
            wf.model_refs.append(ModelRef(
                value=val, node_type="subgraph-widget", input_name=None,
                category=None,
                notes=["value read from an unlabeled subgraph widget; "
                       "searched without category"]))
        if isinstance(data, dict) and isinstance(data.get("widgets_values"), list) \
                and any(isinstance(n, dict) and "widgets_values" in n for n in nodes):
            pass  # per-node positional widget settings: not interpretable
        if any(isinstance(n, dict) and "widgets_values" in n for n in nodes if isinstance(n, dict)):
            wf.coverage_notes.append(
                "widgets_values (positional widget settings) are stored without "
                "input names and are not interpreted; only labeled values were "
                "checked for model references.")
        if sub_map:
            wf.coverage_notes.append(
                "subgraphs: %d definition(s) parsed; instance widget values "
                "were searched as model-file candidates (category unknown), "
                "inner node types were included in the node check; inner "
                "placeholder defaults were not treated as references."
                % len(sub_map))
    elif fmt == "api":
        seen = set()
        for _nid, node in data.items():
            if not isinstance(node, dict):
                continue
            wf.node_count += 1
            ctype = node.get("class_type")
            if isinstance(ctype, str) and ctype:
                seen.add(ctype)
            inputs = node.get("inputs")
            if isinstance(inputs, dict):
                for name, value in inputs.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        wf.edge_count += 1
                    elif isinstance(value, list) and value and all(
                            isinstance(v, (int, float)) for v in value):
                        pass  # numeric widget list
                    _collect_ref(wf, ctype or "", name, value)
        wf.node_classes = sorted(seen)
    else:
        wf.coverage_notes.append(
            "unrecognized workflow structure: not a node list, a saved-graph "
            "object with 'nodes', nor an API prompt object with 'class_type'.")
    return wf


def _detect_format(data: Any) -> str:
    if isinstance(data, list) and data and all(isinstance(n, dict) for n in data):
        if any("type" in n for n in data):
            return "ui"
    if isinstance(data, dict):
        if isinstance(data.get("nodes"), list):
            return "ui"
        values = [v for v in data.values() if isinstance(v, dict)]
        if values and any("class_type" in v for v in values):
            return "api"
    return "unknown"


def _looks_like_model(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower().endswith(MODELS_EXT_TUPLE)


def _model_ext(value: str) -> str:
    low = value.strip().lower()
    for ext in MODEL_EXTS:
        if low.endswith(ext):
            return ext
    return ""


MODELS_EXT_TUPLE = tuple(MODEL_EXTS)


def _guess_category(input_name: Optional[str], value: str) -> Tuple[Optional[str], bool]:
    """Return (category, high_confidence). high_confidence only for the
    exact loader-name table; keyword guesses still only gate search order."""
    if not input_name:
        return None, False
    key = input_name.lower()
    if key in MODEL_INPUT_EXACT:
        return MODEL_INPUT_EXACT[key], True
    low = key
    for hint, cat in MODEL_KEYWORD_HINTS:
        if hint in low:
            return cat, False
    return None, False


def _collect_ref(wf: WorkflowInfo, node_type: str,
                 input_name: Any, value: Any) -> None:
    if not isinstance(value, str):
        return
    name = input_name if isinstance(input_name, str) else None
    key = (name or "").lower()
    high = key in MODEL_INPUT_EXACT
    if not _looks_like_model(value) and not high:
        return
    if not _looks_like_model(value) and high:
        # exact loader input whose value is not a known model extension:
        # could still be a model in another format; skip to avoid noise.
        return
    value = value.strip()
    if not value:
        return
    category, _ = _guess_category(name, value)
    ref = ModelRef(value=value, node_type=node_type, input_name=name,
                   category=category)
    if any(ch in value for ch in _WILDCARD_CHARS):
        ref.status = "unknown"
        ref.notes.append("reference contains wildcard/selector syntax; it "
                         "cannot be matched as a literal file name")
    elif _is_explicit_path(value):
        ref.status, ref.resolved_path = _resolve_absolute(value)
        if ref.status == "missing":
            ref.searched = [value]
    if "\\" in value or "/" in value:
        ref.notes.append("reference includes subfolders; matched relative to "
                         "the category directory")
    wf.model_refs.append(ref)
    if ref.status == "found":
        _attach_identity(ref)


def _is_explicit_path(value: str) -> bool:
    """Drive-absolute ("D:\\x"), UNC, or POSIX-root only. Category-relative
    references like "subfolder/model.safetensors" are NOT explicit paths."""
    if len(value) > 2 and value[1] == ":" and value[2] in "\\/":
        return True
    return value.startswith("\\\\") or value.startswith("/")


def _resolve_absolute(value: str) -> Tuple[str, Optional[str]]:
    if os.path.isfile(value):
        return "found", os.path.abspath(value)
    return "missing", None


def _attach_identity(ref: ModelRef) -> None:
    ident = file_identity(ref.resolved_path or ref.value)
    if ident:
        ref.size = ident["size"]
        ref.mtime = ident["mtime"]


# --------------------------------------------------------------------------
# model resolution against discovered model dirs
# --------------------------------------------------------------------------

def resolve_workflow_models(wf: WorkflowInfo, model_dirs: Dict[str, List[str]],
                            hash_models: bool = False) -> None:
    if wf.parse_error:
        return
    for ref in wf.model_refs:
        if ref.status in ("found", "missing") and ref.resolved_path:
            continue  # absolute refs already decided
        if ref.status == "unknown":
            continue
        _resolve_ref_against_dirs(ref, model_dirs)
        if ref.status == "found" and hash_models and ref.resolved_path:
            digest = sha256_file(ref.resolved_path)
            if digest:
                ref.sha256 = digest
            else:
                ref.notes.append("hashing failed (file unreadable)")


def _candidate_dirs(ref: ModelRef, model_dirs: Dict[str, List[str]]
                    ) -> Tuple[List[str], bool]:
    """(dirs, category_known). Deduplicates by normalized key."""
    cats = CATEGORY_ALIASES.get(ref.category or "", [ref.category] if ref.category else [])
    dirs: List[str] = []
    seen: Set[str] = set()
    category_known = bool(ref.category)
    keys = cats if category_known else list(model_dirs.keys())
    for cat in keys:
        for d in model_dirs.get(cat, []):
            k = norm_key(d)
            if k not in seen:
                seen.add(k)
                dirs.append(d)
    return dirs, category_known   # full list; callers cap deep search only


def _resolve_ref_against_dirs(ref: ModelRef, model_dirs: Dict[str, List[str]]) -> None:
    dirs, category_known = _candidate_dirs(ref, model_dirs)
    if not dirs:
        ref.status = "unknown"
        ref.notes.append("no candidate model directories are known for category "
                         "'%s'; searched locations could not be checked"
                         % (ref.category or "unknown"))
        return
    ref.searched = list(dirs)
    matches: List[str] = []
    normed = ref.value.replace("/", os.sep).replace("\\", os.sep)
    has_subpath = ("/" in ref.value or "\\" in ref.value)
    want_rel = os.path.normcase(os.path.normpath(normed)) if has_subpath else None
    want_base = os.path.normcase(os.path.basename(normed))
    # 1) exact joins are cheap: run them against EVERY candidate dir so the
    #    deep-search cap can never hide a real match.
    for d in dirs:
        exact = os.path.join(d, *normed.split(os.sep))
        if os.path.isfile(exact):
            matches.append(exact)
    # 2) no exact hit: bounded deep search (entry budget + dir cap).
    budget_left = _TOTAL_SCAN_CAP
    all_searchable = True
    if not matches:
        for d in dirs[:_MAX_SEARCH_DIRS]:
            if budget_left <= 0:
                all_searchable = False
                continue
            found, used, readable = _shallow_search(d, want_rel if has_subpath
                                                    else None,
                                                    want_base, budget_left)
            budget_left -= used
            matches.extend(found)
            if not readable:
                all_searchable = False
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        ref.status = "found"
        ref.resolved_path = matches[0]
        _attach_identity(ref)
    elif len(matches) > 1:
        ref.status = "ambiguous"
        ref.notes.append("%d matching files in searched locations; not guessing"
                         % len(matches))
    elif all_searchable:
        if category_known:
            ref.status = "missing"
            ref.notes.append("not present in the searched directories for "
                             "category '%s' (searched directories are listed)"
                             % (ref.category or "unknown"))
        else:
            ref.status = "unknown"
            ref.notes.append("no match, but the model category was not "
                             "identifiable, so 'missing' is not claimed")
    else:
        ref.status = "unknown"
        ref.notes.append("search budget reached or some directories unreadable; "
                         "'missing' is not claimed")


def _shallow_search(directory: str, want_rel: Optional[str], want_base: str,
                    budget: int) -> Tuple[List[str], int, bool]:
    """Match by relative path (want_rel) or bare file name (want_base),
    bounded depth (2 sub-levels) and entry budget. Returns
    (matches, entries_used, fully_searchable)."""
    matches: List[str] = []
    used = 0
    stack: List[Tuple[str, str, int]] = [(directory, "", 0)]
    while stack:
        current, rel, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            return matches, used, False
        for ent in entries:
            used += 1
            if used > budget or used > _DIR_ENTRY_CAP * (depth + 1):
                return matches, used, False
            child_rel = (rel + "/" + ent.name) if rel else ent.name
            try:
                is_file = ent.is_file()
            except OSError:
                is_file = False
            if is_file:
                if want_rel is not None:
                    if os.path.normcase(child_rel.replace("/", os.sep)) == want_rel:
                        matches.append(ent.path)
                elif os.path.normcase(ent.name) == want_base:
                    matches.append(ent.path)
            elif depth < 2:
                stack.append((ent.path, child_rel, depth + 1))
        if used > budget:
            return matches, used, False
    return matches, used, True


# --------------------------------------------------------------------------
# static node inventory
# --------------------------------------------------------------------------

_INVENTORY_SKIP_DIRS = {".git", "__pycache__", "examples", "example", "tests",
                        "test", "docs", "venv", ".venv", ".github", ".ci",
                        "node_modules", "dist", "build"}
_MAX_INVENTORY_FILES = 3000
_MAX_PY_BYTES = 512 * 1024


def build_node_inventory(root: str, extra_objectinfo: Optional[str] = None
                         ) -> Dict[str, Any]:
    """Static NODE_CLASS_MAPPINGS scan of core + custom nodes.

    Returns {"types": set, "meta": {...}}; meta records coverage so unknown
    mappings can be reported as unknown rather than missing.
    """
    types: Set[str] = set()
    meta: Dict[str, Any] = {"files_parsed": 0, "files_failed": 0,
                            "dynamic_mapping_files": [], "files_scanned": 0,
                            "notes": [], "object_info_file": None}
    py_files: List[str] = []
    core_nodes = os.path.join(root, "nodes.py")
    if os.path.isfile(core_nodes):
        py_files.append(core_nodes)
    extras = os.path.join(root, "comfy_extras")
    if os.path.isdir(extras):
        py_files.extend(_list_py_files(extras, max_depth=1))
    cn = os.path.join(root, "custom_nodes")
    if os.path.isdir(cn):
        try:
            entries = sorted(os.scandir(cn), key=lambda e: e.name.lower())
        except OSError:
            entries = []
        for ent in entries:
            if ent.name.startswith(".") or ent.name.startswith("__"):
                continue
            try:
                if ent.is_dir(follow_symlinks=False):
                    py_files.extend(_list_py_files(ent.path, max_depth=3,
                                                   skip_dotted=True))
                elif ent.is_file(follow_symlinks=False) and ent.name.endswith(".py"):
                    py_files.append(ent.path)
            except OSError:
                continue

    for path in py_files[:_MAX_INVENTORY_FILES]:
        meta["files_scanned"] += 1
        got, status = _node_classes_in_file(path)
        if status == "ok":
            types.update(got)
            meta["files_parsed"] += 1
        elif status == "dynamic":
            types.update(got)
            meta["files_parsed"] += 1
            meta["dynamic_mapping_files"].append(_rel(root, path))
        else:
            meta["files_failed"] += 1
    if len(py_files) > _MAX_INVENTORY_FILES:
        meta["notes"].append("file cap reached: %d of %d candidate files parsed"
                             % (_MAX_INVENTORY_FILES, len(py_files)))

    if extra_objectinfo:
        got, err = _types_from_object_info(extra_objectinfo)
        if got is not None:
            types.update(got)
            meta["object_info_file"] = os.path.abspath(extra_objectinfo)
        else:
            meta["object_info_error"] = err
            meta["notes"].append("object-info file could not be used: %s" % err)
    return {"types": types, "meta": meta}


def _rel(root: str, path: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return path


def _list_py_files(directory: str, max_depth: int, skip_dotted: bool = False
                   ) -> List[str]:
    out: List[str] = []
    stack: List[Tuple[str, int]] = [(directory, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for ent in entries:
            try:
                if ent.is_dir(follow_symlinks=False):
                    if ent.name in _INVENTORY_SKIP_DIRS:
                        continue
                    if skip_dotted and ent.name.startswith("."):
                        continue
                    if depth < max_depth:
                        stack.append((ent.path, depth + 1))
                elif ent.is_file(follow_symlinks=False) and ent.name.endswith(".py"):
                    out.append(ent.path)
            except OSError:
                continue
    return out


_V3_API_MARKERS = ("comfy_entrypoint", "get_node_list")


def _node_classes_in_file(path: str) -> Tuple[Set[str], str]:
    """Parse one file statically for NODE_CLASS_MAPPINGS literal keys
    (assignment or .update() argument). Files using the new ComfyUI v3 API
    (comfy_entrypoint / get_node_list) register nodes dynamically and count
    as dynamic coverage.

    status: ok | dynamic (mappings not fully literal) | failed.
    """
    text, err = read_text(path, _MAX_PY_BYTES)
    if text is None:
        return set(), "failed"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return set(), "failed"
    types: Set[str] = set()
    dynamic = False
    for node in ast.walk(tree):
        targets: List[ast.expr] = []
        value: Optional[ast.expr] = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            base = t
            if isinstance(base, ast.Starred):
                continue
            name = getattr(base, "id", None)
            if name != "NODE_CLASS_MAPPINGS":
                continue
            if isinstance(value, ast.Dict):
                for k in value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        types.add(k.value)
                if any(k is None for k in value.keys) or \
                        any(isinstance(v, ast.Starred) for v in value.values):
                    dynamic = True
            else:
                dynamic = True
    for node in ast.walk(tree):  # NODE_CLASS_MAPPINGS.update({...})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "update" \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "NODE_CLASS_MAPPINGS" \
                and node.args and isinstance(node.args[0], ast.Dict):
            for k in node.args[0].keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    types.add(k.value)
    if any(marker in text for marker in _V3_API_MARKERS):
        dynamic = True
    if dynamic:
        return types, "dynamic"
    return types, "ok"


_MAX_OBJECTINFO_BYTES = 256 * 1024 * 1024


def _types_from_object_info(path: str) -> Tuple[Optional[Set[str]], str]:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return None, "unreadable: %s" % exc
    if size > _MAX_OBJECTINFO_BYTES:
        return None, "file is %d MB (cap %d MB)" % (
            size // (1024 * 1024), _MAX_OBJECTINFO_BYTES // (1024 * 1024))
    text, err = read_text(path, _MAX_OBJECTINFO_BYTES)
    if text is None:
        return None, err or "unreadable"
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, "invalid JSON (%s)" % exc.msg
    node_map = data.get("nodes", data) if isinstance(data, dict) else None
    if not isinstance(node_map, dict):
        return None, "no node mapping object found"
    return {k for k in node_map if isinstance(k, str)}, ""


# --------------------------------------------------------------------------
# workflow checks against an installation
# --------------------------------------------------------------------------

def workflow_findings(wf: WorkflowInfo, inventory: Optional[Dict[str, Any]]
                      ) -> List[Finding]:
    findings: List[Finding] = []
    if wf.parse_error:
        findings.append(Finding(
            title="Workflow could not be parsed",
            category="input", certainty="observed",
            evidence="%s: %s" % (wf.path, wf.parse_error),
            why_it_matters="No node or model analysis is possible for this file.",
            next_check="Re-save the workflow from the ComfyUI UI, or fix the "
                       "JSON syntax; re-run 'workflow check'.",
            refs=[wf.path]))
        return findings

    for note in wf.coverage_notes:
        findings.append(Finding(
            title="Workflow coverage note", category="workflow",
            certainty="unknown", evidence="%s: %s" % (wf.path, note),
            next_check="Treat undetected nodes/models as unverified."))

    if inventory is not None:
        known = inventory["types"]
        meta = inventory["meta"]
        missing = [t for t in wf.node_classes
                   if t not in known and t not in PSEUDO_NODE_TYPES]
        if missing:
            partial = bool(meta.get("dynamic_mapping_files")) or \
                bool(meta.get("files_failed")) or bool(meta.get("notes"))
            extra = ""
            if meta.get("object_info_file"):
                extra = "; supplemented by object_info dump"
            elif meta.get("object_info_error"):
                extra = ("; supplied object_info dump could NOT be used: %s"
                         % meta["object_info_error"])
            findings.append(Finding(
                title="%d node type(s) not found in the installation's static node inventory"
                      % len(missing),
                category="workflow",
                certainty="possible" if partial else "observed",
                evidence="not in inventory: %s. Inventory built from %d files "
                         "(%d unreadable, %d with non-literal NODE_CLASS_MAPPINGS)%s%s"
                         % (", ".join(missing[:15]), meta["files_parsed"],
                            meta["files_failed"],
                            len(meta["dynamic_mapping_files"]),
                            "; coverage is partial - some packages register "
                            "nodes dynamically" if partial else "",
                            extra),
                why_it_matters="A workflow referencing node types the "
                               "installation does not provide will fail to "
                               "load those nodes.",
                next_check="Confirm against the running UI (search the node "
                           "menu), or pass --objectinfo with an object_info "
                           "dump for a reliable inventory.",
                refs=missing[:15]))

    for ref in wf.model_refs:
        if ref.status == "missing":
            findings.append(Finding(
                title="Referenced model file not found: %s" % ref.value,
                category="workflow", certainty="observed",
                evidence="input '%s' on node type '%s' references '%s' "
                         "(category guess: %s); not present in: %s"
                         % (ref.input_name, ref.node_type, ref.value,
                            ref.category, "; ".join(ref.searched)),
                why_it_matters="The node that loads this model will error if "
                               "the file is truly absent.",
                next_check="Search manually for the file name; check whether "
                           "a model directory or drive is unmounted.",
                refs=[ref.value]))
        elif ref.status == "ambiguous":
            findings.append(Finding(
                title="Model reference is ambiguous: %s" % ref.value,
                category="workflow", certainty="unknown",
                evidence="%s (searched: %s)"
                         % (ref.notes[0] if ref.notes else "multiple matches",
                            "; ".join(ref.searched)),
                next_check="Use the full subfolder path in the workflow to "
                           "remove the ambiguity."))
        elif ref.status == "unknown":
            findings.append(Finding(
                title="Model reference could not be verified: %s" % ref.value,
                category="workflow", certainty="unknown",
                evidence="; ".join(ref.notes) or "resolution skipped",
                next_check="Verify in the UI file browser if this matters."))
    return findings
