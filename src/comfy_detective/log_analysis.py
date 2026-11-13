"""Optional analysis of a user-supplied log file.

Conservative extraction only; every lead is at best 'possible' and carries
the evidence chain (what the log said + what changed). The full log is never
copied into reports - only short excerpts with line numbers.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Finding
from .util import read_text

_LINE_CAP = 200
_MAX_EXCERPTS_PER_LEAD = 3
_MAX_LEAD_FINDINGS = 12

_MODULE_RE = re.compile(
    r"(?:ModuleNotFoundError|ImportError):\s*No module named ['\"]?([\w.]+)")
_CANNOT_IMPORT_RE = re.compile(
    r"Cannot import\s+(\S+?\.py)\s+module for custom nodes[:\s]*(.*)", re.I)
_CUSTOM_NODE_PATH_RE = re.compile(r"custom_nodes[\\/](?!\.)([\w.\-]+)")
_EXC_CLASS_RE = re.compile(r"^\s*(?:raise\s+)?([A-Za-z_][\w\.]*"
                           r"(?:Error|Exception|Interrupt))(?::|$)")
_MODEL_FILE_RE = re.compile(
    r"([\w\s\-./\\:+]*?\.(?:safetensors|ckpt|pt|pth|gguf|onnx|bin))(?![\w])")
_MISSING_HINT_RE = re.compile(
    r"not\s+found|cannot|can't|missing|no such file|does not exist|failed", re.I)
_NODE_CLASS_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]{2,})\b")

_TOKEN_STOP = {"ERROR", "WARNING", "INFO", "DEBUG", "CRITICAL", "Traceback",
               "Exception", "Error", "None", "True", "False", "Python",
               "Windows", "ComfyUI", "Comfy"}


def analyze_log(log_path: str, scan: Optional[Dict[str, Any]],
                comparison: Optional[Dict[str, Any]],
                inventory_types: Optional[Set[str]] = None) -> Dict[str, Any]:
    text, err = read_text(log_path, 4 * 1024 * 1024)
    if text is None:
        return {"path": os.path.abspath(log_path),
                "findings": [Finding(
                    title="Log file unreadable", category="input",
                    certainty="unknown",
                    evidence="%s: %s" % (log_path, err),
                    next_check="Pass the log file with --log <path>.")],
                "extracted": {}}
    lines = text.splitlines()
    truncated = text.endswith("]") and "truncated at" in (lines[-1] if lines else "")

    extracted = _extract(lines, inventory_types)
    findings = _correlate(extracted, scan, comparison, lines)
    return {"path": os.path.abspath(log_path), "line_count": len(lines),
            "truncated": bool(truncated), "extracted": extracted,
            "findings": findings}


def _excerpt(lines: List[str], index: int) -> str:
    line = lines[index].strip()
    if len(line) > _LINE_CAP:
        line = line[:_LINE_CAP] + "..."
    return "line %d: %s" % (index + 1, line)


def _extract(lines: List[str], inventory_types: Optional[Set[str]]
             ) -> Dict[str, Any]:
    modules: Dict[str, List[int]] = {}
    cannot_import: List[Tuple[str, str, int]] = []
    node_paths: Dict[str, List[int]] = {}
    exceptions: Dict[str, int] = {}
    traceback_count = 0
    model_missing: Dict[str, List[int]] = {}
    node_mentions: Dict[str, List[int]] = {}

    for i, line in enumerate(lines):
        m = _MODULE_RE.search(line)
        if m:
            modules.setdefault(m.group(1), []).append(i)
        m = _CANNOT_IMPORT_RE.search(line)
        if m:
            cannot_import.append((m.group(1), m.group(2).strip()[:120], i))
        for m in _CUSTOM_NODE_PATH_RE.finditer(line):
            node_paths.setdefault(m.group(1), []).append(i)
        if line.lstrip().startswith("Traceback (most recent call last)"):
            traceback_count += 1
        m = _EXC_CLASS_RE.search(line)
        if m and not line.lstrip().startswith("File "):
            exceptions[m.group(1)] = exceptions.get(m.group(1), 0) + 1
        if _MISSING_HINT_RE.search(line):
            for m in _MODEL_FILE_RE.finditer(line):
                model_missing.setdefault(m.group(1).strip(), []).append(i)
        if inventory_types:
            for m in _NODE_CLASS_RE.finditer(line):
                tok = m.group(1)
                if tok in inventory_types and tok not in _TOKEN_STOP:
                    node_mentions.setdefault(tok, []).append(i)
    return {
        "missing_modules": {k: v[:_MAX_EXCERPTS_PER_LEAD] for k, v in modules.items()},
        "cannot_import": cannot_import[:20],
        "custom_node_names": {k: v[:_MAX_EXCERPTS_PER_LEAD] for k, v in node_paths.items()},
        "exception_types": dict(sorted(exceptions.items(),
                                       key=lambda kv: -kv[1])[:10]),
        "traceback_count": traceback_count,
        "model_files_in_errors": {k: v[:_MAX_EXCERPTS_PER_LEAD]
                                  for k, v in model_missing.items()},
        "node_class_mentions": {k: v[:_MAX_EXCERPTS_PER_LEAD]
                                for k, v in node_mentions.items()},
    }


def _entity_to_nodes(entity: str, node_names: Set[str]) -> Set[str]:
    """Map a module/package/entity name onto known custom-node dir names."""
    entity_c = entity.lower().replace("_", "-").strip("-")
    hits = set()
    for name in node_names:
        n = name.lower()
        if n == entity_c:
            hits.add(name)
            continue
        core = re.sub(r"^comfy-?ui[-_]", "", n).replace("_", "-")
        ent_core = re.sub(r"^comfy-?ui[-_]", "", entity_c)
        if core and core == ent_core:
            hits.add(name)
    return hits


def _changed_index(comparison: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """lower-cased entity name -> change description from a comparison."""
    idx: Dict[str, str] = {}
    if not comparison:
        return idx
    sec = comparison.get("sections", {}).get("custom_nodes", {})
    for name in sec.get("added", []):
        idx[str(name).lower()] = "was added since the earlier snapshot"
    for name in sec.get("removed", []):
        idx[str(name).lower()] = "was removed since the earlier snapshot"
    for ch in sec.get("changed", []):
        changes = ch.get("changes", {})
        if "commit" in changes:
            idx[str(ch["name"]).lower()] = (
                "changed from commit %s to %s between these snapshots"
                % (_short(changes["commit"]["before"]),
                   _short(changes["commit"]["after"])))
        elif changes:
            idx[str(ch["name"]).lower()] = "changed: %s" % changes
    for group in ("added", "removed"):
        for p in comparison.get("sections", {}).get("packages", {}).get(group, []):
            idx[str(p["name"]).lower()] = "package %s %s %s" % (
                group, p["name"], p.get("version", ""))
    for p in comparison.get("sections", {}).get("packages", {}).get("changed", []):
        idx[str(p["name"]).lower()] = "package %s changed %s -> %s" % (
            p["name"], p["before"], p["after"])
    return idx


def _short(sha: Optional[str]) -> str:
    return (sha[:10] if sha else "unknown")


def _correlate(extracted: Dict[str, Any], scan: Optional[Dict[str, Any]],
               comparison: Optional[Dict[str, Any]],
               lines: List[str]) -> List[Finding]:
    findings: List[Finding] = []
    node_names = {n.get("name", "") for n in (scan or {}).get("custom_nodes", [])}
    changed = _changed_index(comparison)

    def entity_lead(entity: str, log_context: str, indices: List[int],
                    kind: str) -> None:
        matched_nodes = _entity_to_nodes(entity, node_names)
        change_bits: List[str] = []
        names = {entity.lower()} | {n.lower() for n in matched_nodes}
        for m in names:
            if m in changed:
                change_bits.append(changed[m])
        excerpts = "; ".join(_excerpt(lines, i) for i in indices[:2])
        if change_bits:
            findings.append(Finding(
                title="Possible lead: %s %s" % (entity, change_bits[0]),
                category="log", certainty="possible",
                evidence="The supplied log contains %s (%s). Separately, this "
                         "package %s. This is a useful lead to investigate, "
                         "but it does not establish that the change caused "
                         "the error." % (log_context, excerpts, change_bits[0]),
                why_it_matters="A changed package implicated by the error "
                               "text is the cheapest thing to check first.",
                next_check="Check that package's recent changes, or test with "
                           "it disabled, to confirm or rule it out.",
                refs=[entity] + sorted(matched_nodes)))
        elif matched_nodes:
            findings.append(Finding(
                title="Log mentions %s (no tracked change)" % entity,
                category="log", certainty="possible",
                evidence="%s (%s); the package exists in the installation but "
                         "showed no change between the compared snapshots%s"
                         % (log_context, excerpts,
                            "" if comparison else " (no comparison supplied)"),
                next_check="If the error persists, test with this package "
                           "disabled."))
        else:
            findings.append(Finding(
                title="Log mentions %s" % entity, category="log",
                certainty="possible",
                evidence="%s (%s); no matching custom node or tracked change "
                         "was found" % (log_context, excerpts),
                next_check="Identify which package provides this name "
                           "(pip show / search the source)."))

    for module, idx in sorted(extracted["missing_modules"].items()):
        entity_lead(module.split(".")[0],
                    "an import error for module '%s'" % module, idx, "module")
    for fname, reason, i in extracted["cannot_import"]:
        stem = os.path.splitext(fname)[0]
        entity_lead(stem, "a 'Cannot import %s' startup error (%s)"
                    % (fname, reason or "no reason given"), [i], "node")
    for name, idx in sorted(extracted["custom_node_names"].items()):
        if name.lower() in {m.split(".")[0] for m in extracted["missing_modules"]}:
            continue
        entity_lead(name, "a reference under custom_nodes", idx, "node")

    for ref, idx in sorted(extracted["model_files_in_errors"].items()):
        excerpts = "; ".join(_excerpt(lines, i) for i in idx[:2])
        findings.append(Finding(
            title="Log suggests a missing model: %s" % os.path.basename(ref),
            category="log", certainty="possible",
            evidence="%s" % excerpts,
            why_it_matters="A missing model file breaks only the nodes that "
                           "load it; it is rarely a startup failure.",
            next_check="Verify the file exists (or run 'workflow check' on "
                       "the failing workflow for a directory-level answer).",
            refs=[ref]))

    for cls, count in sorted(extracted["exception_types"].items()):
        if count >= 2 and not any(f.category == "log" and cls.lower() in
                                  f.title.lower() for f in findings):
            findings.append(Finding(
                title="Repeated exception type in log: %s (x%d)" % (cls, count),
                category="log", certainty="unknown",
                evidence="%d occurrences; class name only, no entity match "
                         "against installation changes" % count,
                next_check="Read the traceback around the first occurrence."))

    if extracted["traceback_count"]:
        findings.append(Finding(
            title="%d traceback(s) in the supplied log" % extracted["traceback_count"],
            category="log", certainty="observed",
            evidence="count of 'Traceback (most recent call last)' lines",
            next_check="Correlate timestamps with when the workflow failed."))

    return findings[:_MAX_LEAD_FINDINGS]
