"""Snapshot comparison and current-state diagnostics.

Every result is a Finding with an evidence string and a certainty label.
Language rule: changes are stated as observations; anything linking a change
to a failure is 'possible', never causal.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Finding
from .util import (canonical_pkg_name, compare_versions, norm_key, read_text)

MAX_FINDINGS_PER_FILE = 20


# --------------------------------------------------------------------------
# snapshot comparison
# --------------------------------------------------------------------------

def _unavailable_fields(scan: Dict[str, Any]) -> Set[str]:
    return {u.get("field", "") for u in scan.get("unavailable", [])}


def compare_snapshots(before: Dict[str, Any], after: Dict[str, Any],
                      before_ident: str, after_ident: str) -> Dict[str, Any]:
    bs, as_ = before.get("scan", {}), after.get("scan", {})
    b_un, a_un = _unavailable_fields(bs), _unavailable_fields(as_)
    findings: List[Finding] = []
    sections: Dict[str, Any] = {}

    newly = sorted(a for a in b_un - a_un)
    lost = sorted(a for a in a_un - b_un)
    for f in newly:
        findings.append(Finding(
            title="Information newly observed: %s" % f,
            category="coverage", certainty="observed",
            evidence="the '%s' snapshot could not collect this field; the "
                     "newer scan could" % before_ident,
            why_it_matters="This is a difference in scan coverage, not proof "
                           "of a change in the installation.",
            next_check="None needed; treat earlier comparisons for this "
                       "field as 'not comparable'."))
    for f in lost:
        findings.append(Finding(
            title="Information no longer observed: %s" % f,
            category="coverage", certainty="observed",
            evidence="the newer scan could not collect '%s' while the older "
                     "one could" % f,
            next_check="Check why the source became unavailable (removed, "
                       "unreadable, moved)."))
    sections["coverage"] = {"newly_observed": newly, "no_longer_observed": lost}

    _compare_comfyui(bs, as_, findings, sections)
    _compare_environment(bs, as_, b_un, findings, sections)
    _compare_custom_nodes(bs, as_, b_un, findings, sections)
    _compare_packages(bs, as_, b_un, findings, sections)
    _compare_workflows(bs, as_, findings, sections)

    return {"before": before_ident, "after": after_ident,
            "before_created_at": before.get("created_at"),
            "after_created_at": after.get("created_at"),
            "sections": sections,
            "findings": findings}


def _change(findings: List[Finding], title: str, evidence: str,
            certainty: str = "observed", category: str = "comfyui",
            why: str = "", nxt: str = "", refs: Optional[List[str]] = None
            ) -> None:
    findings.append(Finding(title=title, category=category,
                            certainty=certainty, evidence=evidence,
                            why_it_matters=why, next_check=nxt, refs=refs or []))


def _short(sha: Optional[str]) -> str:
    return (sha[:10] if sha else "unknown")


def _compare_comfyui(bs: Dict[str, Any], as_: Dict[str, Any],
                     findings: List[Finding], sections: Dict[str, Any]) -> None:
    sec: Dict[str, Any] = {}
    if bs.get("comfy_version") != as_.get("comfy_version"):
        sec["version"] = {"before": bs.get("comfy_version"),
                          "after": as_.get("comfy_version")}
        _change(findings, "ComfyUI version changed",
                "reported version %s -> %s" % (bs.get("comfy_version"),
                                               as_.get("comfy_version")),
                category="comfyui",
                why="A ComfyUI update can change node behavior, APIs, and "
                    "frontend expectations.",
                nxt="Review the ComfyUI changelog for the range of updates.")
    bg, ag = bs.get("comfy_git") or {}, as_.get("comfy_git") or {}
    if bg.get("is_git") and ag.get("is_git"):
        if bg.get("commit") != ag.get("commit"):
            sec["commit"] = {"before": bg.get("commit"), "after": ag.get("commit")}
            _change(findings, "ComfyUI git revision changed",
                    "commit %s -> %s (branch before: %s, after: %s)"
                    % (_short(bg.get("commit")), _short(ag.get("commit")),
                       bg.get("branch"), ag.get("branch")),
                    category="comfyui",
                    why="Code changed between these snapshots; whether it is "
                        "related to a failure needs manual review.",
                    nxt="Inspect the update: git -C \"%s\" log --oneline %s..%s"
                        % (ag.get("repo_path", ""), _short(bg.get("commit")),
                           _short(ag.get("commit"))),
                    refs=[_short(bg.get("commit")), _short(ag.get("commit"))])
    elif bool(bg.get("is_git")) != bool(ag.get("is_git")):
        sec["is_git"] = {"before": bg.get("is_git"), "after": ag.get("is_git")}
        _change(findings, "ComfyUI git checkout status changed",
                "is_git %s -> %s" % (bg.get("is_git"), ag.get("is_git")),
                category="comfyui")
    for key, title in (("branch", "ComfyUI branch changed"),
                       ("dirty", "ComfyUI worktree dirty state changed"),
                       ("detached", "ComfyUI detached-HEAD state changed")):
        if bg.get(key) != ag.get(key) and not (key == "dirty" and
                                               (bg.get(key) is None or ag.get(key) is None)):
            sec[key] = {"before": bg.get(key), "after": ag.get(key)}
            _change(findings, title, "%s: %s -> %s" % (key, bg.get(key), ag.get(key)),
                    category="comfyui")
    sections["comfyui"] = sec


def _compare_environment(bs: Dict[str, Any], as_: Dict[str, Any],
                         b_un: Set[str], findings: List[Finding],
                         sections: Dict[str, Any]) -> None:
    sec: Dict[str, Any] = {}
    be, ae = bs.get("environment") or {}, as_.get("environment") or {}
    for key in ("venv_dir", "python_exe"):
        bv, av = be.get(key), ae.get(key)
        if bv and av and norm_key(bv) != norm_key(av):
            sec[key] = {"before": bv, "after": av}
            _change(findings, "Python environment path changed (%s)" % key,
                    "%s -> %s" % (bv, av), category="environment",
                    why="A different interpreter/venv resolves packages "
                        "differently.",
                    nxt="Confirm which launcher the ComfyUI start menu uses.")
    bver = (be.get("pyvenv") or {}).get("version")
    aver = (ae.get("pyvenv") or {}).get("version")
    if bver and aver and bver != aver:
        sec["python_version"] = {"before": bver, "after": aver}
        _change(findings, "venv Python version changed",
                "pyvenv.cfg version %s -> %s" % (bver, aver),
                category="environment",
                why="Python major/minor changes replace the whole package set.",
                nxt="Check whether Stability Matrix (re)created the venv; "
                    "reinstall custom-node dependencies after such a change.")
    sections["environment"] = sec


def _compare_custom_nodes(bs: Dict[str, Any], as_: Dict[str, Any],
                          b_un: Set[str], findings: List[Finding],
                          sections: Dict[str, Any]) -> None:
    sec: Dict[str, Any] = {"added": [], "removed": [], "changed": [],
                           "unchanged_count": 0}
    if "custom_nodes" in b_un:
        sec["comparable"] = False
        sections["custom_nodes"] = sec
        return  # before-side coverage gap: additions would be false results
    bmap = {n.get("name", "").lower(): n for n in bs.get("custom_nodes", [])}
    amap = {n.get("name", "").lower(): n for n in as_.get("custom_nodes", [])}
    for name in sorted(set(amap) - set(bmap)):
        sec["added"].append(name)
        _change(findings, "Custom node added: %s" % name,
                "present in the newer scan, absent in the older one",
                category="custom_node",
                why="New nodes can register conflicting node types or pull in "
                    "new dependencies.",
                nxt="If startup fails on this package, test disabling it "
                    "(move its directory to custom_nodes\\.disabled).",
                refs=[name])
    for name in sorted(set(bmap) - set(amap)):
        sec["removed"].append(name)
        _change(findings, "Custom node removed: %s" % name,
                "present in the older scan, absent in the newer one",
                category="custom_node",
                why="Workflows referencing this package's nodes will fail to "
                    "load them.",
                nxt="Confirm whether it was intentionally uninstalled.",
                refs=[name])
    for name in sorted(set(bmap) & set(amap)):
        bn, an = bmap[name], amap[name]
        changes: Dict[str, Any] = {}
        bg, ag = bn.get("git") or {}, an.get("git") or {}
        if bg.get("commit") != ag.get("commit") and bg.get("commit") and ag.get("commit"):
            changes["commit"] = {"before": bg.get("commit"), "after": ag.get("commit")}
        if bool(bg.get("is_git")) != bool(ag.get("is_git")):
            changes["is_git"] = {"before": bg.get("is_git"), "after": ag.get("is_git")}
        if None not in (bg.get("dirty"), ag.get("dirty")) and \
                bg.get("dirty") != ag.get("dirty"):
            changes["dirty"] = {"before": bg.get("dirty"), "after": ag.get("dirty")}
        if changes:
            sec["changed"].append({"name": name, "changes": changes})
            parts = []
            if "commit" in changes:
                parts.append("commit %s -> %s" % (_short(changes["commit"]["before"]),
                                                  _short(changes["commit"]["after"])))
            if "is_git" in changes:
                parts.append("is_git %s -> %s" % (changes["is_git"]["before"],
                                                  changes["is_git"]["after"]))
            if "dirty" in changes:
                parts.append("dirty %s -> %s" % (changes["dirty"]["before"],
                                                 changes["dirty"]["after"]))
            _change(findings, "Custom node changed: %s" % name,
                    "; ".join(parts), category="custom_node",
                    why="An updated custom node is a common source of node "
                        "errors; this is evidence to check, not proof.",
                    nxt="Review that package's recent commits or test "
                        "disabling it.",
                    refs=[name])
        else:
            sec["unchanged_count"] += 1
    sections["custom_nodes"] = sec


def _compare_packages(bs: Dict[str, Any], as_: Dict[str, Any],
                      b_un: Set[str], findings: List[Finding],
                      sections: Dict[str, Any]) -> None:
    sec: Dict[str, Any] = {"added": [], "removed": [], "changed": [],
                           "unchanged_count": 0}
    if "packages" in b_un or "environment" in b_un:
        sec["comparable"] = False
        sections["packages"] = sec
        return
    bmap = {canonical_pkg_name(p.get("name", "")): p.get("version")
            for p in bs.get("packages", [])}
    amap = {canonical_pkg_name(p.get("name", "")): p.get("version")
            for p in as_.get("packages", [])}
    for name in sorted(set(amap) - set(bmap)):
        sec["added"].append({"name": name, "version": amap[name]})
    for name in sorted(set(bmap) - set(amap)):
        sec["removed"].append({"name": name, "version": bmap[name]})
    for name in sorted(set(bmap) & set(amap)):
        if bmap[name] != amap[name]:
            sec["changed"].append({"name": name, "before": bmap[name],
                                   "after": amap[name]})
    n_changed = len(sec["added"]) + len(sec["removed"]) + len(sec["changed"])
    sec["unchanged_count"] = len(set(bmap) & set(amap)) - len(sec["changed"])
    if n_changed:
        _change(findings,
                "%d Python distribution(s) differ (%d added, %d removed, %d version change)"
                % (n_changed, len(sec["added"]), len(sec["removed"]),
                   len(sec["changed"])),
                evidence="details in the packages section; e.g. first changes: "
                         + "; ".join(
                             (["+%s %s" % (a["name"], a["version"]) for a in sec["added"][:5]]
                              + ["-%s %s" % (r["name"], r["version"]) for r in sec["removed"][:5]]
                              + ["%s %s->%s" % (c["name"], c["before"], c["after"])
                                 for c in sec["changed"][:5]])[:12]),
                category="packages",
                why="Package updates (especially torch/transformers-class "
                    "libraries) frequently change behavior; correlate with "
                    "the error before drawing conclusions.",
                nxt="Correlate error timestamps with the update; if known-good "
                    "versions matter, note them now.")
    sections["packages"] = sec


def _compare_workflows(bs: Dict[str, Any], as_: Dict[str, Any],
                       findings: List[Finding], sections: Dict[str, Any]) -> None:
    bmap = {norm_key(w.get("path", "")): w for w in bs.get("workflows", [])}
    amap = {norm_key(w.get("path", "")): w for w in as_.get("workflows", [])}
    entries: List[Dict[str, Any]] = []
    for key in sorted(set(bmap) | set(amap)):
        bw, aw = bmap.get(key), amap.get(key)
        if bw is None or aw is None:
            entries.append({"path": (aw or bw).get("path"),
                            "note": "analyzed only in %s scan"
                                    % ("newer" if bw is None else "older")})
            continue
        entry: Dict[str, Any] = {"path": aw.get("path"), "changes": {}}
        bclasses, aclasses = set(bw.get("node_classes", [])), set(aw.get("node_classes", []))
        added_c, removed_c = sorted(aclasses - bclasses), sorted(bclasses - aclasses)
        if added_c or removed_c:
            entry["changes"]["node_classes"] = {"added": added_c, "removed": removed_c}
            _change(findings, "Workflow node classes changed: %s"
                    % os.path.basename(str(aw.get("path"))),
                    "added: %s; removed: %s" % (added_c[:10] or "none",
                                                removed_c[:10] or "none"),
                    category="workflow",
                    evidence="workflow %s: +%s -%s" % (aw.get("path"),
                                                       ", ".join(added_c[:10]) or "none",
                                                       ", ".join(removed_c[:10]) or "none"),
                    why="If a workflow previously loaded and now fails, the "
                        "added/removed nodes are the first suspects.",
                    nxt="Re-open the workflow and look for red (missing) nodes.",
                    refs=added_c[:10] + removed_c[:10])
        for field in ("node_count", "edge_count"):
            if bw.get(field) != aw.get(field):
                entry["changes"][field] = {"before": bw.get(field),
                                           "after": aw.get(field)}
        brefs = {r.get("value", ""): r for r in bw.get("model_refs", [])}
        arefs = {r.get("value", ""): r for r in aw.get("model_refs", [])}
        for ref in sorted(set(brefs) & set(arefs)):
            br, ar = brefs[ref], arefs[ref]
            delta: Dict[str, Any] = {}
            if br.get("status") != ar.get("status"):
                delta["status"] = {"before": br.get("status"),
                                   "after": ar.get("status")}
            elif br.get("status") == "found" and ar.get("status") == "found":
                if norm_key(str(br.get("resolved_path") or "")) != \
                        norm_key(str(ar.get("resolved_path") or "")):
                    delta["resolved_path"] = {"before": br.get("resolved_path"),
                                              "after": ar.get("resolved_path")}
                if br.get("size") != ar.get("size"):
                    delta["size"] = {"before": br.get("size"), "after": ar.get("size")}
                elif br.get("mtime") != ar.get("mtime"):
                    delta["mtime_only"] = {"before": br.get("mtime"),
                                           "after": ar.get("mtime")}
            if delta:
                entry["changes"].setdefault("model_refs", []).append(
                    {"ref": ref, "delta": delta})
                if "mtime_only" in delta and len(delta) == 1:
                    _change(findings,
                            "Model file touched (mtime only): %s" % ref,
                            "same size (%s); mtime %s -> %s"
                            % (br.get("size"), br.get("mtime"), ar.get("mtime")),
                            category="workflow", certainty="possible",
                            why="mtime changes can come from copies, sync "
                                "tools, or restores - content may be identical "
                                "(no hash was taken).",
                            nxt="Compare hashes manually if this model matters "
                                "(comfy-detective scan --hash-models).")
                elif "status" in delta and delta["status"].get("after") == "missing":
                    _change(findings, "Model reference became unresolvable: %s" % ref,
                            "status %s -> %s (searched: %s)"
                            % (delta["status"]["before"], delta["status"]["after"],
                               "; ".join((ar.get("searched") or [])[:4])),
                            category="workflow",
                            why="A model that moved, was renamed, or its drive "
                                "unmounted breaks the loading node.",
                            nxt="Search for the file name on the model drives.",
                            refs=[ref])
                else:
                    _change(findings, "Model reference changed: %s" % ref,
                            "%s" % delta, category="workflow")
        entries.append(entry)
    sections["workflows"] = entries


# --------------------------------------------------------------------------
# current-state diagnostics
# --------------------------------------------------------------------------

def state_findings(scan: Dict[str, Any]) -> List[Finding]:
    findings: List[Finding] = []
    nodes = scan.get("custom_nodes", [])
    non_git = [n.get("name") for n in nodes
               if n.get("kind") == "dir" and not (n.get("git") or {}).get("is_git")]
    if non_git:
        _change(findings,
                "%d custom-node director%s not under git" % (
                    len(non_git), "y is" if len(non_git) == 1 else "ies are"),
                evidence="no .git directory in: %s"
                         % ", ".join(str(n) for n in non_git[:15]),
                category="custom_node",
                why="Without git there is no revision history: updates and "
                    "local edits cannot be compared or rolled back.",
                nxt="Note how these were installed (zip/manager); consider "
                    "tracking versions manually.",
                refs=[str(n) for n in non_git[:15]])
    dirty = [n.get("name") for n in nodes if (n.get("git") or {}).get("dirty")]
    if dirty:
        _change(findings, "%d custom-node repos have uncommitted changes" % len(dirty),
                category="custom_node",
                evidence="'git status --porcelain' returned changes in: %s"
                         % ", ".join(str(d) for d in dirty[:15]),
                why="Local edits are part of the running behavior and would "
                    "be lost or conflict on a git update; they may also be "
                    "the difference from a known-good state.",
                nxt="Inspect with: git -C <dir> status",
                refs=[str(d) for d in dirty[:15]])
    unknown_dirty = [n.get("name") for n in nodes
                     if (n.get("git") or {}).get("is_git")
                     and (n.get("git") or {}).get("dirty") is None]
    if unknown_dirty:
        _change(findings,
                "dirty state unknown for %d git repo(s)" % len(unknown_dirty),
                "could not run 'git status' (git binary missing or sandbox "
                "restriction); repos include: %s"
                % ", ".join(str(n) for n in unknown_dirty[:10]),
                category="coverage", certainty="unknown",
                nxt="Install/run with git available to fill this in.")
    cg = scan.get("comfy_git") or {}
    if cg.get("is_git") and cg.get("dirty"):
        _change(findings, "ComfyUI worktree has uncommitted changes",
                "git status reports changes in %s" % cg.get("repo_path"),
                category="comfyui",
                why="Local modifications to ComfyUI itself change behavior "
                    "outside of any release.",
                nxt="Review with: git -C \"%s\" status" % cg.get("repo_path", ""))
    if cg.get("detached"):
        _change(findings, "ComfyUI is on a detached HEAD",
                "commit %s, no branch" % _short(cg.get("commit")),
                category="comfyui",
                why="Updates via git pull are unlikely to work from a "
                    "detached state.",
                nxt="Confirm whether the pin is intentional.")
    return findings


# --------------------------------------------------------------------------
# declared requirements vs installed distributions (static, no pip)
# --------------------------------------------------------------------------

_REQ_NAME_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")
_SPEC_RE = re.compile(r"^(===|==|>=|<=|~=|!=|>|<)\s*([A-Za-z0-9.*+!_-]+)$")


def parse_requirements(text: str) -> Tuple[List[Tuple[str, List[Tuple[str, str]], bool]],
                                           List[str]]:
    reqs: List[Tuple[str, List[Tuple[str, str]], bool]] = []
    skipped: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith(("-", "--")):
            skipped.append(raw.strip() + "  (option/include line)")
            continue
        has_marker = ";" in line
        if has_marker:
            line = line.split(";", 1)[0].strip()
        m = _REQ_NAME_RE.match(line)
        if not m:
            skipped.append(raw.strip() + "  (unrecognized syntax)")
            continue
        name, rest = m.group(1), m.group(2).strip()
        if "://" in rest or rest.startswith((".", "/", "file:", "git+")):
            skipped.append(raw.strip() + "  (URL/local path)")
            continue
        specs: List[Tuple[str, str]] = []
        if rest:
            ok = True
            for part in rest.split(","):
                sm = _SPEC_RE.match(part.strip())
                if not sm:
                    ok = False
                    break
                specs.append((sm.group(1), sm.group(2)))
            if not ok:
                skipped.append(raw.strip() + "  (version syntax)")
                continue
        reqs.append((name, specs, has_marker))
    return reqs, skipped


def _spec_satisfied(op: str, spec_ver: str, installed: str
                    ) -> Optional[bool]:
    if op == "===":
        return spec_ver.strip() == installed.strip()
    if spec_ver.endswith("*"):
        base = spec_ver.rstrip("*").rstrip(".")
        if op == "==":
            return installed.startswith(base)
        cmp_ = compare_versions(installed, base)
        if cmp_ is None:
            return None
        return cmp_ >= 0 if op in (">=", "<=") else (cmp_ > 0 if op == ">" else
                                                     (cmp_ <= 0 if op == "<" else
                                                      installed != base))
    cmp_ = compare_versions(installed, spec_ver)
    if cmp_ is None:
        return None
    if op in ("==", ">="):
        return cmp_ >= 0 if op == ">=" else cmp_ == 0
    if op == "<=":
        return cmp_ <= 0
    if op == ">":
        return cmp_ > 0
    if op == "<":
        return cmp_ < 0
    if op == "!=":
        return cmp_ != 0
    if op == "~=":
        # PEP 440 compatible release: ~=X.Y.Z means >= X.Y.Z, < X.(Y+1);
        # ~=X.Y means >= X.Y, < (X+1). Drop the final segment, bump the rest.
        low = compare_versions(installed, spec_ver)
        if low is None:
            return None
        if low < 0:
            return False
        head = spec_ver.split(".")[:-1]
        if len(head) < 1:
            return None
        try:
            head[-1] = str(int(head[-1].split("-")[0].split("+")[0]) + 1)
        except ValueError:
            return None
        return compare_versions(installed, ".".join(head)) < 0
    return None


def requirement_findings(requirements_path: str, source_label: str,
                         installed: Dict[str, str]) -> List[Finding]:
    findings: List[Finding] = []
    text, err = read_text(requirements_path, 256 * 1024)
    if text is None:
        return [Finding(
            title="Requirements file unreadable", category="input",
            certainty="unknown", evidence="%s: %s" % (requirements_path, err))]
    reqs, skipped = parse_requirements(text)
    for name, specs, has_marker in reqs:
        canon = canonical_pkg_name(name)
        inst = installed.get(canon)
        if inst is None:
            if has_marker:
                continue
            findings.append(Finding(
                title="Declared requirement not found installed: %s" % name,
                category="packages", certainty="possible",
                evidence="'%s' declared in %s; no distribution with the "
                         "canonical name '%s' present in the scanned venv's "
                         "installed metadata" % (name, requirements_path, canon),
                why_it_matters="If imports resolve the distribution under a "
                               "different name, this is a false positive.",
                next_check="Confirm with the error log or the package's "
                           "import name.",
                refs=[name]))
            continue
        bad: List[str] = []
        unknown: List[str] = []
        for op, spec_ver in specs:
            ok = _spec_satisfied(op, spec_ver, inst)
            if ok is False:
                bad.append("%s%s" % (op, spec_ver))
            elif ok is None:
                unknown.append("%s%s" % (op, spec_ver))
        if bad:
            findings.append(Finding(
                title="Installed version does not satisfy declaration: %s" % name,
                category="packages", certainty="observed",
                evidence="%s requires %s%s; installed version is %s"
                         % (requirements_path, name, ",".join(bad), inst),
                why_it_matters="The package's own declared floor/ceiling is "
                               "violated in this environment.",
                next_check="Check who last changed this version (package "
                           "manager history) before adjusting.",
                refs=[name]))
        elif unknown:
            findings.append(Finding(
                title="Version comparison inconclusive: %s" % name,
                category="packages", certainty="unknown",
                evidence="declared %s%s in %s; installed %s - non-standard "
                         "version syntax" % (name, ",".join(unknown),
                                             requirements_path, inst),
                next_check="Compare manually if this package is implicated."))
    for note in skipped[:5]:
        findings.append(Finding(
            title="Requirement line not interpreted", category="coverage",
            certainty="unknown",
            evidence="%s: %s" % (requirements_path, note),
            next_check="None; noted for coverage."))
    return findings[:MAX_FINDINGS_PER_FILE]


def dependency_findings(scan: Dict[str, Any]) -> List[Finding]:
    installed = {canonical_pkg_name(p.get("name", "")): p.get("version", "")
                 for p in scan.get("packages", [])}
    if not installed:
        return [Finding(
            title="Dependency checks skipped", category="coverage",
            certainty="unknown",
            evidence="no installed distribution metadata was collected "
                     "(no venv found or site-packages unreadable)",
            next_check="Point the scan at the ComfyUI root that contains the "
                       "venv used to run ComfyUI.")]
    findings: List[Finding] = []
    root = scan.get("comfyui_root", "")
    root_req = os.path.join(root, "requirements.txt")
    if os.path.isfile(root_req):
        findings += requirement_findings(root_req, "comfyui", installed)
    for node in scan.get("custom_nodes", []):
        req = node.get("requirements")
        if req and os.path.isfile(req):
            findings += requirement_findings(req, "custom_node", installed)
    return findings
