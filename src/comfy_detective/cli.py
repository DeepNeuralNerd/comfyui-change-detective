"""Command-line interface.

Exit codes: 0 success (diagnostic findings are not failures); 2 usage error;
3 snapshot problems; 4 invalid ComfyUI path; 6 supplied workflow/log could
not be parsed or read; 1 unexpected error.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from . import TOOL_NAME, TOOL_VERSION
from . import diagnostics, log_analysis, report, snapshots
from . import workflow as wfmod
from .models import Finding, ScanResult, to_dict
from .scanner import scan_install
from .util import ToolError, default_store_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Read-only diagnostics for a ComfyUI installation: what "
                    "changed, and what to check first. Never modifies the "
                    "installation, never uses the network.",
        epilog="examples:\n"
               "  %s scan --comfyui \"D:\\StabilityMatrix\\Data\\Packages\\ComfyUI\"\n"
               "  %s snapshot create --comfyui \"<path>\" --name known-good\n"
               "  %s snapshot list\n"
               "  %s compare --before known-good --comfyui \"<path>\" "
               "--workflow \"wf.json\" --log \"comfyui.log\"\n"
               "  %s workflow check \"wf.json\" --comfyui \"<path>\"\n"
               % ((TOOL_NAME,) * 5),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version",
                   version="%s %s" % (TOOL_NAME, TOOL_VERSION))
    p.add_argument("--debug", action="store_true",
                   help="show Python tracebacks on internal errors")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, workflows: bool = True) -> None:
        if workflows:
            sp.add_argument("--workflow", action="append", metavar="PATH",
                            help="workflow JSON to analyze (repeatable; files "
                                 "are explicitly supplied only)")
        sp.add_argument("--json", metavar="PATH",
                        help="also write the machine-readable JSON report")
        sp.add_argument("--html", metavar="PATH",
                        help="also write a self-contained local HTML report")

    sp = sub.add_parser("scan", help="inspect a ComfyUI installation now")
    sp.add_argument("--comfyui", required=True, metavar="PATH",
                    help="ComfyUI root directory")
    sp.add_argument("--hash-models", action="store_true",
                    help="sha256 model files referenced by supplied workflows "
                         "(slow on large models; opt-in)")
    sp.add_argument("--no-git-status", action="store_true",
                    help="skip 'git status' dirty checks (faster; dirty state "
                         "then reported as unknown)")
    sp.add_argument("--log", metavar="PATH",
                    help="error log for lead extraction (no snapshot "
                         "comparison in scan mode)")
    add_common(sp)

    sps = sub.add_parser("snapshot", help="manage local snapshots")
    spc = sps.add_subparsers(dest="snapshot_command", required=True)
    sp_create = spc.add_parser("create", help="save a named snapshot of a scan")
    sp_create.add_argument("--comfyui", required=True, metavar="PATH")
    sp_create.add_argument("--name", metavar="NAME",
                           help="human name for later reference")
    sp_create.add_argument("--no-git-status", action="store_true")
    sp_create.add_argument("--workflow", action="append", metavar="PATH",
                           help="workflow JSON to store in the snapshot "
                                "(repeatable)")
    sp_create.add_argument("--store", metavar="DIR",
                           help="snapshot store directory")
    sp_list = spc.add_parser("list", help="list stored snapshots")
    sp_list.add_argument("--store", metavar="DIR",
                         help="snapshot store (default: %%LOCALAPPDATA%%\\ComfyUIChangeDetective"
                              "\\snapshots, or COMFY_DETECTIVE_STORE)")

    sp = sub.add_parser(
        "compare", help="compare a snapshot with another snapshot or a fresh scan")
    sp.add_argument("--before", required=True, metavar="ID_OR_NAME")
    sp.add_argument("--after", metavar="ID_OR_NAME",
                    help="another snapshot; omit to scan --comfyui live")
    sp.add_argument("--comfyui", metavar="PATH",
                    help="required when --after is not given")
    sp.add_argument("--log", metavar="PATH",
                    help="error log (file you supply) for evidence-ranked leads")
    sp.add_argument("--store", metavar="DIR")
    sp.add_argument("--no-git-status", action="store_true")
    add_common(sp)

    spw = sub.add_parser("workflow", help="workflow inspection")
    spwc = spw.add_subparsers(dest="workflow_command", required=True)
    sp_check = spwc.add_parser("check", help="parse and check one workflow JSON")
    sp_check.add_argument("workflow_path", metavar="WORKFLOW.json")
    sp_check.add_argument("--comfyui", metavar="PATH",
                          help="installation to check node types and model "
                               "files against (recommended)")
    sp_check.add_argument("--objectinfo", metavar="PATH",
                          help="object_info JSON dump to supplement the "
                               "static node inventory")
    sp_check.add_argument("--log", metavar="PATH")
    add_common(sp_check, workflows=False)
    return p


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except ToolError as exc:
        sys.stderr.write("error: %s\n" % exc)
        return exc.code
    except KeyboardInterrupt:
        sys.stderr.write("interrupted\n")
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "debug", False):
            raise
        sys.stderr.write("unexpected error (%s): %s\n"
                         "-- re-run with --debug for the traceback\n"
                         % (type(exc).__name__, exc))
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "scan":
        return _cmd_scan(args)
    if args.command == "snapshot":
        if args.snapshot_command == "create":
            return _cmd_snapshot_create(args)
        return _cmd_snapshot_list(args)
    if args.command == "compare":
        return _cmd_compare(args)
    if args.command == "workflow":
        return _cmd_workflow_check(args)
    raise ToolError("unknown command", code=2)


def _emit(rep: Dict[str, Any], args: argparse.Namespace) -> None:
    sys.stdout.write(report.render_text(rep) + "\n")
    for flag in ("json", "html"):
        path = getattr(args, flag, None)
        if not path:
            continue
        text = (report.render_json(rep) if flag == "json"
                else report.render_html(rep))
        try:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            sys.stdout.write("\n%s report written: %s\n" % (flag.upper(), path))
        except OSError as exc:
            sys.stderr.write("could not write %s report %s: %s\n"
                             % (flag, path, exc))


def _do_scan(args: argparse.Namespace, extra_workflows: Sequence[str] = ()
             ) -> ScanResult:
    workflows = list(args.workflow or []) + list(extra_workflows)
    return scan_install(
        args.comfyui, tuple(workflows),
        hash_models=getattr(args, "hash_models", False),
        allow_git_cmd=not getattr(args, "no_git_status", False))


def _scan_summary_lines(sd: Dict[str, Any], counts: Dict[str, int]) -> List[str]:
    git = sd.get("comfy_git") or {}
    nodes = sd.get("custom_nodes", [])
    lines = [
        "root: %s" % sd.get("comfyui_root"),
        "scanned at: %s" % (sd.get("scan_time") or "?"),
        "ComfyUI version: %s | git: %s branch=%s dirty=%s"
        % (sd.get("comfy_version") or "unknown",
           (git.get("commit") or "unknown")[:10],
           "detached" if git.get("detached") else (git.get("branch") or "unknown"),
           git.get("dirty")),
        "packages: %d | custom nodes: %d dir(s) | workflows analyzed: %d"
        % (len(sd.get("packages", [])),
           len([n for n in nodes if n.get("kind") == "dir"]),
           len(sd.get("workflows", []))),
        "findings: %d observed, %d possible, %d unknown"
        % (counts.get("observed", 0), counts.get("possible", 0),
           counts.get("unknown", 0)),
    ]
    for u in sd.get("unavailable", []):
        lines.append("not collected: %s (%s)" % (u.get("field"), u.get("reason")))
    return lines


def _collect_findings(args: argparse.Namespace, sd: Dict[str, Any],
                      scan: ScanResult, with_inventory: bool,
                      objectinfo: Optional[str] = None, log_path: Optional[str] = None,
                      comparison: Optional[Dict[str, Any]] = None
                      ) -> (List[Finding], Optional[Dict[str, Any]], Optional[Dict[str, Any]]):
    findings: List[Finding] = diagnostics.state_findings(sd)
    findings += diagnostics.dependency_findings(sd)
    inventory = None
    if with_inventory and sd.get("workflows"):
        inventory = wfmod.build_node_inventory(sd.get("comfyui_root", ""),
                                               extra_objectinfo=objectinfo)
        for wf in scan.workflows:
            findings += wfmod.workflow_findings(wf, inventory)
    log_info = None
    if log_path:
        types = inventory["types"] if inventory else None
        log_info = log_analysis.analyze_log(log_path, sd, comparison, types)
        findings += log_info["findings"]
        if log_info.get("truncated"):
            findings.append(Finding(
                title="Log was truncated for analysis", category="coverage",
                certainty="unknown",
                evidence="only the first part of %s was read" % log_path,
                next_check="Supply a smaller, time-bounded log slice."))
    return findings, inventory, log_info


def _cmd_scan(args: argparse.Namespace) -> int:
    scan = _do_scan(args)
    sd = to_dict(scan)
    findings, _, log_info = _collect_findings(args, sd, scan, with_inventory=True,
                                              log_path=getattr(args, "log", None))
    rep = report.build_report("scan", _scan_summary_lines(
        sd, report.counts_by_certainty(findings)), findings,
        meta={"comfyui_root": sd.get("comfyui_root")},
        scan_payload=sd, log_info=_log_brief(log_info))
    _emit(rep, args)
    return 6 if any(w.parse_error for w in scan.workflows) else 0


def _log_brief(log_info: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not log_info:
        return None
    return {k: log_info.get(k) for k in
            ("path", "line_count", "truncated", "extracted")}


def _cmd_snapshot_create(args: argparse.Namespace) -> int:
    store = args.store or default_store_dir()
    scan = _do_scan(args)
    sd = to_dict(scan)
    counts = {"observed": 0, "possible": 0, "unknown": 0}
    sid, path = snapshots.create_snapshot(store, scan, args.name)
    sys.stdout.write(
        "snapshot created\n"
        "  id:     %s\n"
        "  name:   %s\n"
        "  file:   %s\n"
        "  root:   %s\n"
        "  content: %d packages, %d custom node(s)%s\n"
        % (sid, args.name or "(none)", path, sd.get("comfyui_root"),
           len(sd.get("packages", [])),
           len([n for n in sd.get("custom_nodes", []) if n.get("kind") == "dir"]),
           ", %d workflow(s)" % len(sd.get("workflows", []))
           if sd.get("workflows") else ""))
    sys.stdout.write("compare later with: %s compare --before %s --comfyui \"%s\"\n"
                     % (TOOL_NAME, sid if not args.name else args.name,
                        sd.get("comfyui_root")))
    del counts
    return 0


def _cmd_snapshot_list(args: argparse.Namespace) -> int:
    store = args.store or default_store_dir()
    sys.stdout.write("snapshot store: %s\n" % store)
    entries = snapshots.list_snapshots(store)
    if not entries:
        sys.stdout.write("  (no snapshots yet) create one with: "
                         "%s snapshot create --comfyui <path>\n" % TOOL_NAME)
        return 0
    for e in entries:
        status = "" if e["valid"] else "  [INVALID: %s]" % e["error"]
        sys.stdout.write("  %-25s %-18s %s%s\n" % (
            e.get("created_at") or "?", e["snapshot_id"],
            e.get("name") or "", status))
        sys.stdout.write(" " * 27 + (e.get("comfyui_root") or "") + "\n")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    store = args.store or default_store_dir()
    before = snapshots.resolve_snapshot(store, args.before)
    scan: Optional[ScanResult] = None
    if args.after:
        after = snapshots.resolve_snapshot(store, args.after)
        after_ident = after.get("snapshot_id", args.after)
    else:
        if not args.comfyui:
            raise ToolError("compare needs --after <snapshot> or --comfyui "
                            "<path> for a fresh scan", code=2)
        scan = _do_scan(args)
        sd = to_dict(scan)
        after = {"snapshot_id": "(live scan)", "created_at": sd.get("scan_time"),
                 "scan": sd}
        after_ident = "live scan of %s" % sd.get("comfyui_root")
    after_sd = after.get("scan", {})
    comparison = diagnostics.compare_snapshots(before, after,
                                               args.before, after_ident)
    findings = list(comparison["findings"])
    findings += diagnostics.state_findings(after_sd)
    findings += diagnostics.dependency_findings(after_sd)
    log_info = None
    if scan is not None:
        extra, _, log_info = _collect_findings(
            args, after_sd, scan, with_inventory=True,
            log_path=args.log, comparison=comparison)
        findings += [f for f in extra if f.category in ("workflow", "log")]
    elif args.log:
        log_info = log_analysis.analyze_log(args.log, after_sd, comparison, None)
        findings += log_info["findings"]
    sec = comparison.get("sections", {})
    cn = sec.get("custom_nodes", {})
    summary = [
        "before: %s (%s)" % (args.before, before.get("created_at") or "?"),
        "after:  %s (%s)" % (after_ident, after.get("created_at") or "?"),
        "custom nodes: %d added, %d removed, %d changed, %d unchanged"
        % (len(cn.get("added", [])), len(cn.get("removed", [])),
           len(cn.get("changed", [])), cn.get("unchanged_count", 0))
        if cn.get("comparable", True) else
        "custom nodes: not comparable (earlier snapshot could not collect them)",
    ] + _scan_summary_lines(after_sd, report.counts_by_certainty(findings))
    rep = report.build_report(
        "compare", summary, findings,
        meta={"before": args.before, "after": after_ident,
              "comfyui_root": after_sd.get("comfyui_root")},
        sections=comparison.get("sections"),
        scan_payload=None if args.after else after_sd,
        log_info=_log_brief(log_info))
    _emit(rep, args)
    return 0


def _cmd_workflow_check(args: argparse.Namespace) -> int:
    scan: Optional[ScanResult] = None
    findings: List[Finding] = []
    wf = wfmod.parse_workflow(args.workflow_path)
    inventory = None
    sd: Optional[Dict[str, Any]] = None
    if args.comfyui:
        scan = scan_install(args.comfyui, (args.workflow_path,),
                            allow_git_cmd=True)
        sd = to_dict(scan)
        wf = scan.workflows[0]
        inventory = wfmod.build_node_inventory(
            sd.get("comfyui_root", ""), extra_objectinfo=args.objectinfo)
        findings += wfmod.workflow_findings(wf, inventory)
    else:
        findings += wfmod.workflow_findings(wf, None)
        findings.append(Finding(
            title="Installation checks skipped", category="coverage",
            certainty="unknown",
            evidence="no --comfyui path given: node-type presence and model "
                     "file existence were not checked",
            next_check="Re-run with --comfyui <path>."))
    log_info = None
    if args.log:
        log_info = log_analysis.analyze_log(
            args.log, sd, None, inventory["types"] if inventory else None)
        findings += log_info["findings"]

    status_counts: Dict[str, int] = {}
    for ref in wf.model_refs:
        status_counts[ref.status or "unresolved"] = \
            status_counts.get(ref.status or "unresolved", 0) + 1
    meta = inventory["meta"] if inventory else None
    summary = [
        "workflow: %s" % wf.path,
        "format: %s | nodes: %d | edges: %d | model references: %d %s"
        % (wf.format, wf.node_count, wf.edge_count, len(wf.model_refs),
           status_counts or ""),
    ]
    if meta:
        summary.append("node inventory: %d types from %d files (%d unreadable, "
                       "%d with dynamic mappings)%s"
                       % (len(inventory["types"]), meta["files_parsed"],
                          meta["files_failed"], len(meta["dynamic_mapping_files"]),
                          " + object_info" if meta.get("object_info_file") else ""))
    summary.append("findings: %d observed, %d possible, %d unknown"
                   % tuple(report.counts_by_certainty(findings)[k]
                           for k in ("observed", "possible", "unknown")))
    if wf.parse_error:
        summary.insert(1, "PARSE ERROR: %s" % wf.parse_error)
    rep = report.build_report("workflow check", summary, findings,
                              meta={"workflow": wf.path,
                                    "comfyui_root": (sd or {}).get("comfyui_root")},
                              sections={"workflow": to_dict(wf)},
                              scan_payload=None, log_info=_log_brief(log_info))
    _emit(rep, args)
    return 6 if wf.parse_error else 0
