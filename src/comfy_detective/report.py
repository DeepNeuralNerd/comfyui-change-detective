"""Report model + renderers (terminal text, JSON, optional HTML).

Section order follows the design: Summary, Observed findings, Workflow
issues, Possible leads (log), Missing/incomplete information, Suggested
next checks. Certainty labels are rendered for every finding. HTML escapes
all content and opens nothing automatically.
"""

from __future__ import annotations

import datetime as _dt
import html as _html
import json
from typing import Any, Dict, List, Optional

from . import TOOL_VERSION
from .models import Finding

_NEXT_CHECK_CAP = 8


def build_report(mode: str, summary: List[str], findings: List[Finding],
                 meta: Optional[Dict[str, Any]] = None,
                 sections: Optional[Dict[str, Any]] = None,
                 scan_payload: Optional[Dict[str, Any]] = None,
                 log_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "tool_version": TOOL_VERSION,
        "generated_at": _dt.datetime.now(_dt.timezone.utc)
                        .isoformat(timespec="seconds"),
        "mode": mode,
        "meta": meta or {},
        "summary": summary,
        "findings": [_finding_dict(f) for f in findings],
        "sections": sections or {},
        "log_info": log_info,
        "scan": scan_payload,
    }


def _finding_dict(f: Finding) -> Dict[str, Any]:
    return {"title": f.title, "category": f.category,
            "certainty": f.certainty, "evidence": f.evidence,
            "why_it_matters": f.why_it_matters, "next_check": f.next_check,
            "refs": f.refs}


def counts_by_certainty(findings: List[Finding]) -> Dict[str, int]:
    out = {"observed": 0, "possible": 0, "unknown": 0}
    for f in findings:
        out[f.certainty] = out.get(f.certainty, 0) + 1
    return out


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------

_WIDTH = 78


def render_text(report: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("=" * _WIDTH)
    out.append("COMFYUI CHANGE DETECTIVE - %s" % report["mode"].upper())
    out.append("=" * _WIDTH)
    out.append("")
    out.append("SUMMARY")
    for line in report["summary"]:
        out.append("  %s" % line)
    out.append("")

    findings = [Finding(**f) for f in report["findings"]]
    observed = [f for f in findings if f.category not in ("workflow", "log")]
    workflow = [f for f in findings if f.category == "workflow"]
    leads = [f for f in findings if f.category == "log"]
    incomplete = [f for f in findings if f.category == "coverage"]
    observed = [f for f in observed if f.category != "coverage"]

    def section(title: str, items: List[Finding]) -> None:
        out.append(title)
        if not items:
            out.append("  (none)")
        for f in items:
            _render_finding(out, f)
        out.append("")

    order = {"observed": 0, "possible": 1, "unknown": 2}
    observed.sort(key=lambda f: order.get(f.certainty, 3))
    section("OBSERVED FACTS AND CHANGES", observed)
    section("WORKFLOW FINDINGS", workflow)
    section("POSSIBLE LEADS FROM SUPPLIED LOG", leads)
    section("MISSING / INCOMPLETE SCAN INFORMATION", incomplete)

    out.append("SUGGESTED NEXT CHECKS")
    seen: set = set()
    n = 0
    for f in observed + workflow + leads:
        if f.next_check and f.next_check not in seen and \
                f.certainty in ("observed", "possible"):
            seen.add(f.next_check)
            out.append("  - %s" % f.next_check)
            n += 1
            if n >= _NEXT_CHECK_CAP:
                break
    if n == 0:
        out.append("  (nothing to check beyond the notes above)")
    out.append("")
    out.append("Certainty legend: observed = directly measured now; possible "
               "= consistent with evidence but unproven; unknown = could not "
               "be determined. No finding claims that a change caused a "
               "failure.")
    return "\n".join(out)


def _render_finding(out: List[str], f: Finding) -> None:
    out.append("  [%s] %s" % (f.certainty.upper(), f.title))
    out.append("      evidence: %s" % _wrap(f.evidence, 67, "      "))
    if f.why_it_matters:
        out.append("      why:      %s" % _wrap(f.why_it_matters, 67, "      "))
    if f.next_check:
        out.append("      next:     %s" % _wrap(f.next_check, 67, "      "))


def _wrap(text: str, width: int, indent: str) -> str:
    words = str(text).split(" ")
    lines: List[str] = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = indent.strip() + w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)
    return ("\n").join(lines)


# --------------------------------------------------------------------------
# json / html
# --------------------------------------------------------------------------

def render_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


_CSS = """
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:60rem;
padding:0 1rem;color:#1c1c1c;background:#fafafa}
h1{font-size:1.3rem} h2{font-size:1.05rem;margin-top:1.6rem;border-bottom:1px solid #ddd}
.summary li{margin:.2rem 0}
.finding{border-left:4px solid #bbb;padding:.4rem .8rem;margin:.6rem 0;background:#fff}
.observed{border-color:#2a7}.possible{border-color:#c80}.unknown{border-color:#999}
.evidence{white-space:pre-wrap;font-size:.9rem;color:#333;margin-top:.2rem}
.meta{color:#666;font-size:.85rem}
"""


def render_html(report: Dict[str, Any]) -> str:
    esc = _html.escape
    parts: List[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>"
                 "<title>comfy-detective report</title><style>%s</style></head><body>"
                 % _CSS)
    parts.append("<h1>ComfyUI Change Detective - %s report</h1>"
                 % esc(str(report.get("mode", ""))))
    parts.append("<p class='meta'>generated %s · tool v%s</p>"
                 % (esc(str(report.get("generated_at", ""))),
                    esc(str(report.get("tool_version", "")))))
    parts.append("<h2>Summary</h2><ul class='summary'>")
    for line in report.get("summary", []):
        parts.append("<li>%s</li>" % esc(line))
    parts.append("</ul>")

    def emit(title: str, items: List[Dict[str, Any]]) -> None:
        parts.append("<h2>%s</h2>" % esc(title))
        if not items:
            parts.append("<p>(none)</p>")
        for f in items:
            parts.append("<div class='finding %s'><strong>[%s] %s</strong>"
                         "<div class='evidence'>evidence: %s</div>"
                         "<div class='evidence'>why: %s</div>"
                         "<div class='evidence'>next: %s</div></div>"
                         % (esc(f.get("certainty", "unknown")),
                            esc(str(f.get("certainty", "")).upper()),
                            esc(f.get("title", "")),
                            esc(f.get("evidence", "")),
                            esc(f.get("why_it_matters") or "-"),
                            esc(f.get("next_check") or "-")))

    findings = report.get("findings", [])
    emit("Observed facts and changes",
         [f for f in findings
          if f.get("category") not in ("workflow", "log", "coverage")])
    emit("Workflow findings", [f for f in findings if f.get("category") == "workflow"])
    emit("Possible leads from supplied log",
         [f for f in findings if f.get("category") == "log"])
    emit("Missing / incomplete scan information",
         [f for f in findings if f.get("category") == "coverage"])
    parts.append("<p class='meta'>No finding claims that a change caused a "
                 "failure. 'observed' = directly measured; 'possible' = "
                 "consistent with evidence, unproven; 'unknown' = could not "
                 "be determined.</p>")
    parts.append("</body></html>")
    return "".join(parts)
