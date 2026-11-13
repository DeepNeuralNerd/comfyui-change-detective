"""Typed data structures. Snapshots serialize these with ``dataclasses.asdict``.

Convention that keeps reports honest: a field that could not be collected is
``None`` AND has a matching entry in ``ScanResult.unavailable``. Comparison
uses that list so "newly observed" is never rendered as "changed".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CERTAINTY_LEVELS = ("observed", "possible", "unknown")


def to_dict(obj: Any) -> Any:
    """Dataclass-aware JSON-safe conversion."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


@dataclass
class Unavailable:
    """One piece of information the scan could not collect, with the reason."""
    field: str
    reason: str


@dataclass
class GitInfo:
    repo_path: str
    is_git: bool = False
    commit: Optional[str] = None
    branch: Optional[str] = None
    detached: bool = False
    dirty: Optional[bool] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class Environment:
    python_exe: Optional[str] = None
    venv_dir: Optional[str] = None
    site_packages: Optional[str] = None
    pyvenv: Dict[str, str] = field(default_factory=dict)


@dataclass
class PackageInfo:
    name: str
    version: str
    source: str = "dist-info"  # dist-info | dist-info-dirname


@dataclass
class DeclaredFile:
    path: str
    exists: bool
    role: str  # requirements | pyproject | extra_model_paths | node_requirements


@dataclass
class CustomNode:
    name: str
    path: str
    kind: str = "dir"  # dir | file
    git: Optional[GitInfo] = None
    requirements: Optional[str] = None  # path when a requirements.txt exists
    notes: List[str] = field(default_factory=list)


@dataclass
class ModelRef:
    """One model file reference extracted from a workflow."""
    value: str
    node_type: str = ""
    input_name: Optional[str] = None
    category: Optional[str] = None       # guessed category
    status: Optional[str] = None         # found | missing | ambiguous | unknown
    resolved_path: Optional[str] = None
    size: Optional[int] = None
    mtime: Optional[int] = None
    sha256: Optional[str] = None
    searched: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class WorkflowInfo:
    path: str
    format: str = "unknown"              # ui | api | unknown
    parse_error: Optional[str] = None
    node_classes: List[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    model_refs: List[ModelRef] = field(default_factory=list)
    coverage_notes: List[str] = field(default_factory=list)
    subgraphs: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ScanResult:
    comfyui_root: str
    scan_time: str = ""
    comfy_version: Optional[str] = None
    comfy_git: Optional[GitInfo] = None
    environment: Environment = field(default_factory=Environment)
    packages: List[PackageInfo] = field(default_factory=list)
    declared_files: List[DeclaredFile] = field(default_factory=list)
    custom_nodes: List[CustomNode] = field(default_factory=list)
    workflows: List[WorkflowInfo] = field(default_factory=list)
    model_dirs: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    unavailable: List[Unavailable] = field(default_factory=list)


@dataclass
class Snapshot:
    snapshot_id: str
    created_at: str
    scan: Dict[str, Any]
    schema_version: int = 1
    tool_version: str = ""
    name: Optional[str] = None


@dataclass
class Finding:
    """One diagnostic finding. Language must separate evidence from theory:
    never 'caused', only 'possible lead' unless causation was established
    (which v1 never does)."""
    title: str
    category: str            # comfyui | custom_node | packages | environment | workflow | log | coverage | input
    certainty: str           # observed | possible | unknown
    evidence: str
    why_it_matters: str = ""
    next_check: str = ""
    refs: List[str] = field(default_factory=list)
