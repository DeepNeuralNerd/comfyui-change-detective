# ComfyUI Change Detective

A small, **read-only** diagnostic CLI for one question:

> *"This workflow used to work. What changed, and what should I check first?"*

It scans a ComfyUI installation (works fine while ComfyUI is down, e.g.
Stability Matrix's managed install), saves snapshots of what it sees,
compares them, inspects workflow JSON against the installation, and — if
you supply an error log — turns log errors into evidence-ranked *leads*.

It never claims a change **caused** a failure. Every finding carries a
certainty label:

| label | meaning |
|---|---|
| `observed` | directly measured now (file present/absent, commit differs, version mismatch) |
| `possible` | consistent with the evidence, but not proven (a log error near a change, a missing declared dependency that may be vendored) |
| `unknown` | could not be determined (data not collected, dynamic registration, truncated log) |

Missing or uncollectable information is reported as such — never as
"unchanged" or "healthy".

## Install

Pure Python ≥ 3.9, stdlib only — no dependencies, no network, nothing to
build. Either run from a checkout:

```powershell
set PYTHONPATH=src
python -m comfy_detective --help
```

or install it (console script `comfy-detective`):

```powershell
pip install -e .
```

## Usage

```powershell
# Look at the installation now (Stability Matrix managed ComfyUI example)
comfy-detective scan --comfyui "D:\StabilityMatrix\Data\Packages\ComfyUI"

# Save the current state as a named snapshot ("known-good")
comfy-detective snapshot create --comfyui "D:\...\ComfyUI" --name known-good

# What changed since then? (fresh scan vs snapshot)
comfy-detective compare --before known-good --comfyui "D:\...\ComfyUI"

# Same, plus a workflow check and an error log you supply
comfy-detective compare --before known-good --comfyui "D:\...\ComfyUI" `
    --workflow "my_workflow.json" --log "comfyui_error.log"

# Machine-readable outputs (written; HTML is never auto-opened)
comfy-detective compare --before known-good --comfyui "..." `
    --json report.json --html report.html

# Check one workflow without a comparison
comfy-detective workflow check "my_workflow.json" --comfyui "D:\...\ComfyUI"

# Supplement the static node inventory with an object_info dump
# (GET http://127.0.0.1:8188/object_info while ComfyUI runs, saved to a file)
comfy-detective workflow check "my_workflow.json" --comfyui "..." --objectinfo object_info.json
```

Other commands/flags: `snapshot list`, `compare --before A --after B`
(snapshot-to-snapshot), `--hash-models` (opt-in sha256 of workflow-referenced
models; slow on big files), `--no-git-status` (skips `git status` runs; dirty
state then honestly reported as *unknown*).

## What it collects (all read-only)

* ComfyUI version (`comfyui_version.py` / `pyproject.toml`), git commit,
  branch, detached HEAD, dirty flag (`git status --porcelain` when git is
  available; failure ⇒ unknown, never guessed clean).
* The venv Python and installed distributions, enumerated from
  `*.dist-info/METADATA` (no imports, no `pip` execution).
* Custom nodes: directories (plus loose `.py` files) under `custom_nodes/`,
  each one's git state and `requirements.txt` (parsed, never installed);
  dot-directories (`.disabled`) are skipped and counted, not silently hidden.
* Model directories: `models/*` plus `extra_model_paths.yaml` (both block
  and flow styles, including Stability Matrix's single-line mapping).
* Supplied workflow JSONs (UI export or API export): node types, edges, and
  model references resolved against the model directories. Exact-name hits
  and bounded subfolder searches only — ambiguous names are reported as
  ambiguous with all candidate locations listed, and "missing" is claimed
  only when the searched locations are listed for you to check.

### Comparisons

Snapshot ↔ snapshot or snapshot ↔ fresh scan. Additions, removals and
changes are separated; paths are normalized and output ordering is
deterministic. When one side could not collect something (no venv, git
missing), the gap is reported as *newly observed* / *no longer observed*,
not as a change. File identity is path+size+mtime; an mtime-only change is
flagged `possible` ("touched"), never asserted as a content change.
`--hash-models` adds opt-in content hashing for workflow-referenced models.

### Workflows & node inventory

Node types are checked against a **static** inventory built by parsing
`NODE_CLASS_MAPPINGS` literals (and `.update()` dicts) in the installation's
Python files. Files that register nodes dynamically — including the newer
`comfy_entrypoint`/`get_node_list` v3 API style — are counted and reported;
coverage stays `possible`, never `missing`, while any file failed to parse or
uses dynamic registration. `--objectinfo` (a saved `object_info` response)
gives a reliable inventory when ComfyUI is reachable. Model inputs are
matched by exact loader-name tables first, keyword hints second, and widget
values in UI exports are explicitly *not* interpreted.

### Logs

`--log <file>` extracts import errors, missing-model messages and node-class
names from a log you supply, correlates them with observed changes
("the custom node that errored changed from commit A to B"), and emits
findings like `Possible lead: …` with the evidence chain and quoted line
excerpts. Correlation is not causation — the report says so.

## Where snapshots live

Default store: `%LOCALAPPDATA%\ComfyUIChangeDetective\snapshots`
(override: `--store DIR` or env `COMFY_DETECTIVE_STORE`).
Snapshots are written atomically, schema-versioned, named `<id>.json`
with an optional human name; newer schema versions and corrupt files are
listed with an error, never silently skipped. **Nothing is ever stored
inside the ComfyUI installation.**

## Exit codes

| code | meaning |
|---|---|
| 0 | command succeeded (diagnostic *findings* are not errors) |
| 2 | usage error |
| 3 | snapshot problems (unknown/ambiguous/unreadable) |
| 4 | ComfyUI path invalid |
| 6 | supplied workflow/log unreadable or unparsable |
| 1 | unexpected error (re-run with `--debug` for the traceback) |

## Read-only guarantees

The tool opens files for reading, spawns `git status` (read-only; it may
refresh a repo index, so `--no-git-status` exists), and writes only to the
snapshot store and the `--json/--html` paths you choose. It never modifies
the ComfyUI package, `Data\Packages`, `Data\Assets`, or anything under your
ComfyUI root. No network access, no installs, no admin rights.

## Limitations (honest list)

* Dirty-state checks shell out to `git`; without git they are `unknown`.
* Node inventory is static analysis; v3-API/dynamic registrations are
  coverage gaps, not "missing" claims.
* UI-format exports store most widget values positionally; only values
  that carry input names are checked for model references (said in the
  report too).
* Searches into model dirs are bounded (entry caps); a hit beyond the
  budget is reported as `unknown` with a note, not "missing".
* Symlinks are not followed recursively.
* Log analysis is pattern-based extraction, not full parsing; truncated
  reads are labeled.

## Development

```powershell
python run_tests.py
```

(or manually: `set PYTHONPATH=src;tests` then
`python -m unittest discover -s tests -t tests`)

Tests build throwaway fixture trees in temp dirs; they never touch a real
ComfyUI installation and never mutate one.
