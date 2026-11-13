"""Fixture builders for the test-suite. All fixtures live under a temp dir;
nothing here touches a real ComfyUI installation."""

from __future__ import annotations

import atexit
import itertools
import json
import os
import shutil
import subprocess
import tempfile

GIT = shutil.which("git")

_TMP_ROOT = None
_case = itertools.count()


class TempDir:
    """Sandbox-friendly stand-in for tempfile.TemporaryDirectory.

    Under some sandboxed setups the tmpXXXXXXXX directory created by mkdtemp
    becomes inaccessible right after creation; fixed-name nested directories
    are safe, so use a per-process run root plus numbered case dirs.
    """

    def __init__(self) -> None:
        global _TMP_ROOT
        if _TMP_ROOT is None:
            _TMP_ROOT = os.path.join(tempfile.gettempdir(),
                                     "cd-tests-%d" % os.getpid())
            atexit.register(shutil.rmtree, _TMP_ROOT, True)
        self.name = os.path.join(_TMP_ROOT, "case%05d" % next(_case))
        os.makedirs(self.name, exist_ok=True)

    def cleanup(self) -> None:
        shutil.rmtree(self.name, ignore_errors=True)

    def __enter__(self) -> "TempDir":
        return self

    def __exit__(self, *exc) -> bool:
        self.cleanup()
        return False


try:
    from contextlib import contextmanager

    @contextmanager
    def tmpdir():
        """Context manager yielding a fresh case dir path (like
        TemporaryDirectory used as ``as d``)."""
        t = TempDir()
        try:
            yield t.name
        finally:
            t.cleanup()
except ImportError:  # pragma: no cover
    pass


def w(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def git_run(cwd: str, *args: str) -> None:
    subprocess.run(["git", "-c", "user.email=test@example.invalid",
                    "-c", "user.name=Test", "-C", cwd] + list(args),
                   check=True, capture_output=True)


def git_init_commit(path: str, msg: str = "init") -> str:
    git_run(path, "init", "-q", "-b", "master", ".")
    git_run(path, "add", "-A")
    git_run(path, "commit", "-q", "-m", msg)
    out = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def make_fake_comfy(root: str, with_git: bool = True,
                    with_venv: bool = True) -> dict:
    """Small but structurally realistic ComfyUI layout."""
    info: dict = {"root": root}
    w(os.path.join(root, "comfyui_version.py"),
      '__version__ = "0.99.0"\n')
    w(os.path.join(root, "requirements.txt"),
      "# deps\nalpha==1.0.0\nbeta>=2.0.0\ngamma[extra]<3\n"
      "-r other.txt\nhttps://example.invalid/pkg.zip\n"
      "delta ; python_version < '3.9'\n")
    w(os.path.join(root, "nodes.py"),
      'NODE_CLASS_MAPPINGS = {"KSampler": object, "CheckpointLoaderSimple": object}\n')
    os.makedirs(os.path.join(root, "comfy_extras"), exist_ok=True)
    w(os.path.join(root, "comfy_extras", "nodes_x.py"),
      'NODE_CLASS_MAPPINGS = {"ExtraNode": object}\n')

    if with_venv:
        w(os.path.join(root, "venv", "pyvenv.cfg"),
          "home = C:\\python\nversion = 3.12.4\n")
        w(os.path.join(root, "venv", "Scripts", "python.exe"), "fake")
        sp = os.path.join(root, "venv", "Lib", "site-packages")
        w(os.path.join(sp, "alpha-1.0.0.dist-info", "METADATA"),
          "Name: alpha\nVersion: 1.0.0\n\nbody\n")
        w(os.path.join(sp, "beta-1.9.0.dist-info", "METADATA"),
          "Name: beta\nVersion: 1.9.0\n\nbody\n")
        w(os.path.join(sp, "other_pkg-2.1.dist-info", "METADATA"),
          "Name: other-pkg\nVersion: 2.1\n\nbody\n")

    cn = os.path.join(root, "custom_nodes")
    w(os.path.join(cn, "goodnode", "__init__.py"),
      'NODE_CLASS_MAPPINGS = {"GoodNode": object}\n')
    w(os.path.join(cn, "goodnode", "requirements.txt"), "gamma\n")
    w(os.path.join(cn, "plainnode", "__init__.py"),
      'NODE_CLASS_MAPPINGS.update({"PlainNode": object})\n')
    w(os.path.join(cn, ".disabled", "oldnode", "__init__.py"), "x=1\n")
    w(os.path.join(cn, "__pycache__", "x.py"), "x=1\n")
    w(os.path.join(cn, "loose_node.py"), "x=1\n")

    loras_dir = os.path.join(root, "_assets", "Loras")
    w(os.path.join(root, "models", "checkpoints", "m1.safetensors"), "M" * 10)
    w(os.path.join(root, "models", "checkpoints", "sub", "nested.safetensors"), "N" * 7)
    # real ComfyUI ships these category dirs; exist at scan time
    os.makedirs(os.path.join(root, "models", "diffusion_models"), exist_ok=True)
    w(os.path.join(loras_dir, "extra_lora.safetensors"), "L" * 5)
    w(os.path.join(root, "extra_model_paths.yaml"),
      '{sm: {loras: "%s"}}' % loras_dir.replace("\\", "/"))

    if with_git and GIT:
        git_init_commit(cn, "nodes-init")
        info["goodnode_commit"] = git_init_commit(
            os.path.join(root, "custom_nodes", "goodnode"), "goodnode-init")
        info["comfy_commit"] = git_init_commit(root, "comfy-init")
    return info


def wf_ui(path: str, extra_nodes: str = "") -> str:
    nodes = [
        {"id": 1, "type": "CheckpointLoaderSimple",
         "inputs": [{"name": "ckpt_name", "type": "COMBO", "value": "m1.safetensors"},
                    {"name": "MODEL", "type": "MODEL", "link": None}],
         "outputs": []},
        {"id": 2, "type": "VAELoader",
         "inputs": [{"name": "vae_name", "type": "COMBO", "value": "missing_vae.safetensors"},
                    {"name": "vae", "type": "VAE", "link": 3}],
         "outputs": []},
        {"id": 3, "type": "LoraLoader",
         "inputs": [{"name": "lora_name", "type": "COMBO", "value": "extra_lora.safetensors"}],
         "outputs": []},
    ]
    if extra_nodes:
        nodes.append({"id": 4, "type": extra_nodes, "inputs": [], "outputs": []})
    return w(path, json.dumps(nodes))


def wf_api(path: str) -> str:
    return w(path, json.dumps({
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "sub/nested.safetensors"}},
        "2": {"class_type": "KSampler", "inputs": {"seed": 1, "model": ["1", 0]}},
    }))


def make_scan(with_git=True, with_venv=True) -> dict:
    """Build a minimal scan dict in snapshot-JSON shape for compare tests."""
    return {
        "comfyui_root": "/fake/root",
        "scan_time": "2026-01-01T00:00:00+00:00",
        "comfy_version": "0.45.0",
        "comfy_git": {"repo_path": "/fake/root", "is_git": with_git,
                      "commit": "a" * 40 if with_git else None,
                      "branch": "master" if with_git else None,
                      "detached": False, "dirty": False, "notes": []},
        "environment": {"python_exe": "/fake/root/venv/bin/python",
                        "venv_dir": "/fake/root/venv",
                        "site_packages": "/fake/root/venv/lib/site-packages",
                        "pyvenv": {"version": "3.12.4"}},
        "packages": [{"name": "alpha", "version": "1.0.0", "source": "dist-info"},
                     {"name": "beta", "version": "2.0.0", "source": "dist-info"}],
        "declared_files": [],
        "custom_nodes": [
            {"name": "nodea", "path": "/fake/root/custom_nodes/nodea",
             "kind": "dir",
             "git": {"repo_path": "", "is_git": True, "commit": "b" * 40,
                     "branch": "main", "detached": False, "dirty": False,
                     "notes": []},
             "requirements": None, "notes": []},
            {"name": "nodeb", "path": "/fake/root/custom_nodes/nodeb",
             "kind": "dir",
             "git": {"repo_path": "", "is_git": False, "commit": None,
                     "branch": None, "detached": False, "dirty": None,
                     "notes": ["no .git"]},
             "requirements": None, "notes": []},
        ],
        "workflows": [],
        "model_dirs": {},
        "warnings": [],
        "unavailable": [],
    }
