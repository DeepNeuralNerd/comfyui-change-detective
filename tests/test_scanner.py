import os
import tempfile
import unittest

from comfy_detective import diagnostics
from comfy_detective.models import to_dict
from comfy_detective.scanner import parse_extra_model_paths, scan_install
from comfy_detective.util import ToolError
from fixtures import TempDir, make_fake_comfy, tmpdir, w


class ScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TempDir()
        cls.info = make_fake_comfy(os.path.join(cls.tmp.name, "comfy"))
        cls.scan = to_dict(scan_install(cls.info["root"],
                                        allow_git_cmd=False))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_version_and_git(self):
        self.assertEqual(self.scan["comfy_version"], "0.99.0")
        git = self.scan["comfy_git"]
        self.assertTrue(git["is_git"])
        self.assertEqual(git["commit"], self.info.get("comfy_commit"))

    def test_packages_from_dist_info(self):
        names = {p["name"]: p["version"] for p in self.scan["packages"]}
        self.assertEqual(names.get("alpha"), "1.0.0")
        self.assertEqual(names.get("beta"), "1.9.0")
        self.assertEqual(names.get("other-pkg"), "2.1")

    def test_environment_detected(self):
        env = self.scan["environment"]
        self.assertTrue(env["python_exe"].endswith("python.exe"))
        self.assertEqual(env["pyvenv"]["version"], "3.12.4")

    def test_custom_nodes(self):
        nodes = {n["name"]: n for n in self.scan["custom_nodes"]}
        self.assertIn("goodnode", nodes)
        self.assertIn("plainnode", nodes)
        self.assertNotIn(".disabled", nodes)
        self.assertNotIn("__pycache__", nodes)
        self.assertEqual(nodes["loose_node"]["kind"], "file")
        self.assertIsNotNone(nodes["goodnode"]["requirements"])
        # plainnode has no git → note
        self.assertFalse(nodes["plainnode"]["git"]["is_git"])
        # hidden skip is reported as a warning, not silently dropped
        self.assertTrue(any("hidden entry" in w for w in self.scan["warnings"]))

    def test_model_dirs_include_extra(self):
        self.assertIn("checkpoints", self.scan["model_dirs"])
        loras = self.scan["model_dirs"].get("loras", [])
        self.assertTrue(any(p.endswith("Loras") for p in loras))

    def test_pyvenv_version_info_normalized(self):
        # uv-created venvs write version_info, not version
        with tmpdir() as d:
            make_fake_comfy(d, with_git=False)
            w(os.path.join(d, "venv", "pyvenv.cfg"),
              "home = X\nuv = 0.9.30\nversion_info = 3.12.12\n")
            sd = to_dict(scan_install(d, allow_git_cmd=False))
            self.assertEqual(sd["environment"]["pyvenv"]["version"],
                             "3.12.12")

    def test_missing_root_raises(self):
        with self.assertRaises(ToolError) as ctx:
            scan_install(os.path.join(self.info["root"], "nope"))
        self.assertEqual(ctx.exception.code, 4)

    def test_no_venv_marks_unavailable_not_empty(self):
        with tmpdir() as d:
            make_fake_comfy(os.path.join(d, "c"), with_git=False,
                            with_venv=False)
            sd = to_dict(scan_install(os.path.join(d, "c"),
                                      allow_git_cmd=False))
            fields = {u["field"] for u in sd["unavailable"]}
            self.assertIn("environment", fields)
            self.assertEqual(sd["packages"], [])
            dep = diagnostics.dependency_findings(sd)
            self.assertEqual(dep[0].title, "Dependency checks skipped")

    def test_extra_model_paths_block_style(self):
        with tmpdir() as d:
            p = os.path.join(d, "extra_model_paths.yaml")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("mysave:\n  is_default: true\n  base_path: %s\n"
                         "  loras: loras\n" % d.replace("\\", "/"))
            parsed, notes = parse_extra_model_paths(p, d)
            self.assertIn("loras", parsed)
            self.assertTrue(parsed["loras"][0].endswith(os.path.join(
                os.path.basename(d), "loras")))


if __name__ == "__main__":
    unittest.main()
