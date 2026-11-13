import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

from comfy_detective import report
from comfy_detective.cli import main
from comfy_detective.models import Finding
from fixtures import TempDir, make_fake_comfy, tmpdir, w, wf_ui


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        self.comfy = os.path.join(self.tmp.name, "comfy")
        make_fake_comfy(self.comfy)
        self.store = os.path.join(self.tmp.name, "store")
        self.out = io.StringIO()

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, argv):
        with redirect_stdout(self.out):
            code = main(argv)
        return code

    def test_scan_exit_zero_and_json(self):
        j = os.path.join(self.tmp.name, "scan.json")
        code = self._run(["scan", "--comfyui", self.comfy,
                          "--json", j])
        self.assertEqual(code, 0)
        text = self.out.getvalue()
        self.assertIn("SUMMARY", text)
        self.assertIn("[OBSERVED]", text)
        with open(j, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["mode"], "scan")
        self.assertTrue(data["findings"])

    def test_snapshot_roundtrip_via_cli(self):
        code = self._run(["snapshot", "create", "--comfyui", self.comfy,
                          "--store", self.store, "--name", "known-good"])
        self.assertEqual(code, 0)
        code = self._run(["snapshot", "list", "--store", self.store])
        self.assertEqual(code, 0)
        self.assertIn("known-good", self.out.getvalue())
        code = self._run(["compare", "--before", "known-good",
                          "--comfyui", self.comfy, "--store", self.store])
        self.assertEqual(code, 0)
        self.assertIn("0 added, 0 removed, 0 changed", self.out.getvalue())

    def test_compare_missing_snapshot_exit_3(self):
        code = self._run(["compare", "--before", "ghost",
                          "--comfyui", self.comfy, "--store", self.store])
        self.assertEqual(code, 3)

    def test_workflow_check_exit_6_on_bad_json(self):
        p = w(os.path.join(self.tmp.name, "bad.json"), "{oops")
        code = self._run(["workflow", "check", p])
        self.assertEqual(code, 6)
        self.assertIn("PARSE ERROR", self.out.getvalue())

    def test_workflow_check_valid_exit_0_with_html(self):
        p = wf_ui(os.path.join(self.tmp.name, "wf.json"))
        h = os.path.join(self.tmp.name, "wf.html")
        code = self._run(["workflow", "check", p, "--comfyui", self.comfy,
                          "--html", h])
        self.assertEqual(code, 0)
        with open(h, encoding="utf-8") as fh:
            html = fh.read()
        self.assertIn("<!doctype html>", html)

    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            with redirect_stdout(self.out):
                main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_html_escapes_user_content(self):
        f = Finding(title="<script>alert(1)</script>", category="workflow",
                    certainty="observed",
                    evidence='evil "quotes" & <b>tags</b>')
        rep = report.build_report("scan", ["<summary & test>"], [f])
        html = report.render_html(rep)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)
        text = report.render_text(rep)
        self.assertIn("[OBSERVED] <script>alert(1)</script>", text)


if __name__ == "__main__":
    unittest.main()
