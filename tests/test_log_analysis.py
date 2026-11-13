import os
import tempfile
import unittest

from comfy_detective import log_analysis
from fixtures import TempDir, make_scan, w


def comparison_with_change(name, before_sha, after_sha):
    return {"sections": {"custom_nodes": {
        "added": [], "removed": [],
        "changed": [{"name": name,
                     "changes": {"commit": {"before": before_sha,
                                            "after": after_sha}}}],
        "unchanged_count": 0}}}


class LogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        self.scan = make_scan()
        self.scan["custom_nodes"].append(
            {"name": "comfyui-coolnode", "path": "", "kind": "dir",
             "git": {"is_git": True}, "requirements": None, "notes": []})

    def tearDown(self):
        self.tmp.cleanup()

    def _analyze(self, text, comparison=None):
        p = w(os.path.join(self.tmp.name, "comfyui.log"), text)
        return log_analysis.analyze_log(p, self.scan, comparison, None)

    def test_module_error_correlates_with_change_as_possible(self):
        log = ("FETCH(clip_vision): ok\n"
               "Cannot import utils\n"
               "Traceback (most recent call last):\n"
               '  File "x", line 1\n'
               "ModuleNotFoundError: No module named 'coolnode'\n")
        comp = comparison_with_change("comfyui-coolnode", "abc123" + "0" * 34,
                                      "def456" + "0" * 34)
        result = self._analyze(log, comp)
        self.assertEqual(result["extracted"]["traceback_count"], 1)
        leads = [f for f in result["findings"] if f.category == "log"]
        lead = [f for f in leads if "coolnode" in f.title]
        self.assertTrue(lead)
        self.assertEqual(lead[0].certainty, "possible")
        self.assertIn("changed from commit", lead[0].evidence)
        self.assertIn("does not establish", lead[0].evidence)
        self.assertNotIn("caused the", lead[0].evidence.replace(
            "does not establish that the change caused the error", ""))

    def test_missing_model_line(self):
        log = ("!! Cannot find model 'missing_face.safetensors' "
               "in the checkpoints folder\n")
        result = self._analyze(log, None)
        self.assertIn("missing_face.safetensors",
                      "".join(result["extracted"]["model_files_in_errors"]))
        titles = [f.title for f in result["findings"]]
        self.assertTrue(any("missing model" in t.lower() for t in titles))

    def test_cannot_import_line(self):
        log = ("Cannot import D:\\ComfyUI\\custom_nodes\\comfyui-coolnode\\"
               "__init__.py module for custom nodes: No module named 'x'\n")
        result = self._analyze(log, None)
        self.assertTrue(result["extracted"]["cannot_import"])

    def test_no_full_log_in_output(self):
        log = "secret path C:\\Users\\bob\\secrets.txt\n" * 100 + \
              "ModuleNotFoundError: No module named 'ghost_pkg'\n"
        result = self._analyze(log, None)
        blob = "".join(f.evidence for f in result["findings"])
        self.assertIn("line ", blob)          # excerpt with line number
        self.assertLess(len(blob), 4000)      # not the whole log
        self.assertNotIn("bob", blob)

    def test_unreadable_log(self):
        p = os.path.join(self.tmp.name, "nope.log")
        result = log_analysis.analyze_log(p, None, None, None)
        self.assertEqual(result["findings"][0].title, "Log file unreadable")


if __name__ == "__main__":
    unittest.main()
