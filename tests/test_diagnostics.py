import copy
import os
import tempfile
import unittest

from comfy_detective import diagnostics
from fixtures import make_scan, tmpdir, w


def wrap(scan):
    return {"schema_version": 1, "snapshot_id": "x", "created_at": "t",
            "scan": scan}


class CompareTests(unittest.TestCase):
    def setUp(self):
        self.base = make_scan()

    def _cmp(self, after):
        return diagnostics.compare_snapshots(wrap(self.base), wrap(after),
                                             "before", "after")

    def test_identical_no_findings(self):
        comp = self._cmp(copy.deepcopy(self.base))
        self.assertEqual(comp["findings"], [])

    def test_reordered_lists_are_noise_free(self):
        after = copy.deepcopy(self.base)
        after["packages"] = list(reversed(after["packages"]))
        after["custom_nodes"] = list(reversed(after["custom_nodes"]))
        self.assertEqual(self._cmp(after)["findings"], [])

    def test_commit_change_observed(self):
        after = copy.deepcopy(self.base)
        for n in after["custom_nodes"]:
            if n["git"]["commit"] == "b" * 40:
                n["git"]["commit"] = "c" * 40
        comp = self._cmp(after)
        node_changed = [f for f in comp["findings"] if "nodea" in f.title]
        self.assertTrue(node_changed)
        self.assertEqual(node_changed[0].certainty, "observed")
        self.assertIn("commit", node_changed[0].evidence)

    def test_add_remove_nodes(self):
        after = copy.deepcopy(self.base)
        after["custom_nodes"].append({"name": "nodec", "path": "",
                                      "kind": "dir", "git": {"is_git": True},
                                      "requirements": None, "notes": []})
        after["custom_nodes"] = [n for n in after["custom_nodes"]
                                 if n["name"] != "nodeb"]
        comp = self._cmp(after)
        titles = " | ".join(f.title for f in comp["findings"])
        self.assertIn("Custom node added: nodec", titles)
        self.assertIn("Custom node removed: nodeb", titles)

    def test_package_versions(self):
        after = copy.deepcopy(self.base)
        after["packages"] = [{"name": "alpha", "version": "1.0.1"},
                             {"name": "beta", "version": "2.0.0"},
                             {"name": "newlib", "version": "0.1"}]
        comp = self._cmp(after)
        pkg = [f for f in comp["findings"] if f.category == "packages"]
        self.assertEqual(len(pkg), 1)
        self.assertIn("+newlib", pkg[0].evidence)
        self.assertIn("alpha 1.0.0->1.0.1", pkg[0].evidence)
        sec = comp["sections"]["packages"]
        self.assertEqual(sec["changed"][0]["name"], "alpha")

    def test_coverage_newly_observed_not_change(self):
        before = copy.deepcopy(self.base)
        before["unavailable"] = [{"field": "packages", "reason": "no venv"}]
        before["packages"] = []
        comp = diagnostics.compare_snapshots(wrap(before), wrap(self.base),
                                             "before", "after")
        self.assertEqual(comp["sections"]["coverage"]["newly_observed"],
                         ["packages"])
        self.assertFalse(comp["sections"]["packages"].get("changed"))
        cov = [f for f in comp["findings"] if f.category == "coverage"]
        self.assertIn("newly observed", cov[0].title.lower())

    def test_model_ref_status_change_and_mtime(self):
        wf_before = {"path": "/w/ui.json", "format": "ui", "parse_error": None,
                     "node_classes": ["A"], "node_count": 1, "edge_count": 0,
                     "model_refs": [
                         {"value": "m1.safetensors", "status": "found",
                          "resolved_path": "/models/m1.safetensors",
                          "size": 100, "mtime": 1, "searched": [], "notes": [],
                          "node_type": "A", "input_name": "ckpt_name",
                          "category": "checkpoints", "sha256": None}],
                     "coverage_notes": []}
        wf_after = copy.deepcopy(wf_before)
        b = wf_after["model_refs"][0]
        b["size"] = 100
        b["mtime"] = 2  # mtime only
        comp = self._cmp({**copy.deepcopy(self.base), "workflows": [wf_before]})
        # only one side has workflow: no change claims
        self.assertEqual([f for f in comp["findings"]
                          if f.category == "workflow"], [])
        comp2 = diagnostics.compare_snapshots(
            {**wrap({**copy.deepcopy(self.base), "workflows": [wf_before]})},
            {**wrap({**copy.deepcopy(self.base), "workflows": [wf_after]})},
            "b", "a")
        wf_findings = [f for f in comp2["findings"] if f.category == "workflow"]
        self.assertEqual(len(wf_findings), 1)
        self.assertEqual(wf_findings[0].certainty, "possible")
        self.assertIn("mtime", wf_findings[0].title)


class RequirementsTests(unittest.TestCase):
    def test_compatible_release_operator(self):
        from comfy_detective.diagnostics import _spec_satisfied
        # ~=1.4.5 -> >=1.4.5, <1.5
        self.assertTrue(_spec_satisfied("~=", "1.4.5", "1.4.5"))
        self.assertTrue(_spec_satisfied("~=", "1.4.5", "1.4.9"))
        self.assertFalse(_spec_satisfied("~=", "1.4.5", "1.5.0"))
        self.assertFalse(_spec_satisfied("~=", "1.4.5", "1.4.4"))
        self.assertFalse(_spec_satisfied("~=", "1.4.5", "2.0.0"))
        # ~=1.4 -> >=1.4, <2
        self.assertTrue(_spec_satisfied("~=", "1.4", "1.9"))
        self.assertFalse(_spec_satisfied("~=", "1.4", "2.0"))
        self.assertFalse(_spec_satisfied("~=", "1.4", "1.3"))
        # >= lower bound must hold too: ~=1.4.5 with 1.4 fails low
        self.assertFalse(_spec_satisfied("~=", "1.4.5", "1.4"))
        # uncomparable forms stay unknown, never guessed
        self.assertIsNone(_spec_satisfied("~=", "1.4.5", "weird??"))

    def test_tilde_requirement_end_to_end(self):
        with tmpdir() as d:
            p = w(os.path.join(d, "requirements.txt"),
                  "alpha~=1.4.5\nbeta~=2.1\n")
            findings = diagnostics.requirement_findings(
                p, "custom_node",
                {"alpha": "1.5.0",   # violates ~=1.4.5 upper bound
                 "beta": "2.9"})     # satisfies ~=2.1
            titles = [f.title for f in findings]
            self.assertTrue(any("does not satisfy declaration: alpha" in t
                                for t in titles))
            self.assertFalse(any("alpha" in t and "unknown" in t
                                 for t in titles))
            self.assertFalse(any("beta" in t for t in titles))

    def test_parse(self):
        reqs, skipped = diagnostics.parse_requirements(
            "Alpha [extra] == 1.2.3\n"
            "beta>=2.0,<3.0 ; python_version<'3.12'\n"
            "gamma\n"
            "# comment\n"
            "-r other.txt\n"
            "git+https://example.invalid/x.git\n")
        names = [r[0] for r in reqs]
        self.assertEqual(names, ["Alpha", "beta", "gamma"])
        self.assertEqual(reqs[0][1], [("==", "1.2.3")])
        self.assertEqual(reqs[1][1], [(">=", "2.0"), ("<", "3.0")])
        self.assertTrue(reqs[1][2])   # marker flag
        self.assertEqual(len(skipped), 2)

    def test_findings(self):
        with tmpdir() as d:
            p = w(os.path.join(d, "requirements.txt"),
                  "alpha==9.9.9\nbeta>=2.0\nnotinstalled==1.0\n"
                  "marked==1.0 ; sys_platform=='darwin'\n")
            findings = diagnostics.requirement_findings(
                p, "custom_node", {"alpha": "1.0.0", "beta": "2.5"})
            titles = [f.title for f in findings]
            self.assertTrue(any("does not satisfy declaration: alpha" in t
                                for t in titles))
            self.assertFalse(any("beta" in t for t in titles))
            poss = [f for f in findings if "notinstalled" in f.title]
            self.assertEqual(poss[0].certainty, "possible")
            self.assertFalse(any("marked" in t for t in titles))


class StateTests(unittest.TestCase):
    def test_dirty_and_nongit_aggregates(self):
        scan = make_scan()
        scan["custom_nodes"][0]["git"]["dirty"] = True
        findings = diagnostics.state_findings(scan)
        titles = [f.title for f in findings]
        self.assertTrue(any("uncommitted changes" in t for t in titles))
        self.assertTrue(any("not under git" in t for t in titles))


if __name__ == "__main__":
    unittest.main()
