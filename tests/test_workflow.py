import json
import os
import tempfile
import unittest

from comfy_detective import workflow as wf
from fixtures import TempDir, make_fake_comfy, tmpdir, w, wf_api, wf_ui


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_ui_format(self):
        p = wf_ui(os.path.join(self.d, "ui.json"))
        info = wf.parse_workflow(p)
        self.assertEqual(info.format, "ui")
        self.assertEqual(info.node_count, 3)
        self.assertEqual(info.edge_count, 1)
        self.assertEqual(info.node_classes,
                         ["CheckpointLoaderSimple", "LoraLoader", "VAELoader"])
        refs = {r.value: r for r in info.model_refs}
        self.assertEqual(refs["m1.safetensors"].category, "checkpoints")
        self.assertEqual(refs["missing_vae.safetensors"].category, "vae")

    def test_api_format_and_subpath(self):
        p = wf_api(os.path.join(self.d, "api.json"))
        info = wf.parse_workflow(p)
        self.assertEqual(info.format, "api")
        self.assertEqual(info.node_count, 2)
        refs = {r.value: r for r in info.model_refs}
        self.assertIn("sub/nested.safetensors", refs)

    def test_invalid_json_message(self):
        p = w(os.path.join(self.d, "bad.json"), '{"nodes": [1, ')
        info = wf.parse_workflow(p)
        self.assertIsNotNone(info.parse_error)
        self.assertIn("invalid JSON", info.parse_error)
        self.assertIn("line", info.parse_error)

    def test_unknown_structure(self):
        p = w(os.path.join(self.d, "weird.json"), json.dumps({"foo": "bar"}))
        info = wf.parse_workflow(p)
        self.assertEqual(info.format, "unknown")
        self.assertTrue(info.coverage_notes)

    def test_wildcard_not_resolved_as_missing(self):
        p = w(os.path.join(self.d, "wc.json"), json.dumps({
            "1": {"class_type": "SomethingLoader",
                  "inputs": {"lora_name": "*.safetensors"}}}))
        info = wf.parse_workflow(p)
        ref = info.model_refs[0]
        self.assertEqual(ref.status, "unknown")


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        info = make_fake_comfy(os.path.join(self.tmp.name, "c"), with_git=False)
        from comfy_detective.scanner import collect_model_dirs
        self.model_dirs = collect_model_dirs(info["root"])
        self.info = info

    def tearDown(self):
        self.tmp.cleanup()

    def _resolve(self, p):
        info = wf.parse_workflow(p)
        wf.resolve_workflow_models(info, self.model_dirs)
        return {r.value: r for r in info.model_refs}

    def test_found_missing_and_extra_paths(self):
        p = wf_ui(os.path.join(self.tmp.name, "ui.json"))
        refs = self._resolve(p)
        self.assertEqual(refs["m1.safetensors"].status, "found")
        self.assertEqual(refs["missing_vae.safetensors"].status, "unknown")
        # extra_model_paths category loras → found
        self.assertEqual(refs["extra_lora.safetensors"].status, "found")

    def test_subfolder_found(self):
        p = wf_api(os.path.join(self.tmp.name, "api.json"))
        refs = self._resolve(p)
        self.assertEqual(refs["sub/nested.safetensors"].status, "found")

    def test_ambiguous_not_guessed(self):
        croot = self.info["root"]
        # same plain name in two dirs of the same (aliased) category:
        # checkpoints and diffusion_models are the same lookup category
        w(os.path.join(croot, "models", "checkpoints", "dup.safetensors"), "1")
        w(os.path.join(croot, "models", "diffusion_models",
                       "dup.safetensors"), "2")
        p = w(os.path.join(self.tmp.name, "dup.json"), json.dumps({
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "dup.safetensors"}}}))
        refs = self._resolve(p)
        self.assertEqual(refs["dup.safetensors"].status, "ambiguous")

    def test_missing_claims_list_searched(self):
        p = w(os.path.join(self.tmp.name, "miss.json"), json.dumps({
            "1": {"class_type": "CheckpointLoaderSimple",
                  "inputs": {"ckpt_name": "definitely_absent_xyz.safetensors"}}}))
        refs = self._resolve(p)
        r = refs["definitely_absent_xyz.safetensors"]
        self.assertEqual(r.status, "missing")
        self.assertTrue(r.searched)


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_patterns(self):
        root = os.path.join(self.d, "c")
        make_fake_comfy(root, with_git=False)
        # v3 style file
        w(os.path.join(root, "comfy_extras", "nodes_v3.py"),
          "class FooExtension:\n"
          "    async def get_node_list(self):\n"
          "        return []\n"
          "async def comfy_entrypoint():\n"
          "    return FooExtension()\n")
        # syntax error file
        w(os.path.join(root, "custom_nodes", "goodnode", "broken.py"),
          "def (:\n")
        inv = wf.build_node_inventory(root)
        types = inv["types"]
        self.assertIn("KSampler", types)        # core literal
        self.assertIn("ExtraNode", types)       # extras literal
        self.assertIn("GoodNode", types)        # custom node literal
        self.assertIn("PlainNode", types)       # .update() pattern
        meta = inv["meta"]
        self.assertGreaterEqual(meta["files_failed"], 1)
        self.assertTrue(any("nodes_v3.py" in f for f in
                            meta["dynamic_mapping_files"]))

    def test_missing_type_finding_is_possible_when_partial(self):
        p = w(os.path.join(self.d, "wf.json"), json.dumps({
            "1": {"class_type": "NoSuchNodeAnywhereXYZ", "inputs": {}}}))
        info = wf.parse_workflow(p)
        inv = {"types": {"KSampler"},
               "meta": {"files_parsed": 1, "files_failed": 2,
                        "dynamic_mapping_files": ["x.py"], "files_scanned": 3,
                        "notes": []}}
        findings = wf.workflow_findings(info, inv)
        t = [f for f in findings if "not found" in f.title][0]
        self.assertEqual(t.certainty, "possible")

    def test_no_inventory_no_missing_claim(self):
        p = w(os.path.join(self.d, "wf.json"), json.dumps({
            "1": {"class_type": "NoSuchNodeAnywhereXYZ", "inputs": {}}}))
        info = wf.parse_workflow(p)
        findings = wf.workflow_findings(info, None)
        self.assertFalse(any("not found" in f.title for f in findings))


if __name__ == "__main__":
    unittest.main()
