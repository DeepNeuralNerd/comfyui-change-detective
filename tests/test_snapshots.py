import json
import os
import tempfile
import unittest

from comfy_detective import snapshots
from comfy_detective.models import ScanResult, to_dict
from comfy_detective.scanner import scan_install
from comfy_detective.util import ToolError
from fixtures import TempDir, make_fake_comfy, tmpdir


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TempDir()
        self.store = os.path.join(self.tmp.name, "store")
        comfy = os.path.join(self.tmp.name, "comfy")
        make_fake_comfy(comfy)
        self.scan = scan_install(comfy, allow_git_cmd=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_list_resolve(self):
        sid, path = snapshots.create_snapshot(self.store, self.scan,
                                              name="known-good")
        self.assertTrue(os.path.isfile(path))
        entries = snapshots.list_snapshots(self.store)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["snapshot_id"], sid)
        self.assertEqual(entries[0]["name"], "known-good")
        # resolve by id, name, unique prefix
        by_id = snapshots.resolve_snapshot(self.store, sid)
        by_name = snapshots.resolve_snapshot(self.store, "KNOWN-GOOD")
        by_prefix = snapshots.resolve_snapshot(self.store, sid[:12])
        self.assertEqual(by_id["snapshot_id"], sid)
        self.assertEqual(by_name["snapshot_id"], sid)
        self.assertEqual(by_prefix["snapshot_id"], sid)

    def test_schema_roundtrip(self):
        sid, path = snapshots.create_snapshot(self.store, self.scan)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["scan"]["comfyui_root"],
                         to_dict(self.scan)["comfyui_root"])

    def test_malformed_files_reported_not_crash(self):
        os.makedirs(self.store, exist_ok=True)
        with open(os.path.join(self.store, "garbage.json"), "w",
                  encoding="utf-8") as fh:
            fh.write("{not json")
        with open(os.path.join(self.store, "newer.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"schema_version": 99, "snapshot_id": "newer",
                       "scan": {}}, fh)
        with open(os.path.join(self.store, "noscans.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"schema_version": 1}, fh)
        entries = snapshots.list_snapshots(self.store)
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(not e["valid"] for e in entries))
        self.assertTrue(any("newer than this tool" in (e["error"] or "")
                            for e in entries))

    def test_resolve_missing_error(self):
        os.makedirs(self.store, exist_ok=True)
        with self.assertRaises(ToolError) as ctx:
            snapshots.resolve_snapshot(self.store, "nope")
        self.assertEqual(ctx.exception.code, 3)
        self.assertIn("available", str(ctx.exception))

    def test_ambiguous_prefix(self):
        id1, _ = snapshots.create_snapshot(self.store, self.scan)
        id2, _ = snapshots.create_snapshot(self.store, self.scan)
        common = os.path.commonprefix([id1, id2])
        if common:  # timestamp prefix is shared
            with self.assertRaises(ToolError) as ctx:
                snapshots.resolve_snapshot(self.store, common)
            self.assertIn("ambiguous", str(ctx.exception))


class SnapshotCompareIntegration(unittest.TestCase):
    def test_round_trip_compare_finds_nothing(self):
        with tmpdir() as tmp:
            store = os.path.join(tmp, "store")
            comfy = os.path.join(tmp, "comfy")
            make_fake_comfy(comfy)
            scan = scan_install(comfy, allow_git_cmd=False)
            sid, _ = snapshots.create_snapshot(store, scan, "a")
            snap = snapshots.resolve_snapshot(store, "a")
            again = snapshots.resolve_snapshot(store, sid)
            from comfy_detective import diagnostics
            comp = diagnostics.compare_snapshots(snap, again, "a", sid)
            self.assertEqual(comp["findings"], [])
            cn = comp["sections"]["custom_nodes"]
            self.assertEqual(cn["added"], [])
            self.assertEqual(cn["removed"], [])
            self.assertEqual(cn["changed"], [])


if __name__ == "__main__":
    unittest.main()
