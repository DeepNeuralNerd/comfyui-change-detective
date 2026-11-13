import os
import shutil
import tempfile
import unittest

from comfy_detective import util
from fixtures import GIT, TempDir, git_init_commit, git_run, tmpdir, w


class UtilTests(unittest.TestCase):
    def test_compare_versions(self):
        self.assertEqual(util.compare_versions("1.0.0", "1.0.1"), -1)
        self.assertEqual(util.compare_versions("1.10", "1.9"), 1)
        self.assertEqual(util.compare_versions("2.0", "2.0"), 0)
        self.assertIsNone(util.compare_versions("weird??", "1.0"))

    def test_canonical_name(self):
        self.assertEqual(util.canonical_pkg_name("Comfy_UI-Foo.bar"),
                         "comfy-ui-foo-bar")

    def test_atomic_write_no_tmp_leftovers(self):
        with tmpdir() as d:
            path = os.path.join(d, "out.json")
            util.atomic_write_json(path, {"a": 1})
            util.atomic_write_json(path, {"a": 2})
            self.assertEqual(util.read_text(path, 4096)[0].strip(),
                             '{\n  "a": 2\n}')
            leftovers = [n for n in os.listdir(d) if n.startswith(".tmp")]
            self.assertEqual(leftovers, [])

    def test_read_text_missing(self):
        text, err = util.read_text(os.path.join(tempfile.gettempdir(),
                                                "definitely-not-here-xyz"))
        self.assertIsNone(text)
        self.assertEqual(err, "file not found")

    @unittest.skipUnless(GIT, "git binary required")
    def test_git_info_file_based_and_dirty(self):
        with tmpdir() as d:
            w(os.path.join(d, "f.txt"), "1\n")
            sha = git_init_commit(d)
            info = util.git_info(d, allow_git_cmd=False)
            self.assertTrue(info["is_git"])
            self.assertEqual(info["commit"], sha)
            self.assertEqual(info["branch"], "master")
            self.assertIsNone(info["dirty"])  # cmd skipped -> unknown, not clean
            w(os.path.join(d, "dirty.txt"), "x\n")
            info = util.git_info(d, allow_git_cmd=True)
            self.assertTrue(info["dirty"])
            # detached HEAD
            git_run(d, "checkout", "-q", "--detach", sha)
            info = util.git_info(d, allow_git_cmd=False)
            self.assertTrue(info["detached"])
            self.assertEqual(info["commit"], sha)

    @unittest.skipUnless(GIT, "git binary required")
    def test_git_info_non_repo(self):
        with tmpdir() as d:
            info = util.git_info(d)
            self.assertFalse(info["is_git"])
            self.assertIsNone(info["dirty"])

    def test_store_default_is_absolute(self):
        self.assertTrue(os.path.isabs(util.default_store_dir()))


if __name__ == "__main__":
    unittest.main()
