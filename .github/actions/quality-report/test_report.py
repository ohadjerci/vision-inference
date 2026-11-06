"""Exercise coverage gates, diff mapping, badge-only output, and untrusted input validation."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "quality_report", Path(__file__).with_name("report.py")
)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


class ReportTests(unittest.TestCase):
    def example(self):
        return {
            "schema": 1,
            "repository": "owner/repo",
            "run_id": 12,
            "head_sha": "a" * 40,
            "coverage": {
                "lines_covered": 8,
                "lines_total": 10,
                "branches_covered": 2,
                "branches_total": 4,
            },
            "coverage_ok": True,
            "diff": {"lines_covered": 1, "lines_total": 2},
            "mypy": {"status": "success", "errors": 0, "files": 5},
            "ruff": {"status": "success", "errors": 0, "format_errors": 0},
        }

    def test_exactly_four_badges_and_no_other_text(self):
        text = report.badges(self.example(), "https://github.com/owner/repo/actions/runs/12")
        self.assertEqual(text.count("[!["), 4)
        self.assertNotIn("\n", text)
        self.assertTrue(text.startswith("[![Coverage]("))
        self.assertIn("[![PR coverage](", text)
        self.assertIn("[![Mypy](", text)
        self.assertIn("[![Ruff](", text)
        self.assertIn("branch%2050.00%25", text)
        self.assertIn("line%2080.00%25", text)

    def test_unavailable_tools_and_no_changed_code_still_render_four_badges(self):
        data = self.example()
        data.update(coverage=None, coverage_ok=False, diff=None)
        data["mypy"] = {"status": "skipped", "errors": None, "files": 0}
        data["ruff"] = {"status": "failure", "errors": None, "format_errors": None}
        text = report.badges(data, "https://github.com/run")
        self.assertEqual(text.count("[!["), 4)
        self.assertIn("PR%20coverage-N%2FA", text)
        self.assertNotIn("None", text)

    def test_diff_hunks_include_added_lines_and_ignore_deletions(self):
        diff = "+++ b/src/a.py\n@@ -1,0 +2,3 @@\n+x\n+y\n+z\n@@ -10,1 +12,0 @@\n-x\n"
        self.assertEqual(report.changed_lines(diff), {"src/a.py": {2, 3, 4}})

    def test_nested_project_diff_counts_executable_lines_only(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            project = workspace / "backend"
            project.mkdir()
            data = {"src/inference_backend/a.py": {"executed_lines": [2], "missing_lines": [3]}}
            fake = type(
                "Completed",
                (),
                {
                    "stdout": (
                        "+++ b/backend/src/inference_backend/a.py\n@@ -1,0 +2,3 @@\n+x\n+y\n+# comment\n"
                    )
                },
            )()
            with (
                patch.dict(os.environ, {"PR_BASE_SHA": "b" * 40}),
                patch.object(report.subprocess, "run", return_value=fake),
            ):
                result = report.diff_stats(workspace, project, data)
            self.assertEqual(result, {"lines_covered": 1, "lines_total": 2})

    def test_typed_numeric_artifact_validation(self):
        report.validate_report(self.example(), "owner/repo", 12)
        for value in ("80", True, -1, 100_000_001):
            data = self.example()
            data["coverage"]["lines_covered"] = value
            with self.assertRaises(ValueError):
                report.validate_report(data, "owner/repo", 12)
        data = self.example()
        data["repository"] = "other/repo"
        with self.assertRaises(ValueError):
            report.validate_report(data, "owner/repo", 12)

    def test_independent_branch_gate_and_xml_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".github").mkdir()
            (root / ".github/coverage-baseline.json").write_text(
                json.dumps({"line_percent": 70, "branch_percent": 75})
            )
            (root / "coverage.json").write_text(
                json.dumps(
                    {
                        "meta": {"branch_coverage": True},
                        "files": {},
                        "totals": {
                            "covered_lines": 8,
                            "num_statements": 10,
                            "covered_branches": 2,
                            "num_branches": 4,
                        },
                    }
                )
            )
            (root / "coverage.xml").write_text(
                '<coverage lines-covered="8" lines-valid="10" branches-covered="2" '
                'branches-valid="4" />'
            )
            environment = {
                "GITHUB_WORKSPACE": str(root),
                "PROJECT_DIRECTORY": ".",
                "RUNNER_TEMP": str(root),
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_RUN_ID": "12",
                "GITHUB_SHA": "a" * 40,
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_STEP_SUMMARY": str(root / "summary"),
                "PR_BASE_SHA": "",
                "PR_HEAD_SHA": "",
            }
            with patch.dict(os.environ, environment):
                self.assertEqual(report.collect(), 1)
            output = json.loads((root / "quality-report.json").read_text())
            self.assertFalse(output["coverage_ok"])
            self.assertEqual((root / "summary").read_text().count("[!["), 4)
            (root / "coverage.xml").write_text('<coverage lines-covered="7" />')
            with self.assertRaises(ValueError):
                report.coverage_stats(root)

    def simulate_publish(
        self,
        *,
        current_head="a",
        run_head="a",
        attached=False,
        existing=True,
        report_head="a",
        redirect=False,
    ):
        payload = self.example()
        payload["head_sha"] = report_head * 40
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("quality-report.json", json.dumps(payload))
            output.writestr("ignored.py", "raise RuntimeError('must never execute')")
        event = {
            "workflow_run": {
                "id": 12,
                "event": "pull_request",
                "head_sha": run_head * 40,
                "repository": {"full_name": "owner/repo"},
            }
        }
        if attached:
            event["workflow_run"]["pull_requests"] = [{"number": 7, "head": {"sha": "a" * 40}}]
        calls = []

        def fake_api(path, method="GET", data=None):
            calls.append((path, method, data))
            if path.endswith("/artifacts?per_page=100"):
                return {"artifacts": [{"name": "quality-report", "id": 34}]}
            if "/commits/" in path:
                if "/pulls?" in path:
                    return [{"number": 7}]
                return {"parents": [{"sha": "a" * 40}]}
            if path == "/repos/owner/repo/pulls/7":
                return {
                    "state": "open",
                    "head": {"sha": current_head * 40},
                    "base": {"repo": {"full_name": "owner/repo"}},
                }
            if "/comments?" in path:
                return (
                    [
                        {
                            "id": 56,
                            "user": {"login": "github-actions[bot]"},
                            "body": "[![Coverage](old)" + report.MARKER,
                        }
                    ]
                    if existing
                    else []
                )
            return {}

        def download(request, timeout):
            if redirect:
                self.assertEqual(request.get_header("Authorization"), "Bearer test-token")
                redirected = report.urllib.request.HTTPRedirectHandler().redirect_request(
                    request,
                    None,
                    302,
                    "Found",
                    {},
                    "https://artifact-storage.example/report.zip?signature=temporary",
                )
                self.assertFalse(redirected.has_header("Authorization"))
            return io.BytesIO(archive.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "event.json").write_text(json.dumps(event))
            env = {
                "GITHUB_EVENT_PATH": str(root / "event.json"),
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_TOKEN": "test-token",
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_STEP_SUMMARY": str(root / "summary"),
            }
            with (
                patch.dict(os.environ, env),
                patch.object(report, "api", side_effect=fake_api),
                patch.object(report.urllib.request, "urlopen", side_effect=download),
            ):
                self.assertEqual(report.publish(), 0)
        return [item for item in calls if item[1] != "GET"]

    def test_artifact_redirect_does_not_forward_github_token(self):
        self.assertEqual(len(self.simulate_publish(redirect=True)), 1)

    def test_publish_updates_only_bot_comment_for_current_head(self):
        writes = self.simulate_publish()
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0:2], ("/repos/owner/repo/issues/comments/56", "PATCH"))
        self.assertEqual(writes[0][2]["body"].count("[!["), 4)
        self.assertNotIn("ignored.py", writes[0][2]["body"])

    def test_publish_creates_one_comment_when_none_exists(self):
        writes = self.simulate_publish(existing=False)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][0:2], ("/repos/owner/repo/issues/7/comments", "POST"))

    def test_stale_run_does_not_overwrite_newer_pr(self):
        self.assertEqual(self.simulate_publish(current_head="c"), [])

    def test_merge_run_accepts_actual_head_parent(self):
        self.assertEqual(len(self.simulate_publish(run_head="b")), 1)

    def test_trusted_run_pr_association_is_supported(self):
        self.assertEqual(len(self.simulate_publish(attached=True)), 1)

    def test_artifact_sha_unrelated_to_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "triggering workflow run"):
            self.simulate_publish(report_head="c")

    def test_malformed_tool_reports_render_badges_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ruff.json").write_text("[")
            (root / "mypy.log").write_text("tool execution failed")
            with patch.dict(
                os.environ,
                {
                    "RUNNER_TEMP": str(root),
                    "MYPY_OUTCOME": "failure",
                    "RUFF_LINT_OUTCOME": "failure",
                    "RUFF_FORMAT_OUTCOME": "skipped",
                },
            ):
                mypy, ruff = report.tool_stats()
            data = self.example()
            data.update(mypy=mypy, ruff=ruff)
            self.assertEqual(report.badges(data, "https://github.com/run").count("[!["), 4)
            self.assertIsNone(ruff["errors"])
            self.assertIsNone(mypy["errors"])


if __name__ == "__main__":
    unittest.main()
