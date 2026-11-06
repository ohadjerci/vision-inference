"""Produce four CI badges and safely publish numeric reports from read-only PR runs."""

import io
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = 1
MARKER = "#ci-quality-report"
STATES = {"success", "failure", "skipped", "cancelled"}


def percentage(covered: int, total: int) -> float | None:
    return round(100 * covered / total, 4) if total else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def coverage_stats(project: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    report = read_json(project / "coverage.json")
    if not report.get("meta", {}).get("branch_coverage"):
        raise ValueError("Branch coverage was not enabled")
    totals = report["totals"]
    stats = {
        "lines_covered": totals["covered_lines"],
        "lines_total": totals["num_statements"],
        "branches_covered": totals["covered_branches"],
        "branches_total": totals["num_branches"],
    }
    xml = ET.parse(project / "coverage.xml").getroot()
    for xml_key, key in (
        ("lines-covered", "lines_covered"),
        ("lines-valid", "lines_total"),
        ("branches-covered", "branches_covered"),
        ("branches-valid", "branches_total"),
    ):
        if int(xml.attrib[xml_key]) != stats[key]:
            raise ValueError("Coverage XML and JSON totals disagree")
    if not stats["lines_total"] or not stats["branches_total"]:
        raise ValueError("Expected measured source lines and branches")
    return stats, report["files"]


def changed_lines(diff: str) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    filename = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            filename = line[6:]
            result.setdefault(filename, set())
        match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if filename and match:
            first = int(match[1])
            count = int(match[2]) if match[2] is not None else 1
            result[filename].update(range(first, first + count))
    return result


def diff_stats(workspace: Path, project: Path, files: dict[str, Any]) -> dict[str, int] | None:
    base = os.environ.get("PR_BASE_SHA", "")
    if not base:
        return None
    if not re.fullmatch(r"[a-fA-F0-9]{40}", base):
        raise ValueError("Invalid PR base SHA")
    diff = subprocess.run(
        [
            "git",
            "-c",
            "core.quotePath=false",
            "diff",
            "--unified=0",
            "--diff-filter=ACMRT",
            f"{base}...HEAD",
            "--",
            ".",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    added = changed_lines(diff)
    prefix = project.relative_to(workspace).as_posix()
    covered = total = 0
    for filename, data in files.items():
        path = Path(filename)
        if path.is_absolute():
            try:
                path = path.relative_to(project)
            except ValueError:
                continue
        key = path.as_posix() if prefix == "." else f"{prefix}/{path.as_posix()}"
        executed = set(data["executed_lines"])
        executable = executed | set(data["missing_lines"])
        changed = executable & added.get(key, set())
        covered += len(changed & executed)
        total += len(changed)
    return {"lines_covered": covered, "lines_total": total}


def badge(label: str, message: str, color: str, url: str) -> str:
    def encode(value: str) -> str:
        return urllib.parse.quote(value.replace("-", "--").replace("_", "__"), safe="")

    image = f"https://img.shields.io/badge/{encode(label)}-{encode(message)}-{color}"
    return f"[![{label}]({image})]({url}{MARKER})"


def badges(report: dict[str, Any], url: str) -> str:
    stats = report.get("coverage")
    coverage_text = "unavailable"
    if stats:
        branches = percentage(stats["branches_covered"], stats["branches_total"])
        lines = percentage(stats["lines_covered"], stats["lines_total"])
        coverage_text = f"branch {branches:.2f}% / line {lines:.2f}%"
    diff = report.get("diff")
    diff_text = "N/A"
    if diff and diff["lines_total"]:
        value = percentage(diff["lines_covered"], diff["lines_total"])
        diff_text = f"{value:.2f}% ({diff['lines_covered']}/{diff['lines_total']})"
    mypy = report["mypy"]
    mypy_text = mypy["status"]
    if mypy["errors"] is not None and mypy["files"] is not None:
        mypy_text = f"{mypy['errors']} errors / {mypy['files']} files"
    ruff = report["ruff"]
    ruff_text = ruff["status"]
    if ruff["errors"] is not None and ruff["format_errors"] is not None:
        ruff_text = f"{ruff['errors']} lint / {ruff['format_errors']} format"
    return " ".join(
        [
            badge(
                "Coverage", coverage_text, "brightgreen" if report["coverage_ok"] else "red", url
            ),
            badge(
                "PR coverage",
                diff_text,
                "blue" if diff and diff["lines_total"] else "lightgrey",
                url,
            ),
            badge("Mypy", mypy_text, "brightgreen" if mypy["status"] == "success" else "red", url),
            badge("Ruff", ruff_text, "brightgreen" if ruff["status"] == "success" else "red", url),
        ]
    )


def tool_stats() -> tuple[dict[str, Any], dict[str, Any]]:
    temp = Path(os.environ["RUNNER_TEMP"])

    def outcome(name: str) -> str:
        value = os.environ.get(name, "skipped") or "skipped"
        return value if value in STATES else "failure"

    def text(name: str) -> str:
        try:
            return (temp / name).read_text()
        except (OSError, ValueError):
            return ""

    mypy_status = outcome("MYPY_OUTCOME")
    lint_status = outcome("RUFF_LINT_OUTCOME")
    format_status = outcome("RUFF_FORMAT_OUTCOME")
    mypy_log = text("mypy.log")
    mypy_errors = len(re.findall(r"^.+?:\d+(?::\d+)?: error:", mypy_log, re.MULTILINE))
    source_files = re.search(r"(?:in|checked) (\d+) source files?", mypy_log)
    mypy = {
        "status": mypy_status,
        "errors": mypy_errors if source_files else None,
        "files": int(source_files[1]) if source_files else None,
    }
    if mypy_status == "success" and source_files is None:
        mypy["status"] = "failure"
    diagnostics = None
    try:
        parsed = read_json(temp / "ruff.json")
        if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
            diagnostics = parsed
    except (OSError, ValueError):
        pass
    format_text = text("ruff-format.log")
    unformatted = re.search(r"(\d+) files? would be reformatted", format_text)
    status = "failure"
    if lint_status == format_status == "success" and diagnostics is not None and format_text:
        status = "success"
    elif lint_status == format_status and lint_status in {"skipped", "cancelled"}:
        status = lint_status
    ruff = {
        "status": status,
        "errors": len(diagnostics) if diagnostics is not None else None,
        "format_errors": int(unformatted[1])
        if unformatted
        else (0 if format_text and format_status == "success" else None),
    }
    return mypy, ruff


def collect() -> int:
    workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    project = (workspace / os.environ.get("PROJECT_DIRECTORY", ".")).resolve()
    mypy, ruff = tool_stats()
    report = {
        "schema": SCHEMA,
        "repository": os.environ["GITHUB_REPOSITORY"],
        "run_id": int(os.environ["GITHUB_RUN_ID"]),
        "head_sha": os.environ.get("PR_HEAD_SHA") or os.environ["GITHUB_SHA"],
        "coverage": None,
        "coverage_ok": False,
        "diff": None,
        "mypy": mypy,
        "ruff": ruff,
    }
    failed = False
    try:
        baseline = read_json(workspace / ".github/coverage-baseline.json")
        stats, files = coverage_stats(project)
        report["coverage"] = stats
        line_percent = percentage(stats["lines_covered"], stats["lines_total"])
        branch_percent = percentage(stats["branches_covered"], stats["branches_total"])
        report["coverage_ok"] = (
            line_percent >= baseline["line_percent"]
            and branch_percent >= baseline["branch_percent"]
        )
        report["diff"] = diff_stats(workspace, project, files)
        failed = not report["coverage_ok"]
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError, ET.ParseError) as error:
        print(f"Coverage verification failed: {error}", file=sys.stderr)
        failed = True
    destination = Path(os.environ["RUNNER_TEMP"]) / "quality-report.json"
    destination.write_text(json.dumps(report, allow_nan=False))
    url = (
        f"{os.environ['GITHUB_SERVER_URL']}/{report['repository']}/actions/runs/{report['run_id']}"
    )
    Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(badges(report, url) + "\n")
    if failed:
        print("Coverage report missing, invalid, or below the committed baseline", file=sys.stderr)
    return int(failed)


def api(path: str, method: str = "GET", data: dict[str, Any] | None = None) -> Any:
    url = os.environ.get("GITHUB_API_URL", "https://api.github.com") + path
    request = urllib.request.Request(url, method=method)
    request.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(data).encode()
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def integer(value: Any, maximum: int = 100_000_000) -> bool:
    return type(value) is int and 0 <= value <= maximum


def validate_report(report: Any, repository: str, run_id: int) -> None:
    if not isinstance(report, dict) or report.get("schema") != SCHEMA:
        raise ValueError("Invalid report schema")
    if report.get("repository") != repository or report.get("run_id") != run_id:
        raise ValueError("Report does not belong to this repository/run")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", str(report.get("head_sha", ""))):
        raise ValueError("Invalid report SHA")
    if type(report.get("coverage_ok")) is not bool:
        raise ValueError("Invalid coverage outcome")
    for section in ("coverage", "diff"):
        stats = report.get(section)
        if stats is None:
            continue
        keys = ["lines_covered", "lines_total"]
        if section == "coverage":
            keys += ["branches_covered", "branches_total"]
        if not isinstance(stats, dict) or any(not integer(stats.get(key)) for key in keys):
            raise ValueError("Invalid coverage counts")
        if stats["lines_covered"] > stats["lines_total"]:
            raise ValueError("Impossible coverage counts")
        if section == "coverage" and (
            not stats["lines_total"]
            or not stats["branches_total"]
            or stats["branches_covered"] > stats["branches_total"]
        ):
            raise ValueError("Impossible branch coverage counts")
    for tool in ("mypy", "ruff"):
        stats = report.get(tool)
        if not isinstance(stats, dict) or stats.get("status") not in STATES:
            raise ValueError("Invalid tool outcome")
        for key in ("errors", "files") if tool == "mypy" else ("errors", "format_errors"):
            if stats.get(key) is not None and not integer(stats.get(key)):
                raise ValueError("Invalid tool counts")


def publish() -> int:
    event = read_json(Path(os.environ["GITHUB_EVENT_PATH"]))
    run = event["workflow_run"]
    repository = os.environ["GITHUB_REPOSITORY"]
    run_id = int(run["id"])
    if run["event"] != "pull_request" or run["repository"]["full_name"] != repository:
        raise ValueError("Not a pull request run in this repository")
    artifacts = api(f"/repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100")
    artifact = next((x for x in artifacts["artifacts"] if x["name"] == "quality-report"), None)
    if artifact is None:
        print("No numeric report artifact; no comment published")
        return 0
    request = urllib.request.Request(
        os.environ.get("GITHUB_API_URL", "https://api.github.com")
        + f"/repos/{repository}/actions/artifacts/{int(artifact['id'])}/zip",
    )
    # Authenticate GitHub only; its redirect targets signed artifact storage URLs.
    request.add_unredirected_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read(1_000_001)
    if len(archive_bytes) > 1_000_000:
        raise ValueError("Oversized numeric report archive")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        item = archive.getinfo("quality-report.json")
        if item.file_size > 64_000:
            raise ValueError("Oversized numeric report")
        report = json.loads(archive.read(item))
    validate_report(report, repository, run_id)
    # Bind the artifact to the triggering commit, allowing the tested PR merge commit.
    pulls = run.get("pull_requests", [])
    allowed_heads = {run["head_sha"]}
    allowed_heads.update(item.get("head", {}).get("sha") for item in pulls)
    if report["head_sha"] not in allowed_heads:
        commit = api(f"/repos/{repository}/commits/{run['head_sha']}")
        if report["head_sha"] not in {item["sha"] for item in commit.get("parents", [])}:
            raise ValueError("Report SHA does not belong to the triggering workflow run")
    # Determine candidate PRs from trusted API association, never an artifact PR number.
    if not pulls:
        pulls = api(f"/repos/{repository}/commits/{run['head_sha']}/pulls?per_page=100")
    for candidate in pulls:
        number = int(candidate["number"])
        pr = api(f"/repos/{repository}/pulls/{number}")
        if pr["state"] != "open" or pr["head"]["sha"] != report["head_sha"]:
            continue
        if pr["base"]["repo"]["full_name"] != repository:
            continue
        url = f"{os.environ['GITHUB_SERVER_URL']}/{repository}/actions/runs/{run_id}"
        body = badges(report, url)
        comments = []
        page = 1
        while True:
            batch = api(f"/repos/{repository}/issues/{number}/comments?per_page=100&page={page}")
            comments.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        existing = next(
            (
                x
                for x in comments
                if x["user"]["login"] == "github-actions[bot]"
                and MARKER in x["body"]
                and x["body"].startswith("[![Coverage](")
            ),
            None,
        )
        if existing:
            api(
                f"/repos/{repository}/issues/comments/{int(existing['id'])}",
                "PATCH",
                {"body": body},
            )
        else:
            api(f"/repos/{repository}/issues/{number}/comments", "POST", {"body": body})
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(body + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(publish() if os.environ.get("REPORT_MODE") == "publish" else collect())
