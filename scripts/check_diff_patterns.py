#!/usr/bin/env python3
"""Check patterns only on changed lines in git diff.

More efficient than checking entire files - only validates the actual
changes being made. Useful for pre-commit hooks and CI.

Usage:
    python scripts/check_diff_patterns.py              # Check staged changes
    python scripts/check_diff_patterns.py --unstaged   # Check unstaged changes
    python scripts/check_diff_patterns.py --all        # Check all changes
    python scripts/check_diff_patterns.py --commit HEAD~1  # Check specific commit
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiffHunk:
    """A single diff hunk with changed lines."""
    file_path: str
    start_line: int
    line_count: int
    added_lines: list[tuple[int, str]]  # (line_num, content)
    removed_lines: list[tuple[int, str]]


@dataclass
class DiffViolation:
    """A pattern violation in a diff."""
    rule: str
    file: str
    line: int
    message: str
    severity: str
    diff_line: str


def parse_git_diff(diff_output: str) -> list[DiffHunk]:
    """Parse git diff output into hunks."""
    hunks: list[DiffHunk] = []
    current_file: str | None = None
    current_hunk: DiffHunk | None = None
    new_line_num = 0

    for line in diff_output.split("\n"):
        # File header
        if line.startswith("+++ b/"):
            current_file = line[6:]
        elif line.startswith("@@ "):
            # Hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
            if match and current_file:
                start_line = int(match.group(1))
                line_count = int(match.group(2)) if match.group(2) else 1
                current_hunk = DiffHunk(
                    file_path=current_file,
                    start_line=start_line,
                    line_count=line_count,
                    added_lines=[],
                    removed_lines=[],
                )
                hunks.append(current_hunk)
                new_line_num = start_line
        elif current_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.added_lines.append((new_line_num, line[1:]))
                new_line_num += 1
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.removed_lines.append((new_line_num, line[1:]))
            elif not line.startswith("\\"):  # Not "\ No newline at end of file"
                new_line_num += 1

    return hunks


def get_staged_diff(root: Path) -> str:
    """Get diff of staged changes."""
    result = subprocess.run(
        ["git", "diff", "--cached", "-U0"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    return result.stdout


def get_unstaged_diff(root: Path) -> str:
    """Get diff of unstaged changes."""
    result = subprocess.run(
        ["git", "diff", "-U0"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    return result.stdout


def get_commit_diff(root: Path, commit: str) -> str:
    """Get diff for a specific commit."""
    result = subprocess.run(
        ["git", "show", commit, "--format=", "-U0"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    return result.stdout


def check_line_patterns(line: str, line_num: int, file_path: str) -> list[DiffViolation]:
    """Check a single line against patterns."""
    violations: list[DiffViolation] = []

    # Skip non-Python files
    if not file_path.endswith(".py"):
        return violations

    # Skip comments
    stripped = line.strip()
    if stripped.startswith("#"):
        return violations

    # Pattern checks
    patterns = [
        # Hardcoded model IDs
        (r"['\"]claude-\d+[^'\"]*['\"]", "CFG001", "Hardcoded model ID", "error"),
        (r"['\"]gpt-\d+[^'\"]*['\"]", "CFG001", "Hardcoded model ID", "error"),
        (r"['\"]gemini-[^'\"]*['\"]", "CFG001", "Hardcoded model ID", "error"),
        (r"['\"]o\d+-[^'\"]*['\"]", "CFG001", "Hardcoded model ID", "error"),

        # Hardcoded timeouts (only large values)
        (r"timeout\s*=\s*\d{3,}", "CFG002", "Hardcoded timeout value", "warning"),

        # Bare except
        (r"except\s*:", "ERR001", "Bare except clause - specify exception type", "warning"),

        # Password/secret in code
        (r"password\s*=\s*['\"][^'\"]+['\"]", "SEC001", "Possible hardcoded password", "error"),
        (r"api_key\s*=\s*['\"][A-Za-z0-9_-]{20,}['\"]", "SEC002", "Possible hardcoded API key", "error"),

        # Blocking calls in async hint
        (r"time\.sleep\s*\(", "ASYNC001", "Blocking sleep - use asyncio.sleep in async code", "warning"),
        (r"requests\.(get|post|put|delete)\s*\(", "ASYNC002", "Blocking HTTP - use aiohttp in async code", "warning"),

        # Mutable default
        (r":\s*list\[[^\]]+\]\s*=\s*\[\]", "PYD001", "Mutable default - use Field(default_factory=list)", "warning"),
        (r":\s*dict\[[^\]]+\]\s*=\s*\{\}", "PYD001", "Mutable default - use Field(default_factory=dict)", "warning"),

        # Eval/exec
        (r"\beval\s*\(", "SEC003", "Use of eval() - security risk", "error"),
        (r"\bexec\s*\(", "SEC003", "Use of exec() - security risk", "error"),

        # Print statements (should use logging)
        (r"^\s*print\s*\(", "LOG002", "Use logging instead of print()", "warning"),

        # TODO/FIXME/HACK comments
        (r"#\s*(TODO|FIXME|HACK|XXX):", "DOC001", "Unresolved TODO/FIXME comment", "warning"),
    ]

    for pattern, rule, message, severity in patterns:
        if re.search(pattern, line, re.IGNORECASE):
            # Skip settings.py for config rules
            if rule.startswith("CFG") and "settings.py" in file_path:
                continue
            # Skip test files and CLI entry points for print rules
            if rule == "LOG002":
                if "/tests/" in file_path or "\\tests\\" in file_path:
                    continue
                if "__main__.py" in file_path or "scripts/" in file_path:
                    continue

            violations.append(DiffViolation(
                rule=rule,
                file=file_path,
                line=line_num,
                message=message,
                severity=severity,
                diff_line=line.strip()[:60],
            ))

    return violations


def check_hunks(hunks: list[DiffHunk]) -> list[DiffViolation]:
    """Check all hunks for pattern violations."""
    all_violations: list[DiffViolation] = []

    for hunk in hunks:
        # Only check added lines (new code)
        for line_num, line_content in hunk.added_lines:
            violations = check_line_patterns(line_content, line_num, hunk.file_path)
            all_violations.extend(violations)

    return all_violations


def print_violations(violations: list[DiffViolation], show_context: bool = True) -> None:
    """Print violations in a readable format."""
    if not violations:
        print("\n  [OK] No pattern violations in diff.\n")
        return

    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    print("\n" + "=" * 70)
    print("  DIFF PATTERN CHECK RESULTS")
    print("=" * 70)

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for v in errors:
            print(f"    [{v.rule}] {v.file}:{v.line}")
            print(f"           {v.message}")
            if show_context:
                print(f"           > {v.diff_line}")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for v in warnings:
            print(f"    [{v.rule}] {v.file}:{v.line}")
            print(f"           {v.message}")
            if show_context:
                print(f"           > {v.diff_line}")

    print("\n" + "=" * 70)
    print(f"  Total: {len(errors)} errors, {len(warnings)} warnings")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check patterns on git diff")
    parser.add_argument("--unstaged", action="store_true", help="Check unstaged changes")
    parser.add_argument("--all", action="store_true", help="Check all changes (staged + unstaged)")
    parser.add_argument("--commit", help="Check specific commit")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Exit with error on any violations")
    parser.add_argument("--no-context", action="store_true", help="Don't show diff context")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    # Get appropriate diff
    if args.commit:
        diff_output = get_commit_diff(root, args.commit)
        source = f"commit {args.commit}"
    elif args.all:
        staged = get_staged_diff(root)
        unstaged = get_unstaged_diff(root)
        diff_output = staged + "\n" + unstaged
        source = "all changes"
    elif args.unstaged:
        diff_output = get_unstaged_diff(root)
        source = "unstaged changes"
    else:
        diff_output = get_staged_diff(root)
        source = "staged changes"

    if not diff_output.strip():
        print(f"No {source} to check.")
        sys.exit(0)

    # Parse and check
    hunks = parse_git_diff(diff_output)
    print(f"Checking {len(hunks)} diff hunks from {source}...")

    violations = check_hunks(hunks)

    if args.json:
        output = [
            {
                "rule": v.rule,
                "file": v.file,
                "line": v.line,
                "message": v.message,
                "severity": v.severity,
                "diff_line": v.diff_line,
            }
            for v in violations
        ]
        print(json.dumps(output, indent=2))
    else:
        print_violations(violations, show_context=not args.no_context)

    # Exit code
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        sys.exit(1)
    elif args.strict and violations:
        sys.exit(1)


if __name__ == "__main__":
    main()
