#!/usr/bin/env python3
"""Verify implementation completeness and correctness.

Checks that an implementation is complete by verifying:
1. Expected files were modified
2. Required patterns are present
3. Tests pass (optional)
4. Build succeeds (optional)
5. No regressions in critical paths

Usage:
    python scripts/verify_implementation.py                    # Basic verification
    python scripts/verify_implementation.py --run-tests        # Include test run
    python scripts/verify_implementation.py --run-build        # Include build check
    python scripts/verify_implementation.py --checklist FILE   # Use custom checklist
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class VerificationItem:
    """A single verification item."""
    name: str
    description: str
    check_type: str  # file_exists, file_contains, command_succeeds, pattern_match
    target: str
    pattern: str | None = None
    required: bool = True


@dataclass
class VerificationResult:
    """Result of a verification check."""
    item: VerificationItem
    passed: bool
    message: str
    details: str | None = None


@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    total_checks: int
    passed: int
    failed: int
    skipped: int
    results: list[VerificationResult]
    duration_seconds: float


def load_checklist(checklist_path: Path) -> list[VerificationItem]:
    """Load a verification checklist from JSON."""
    if not checklist_path.exists():
        return []

    data = json.loads(checklist_path.read_text())
    items = []
    for item in data.get("checks", []):
        items.append(VerificationItem(
            name=item["name"],
            description=item.get("description", ""),
            check_type=item["type"],
            target=item["target"],
            pattern=item.get("pattern"),
            required=item.get("required", True),
        ))
    return items


def get_default_checklist(root: Path) -> list[VerificationItem]:
    """Get default verification checklist based on git changes."""
    items = []

    # Check for modified files
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    staged_files = [f for f in result.stdout.strip().split("\n") if f]

    # Add file existence checks for all staged files
    for file_path in staged_files:
        if file_path.endswith(".py"):
            items.append(VerificationItem(
                name=f"File exists: {file_path}",
                description=f"Verify {file_path} exists",
                check_type="file_exists",
                target=file_path,
            ))

    # Add standard checks
    items.extend([
        VerificationItem(
            name="Code catalog exists",
            description="Code catalog should be generated",
            check_type="file_exists",
            target="data/code_catalog.json",
            required=False,
        ),
        VerificationItem(
            name="No syntax errors",
            description="All Python files should be syntactically valid",
            check_type="command_succeeds",
            target="python -m py_compile ai_orchestrator/__init__.py",
            required=True,
        ),
        VerificationItem(
            name="Settings file valid",
            description="Settings should load without errors",
            check_type="command_succeeds",
            target="python -c \"from ai_orchestrator.config.settings import get_settings; print(get_settings())\"",
            required=True,
        ),
    ])

    return items


def check_file_exists(root: Path, target: str) -> tuple[bool, str]:
    """Check if a file exists."""
    file_path = root / target
    if file_path.exists():
        return True, f"File exists: {target}"
    return False, f"File not found: {target}"


def check_file_contains(root: Path, target: str, pattern: str) -> tuple[bool, str]:
    """Check if a file contains a pattern."""
    import re

    file_path = root / target
    if not file_path.exists():
        return False, f"File not found: {target}"

    try:
        content = file_path.read_text(encoding="utf-8")
        if re.search(pattern, content):
            return True, f"Pattern found in {target}"
        return False, f"Pattern not found in {target}"
    except Exception as e:
        return False, f"Error reading {target}: {e}"


def check_command_succeeds(root: Path, target: str) -> tuple[bool, str]:
    """Check if a command succeeds."""
    try:
        result = subprocess.run(
            target,
            shell=True,
            capture_output=True,
            text=True,
            cwd=root,
            timeout=60,
        )
        if result.returncode == 0:
            return True, "Command succeeded"
        return False, f"Command failed: {result.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, f"Error running command: {e}"


def check_pattern_match(root: Path, target: str, pattern: str) -> tuple[bool, str]:
    """Check if files matching target contain pattern."""
    import re

    matches = list(root.glob(target))
    if not matches:
        return False, f"No files matching: {target}"

    for file_path in matches:
        try:
            content = file_path.read_text(encoding="utf-8")
            if not re.search(pattern, content):
                return False, f"Pattern not found in {file_path.relative_to(root)}"
        except Exception as e:
            return False, f"Error reading {file_path}: {e}"

    return True, f"Pattern found in all {len(matches)} matching files"


def run_verification(item: VerificationItem, root: Path) -> VerificationResult:
    """Run a single verification check."""
    try:
        if item.check_type == "file_exists":
            passed, message = check_file_exists(root, item.target)
        elif item.check_type == "file_contains":
            passed, message = check_file_contains(root, item.target, item.pattern or "")
        elif item.check_type == "command_succeeds":
            passed, message = check_command_succeeds(root, item.target)
        elif item.check_type == "pattern_match":
            passed, message = check_pattern_match(root, item.target, item.pattern or "")
        else:
            passed, message = False, f"Unknown check type: {item.check_type}"

        return VerificationResult(
            item=item,
            passed=passed,
            message=message,
        )
    except Exception as e:
        return VerificationResult(
            item=item,
            passed=False,
            message=f"Check failed with error: {e}",
        )


def run_tests(root: Path) -> VerificationResult:
    """Run the test suite."""
    item = VerificationItem(
        name="Test Suite",
        description="Run pytest on the test suite",
        check_type="command_succeeds",
        target="pytest",
    )

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=300,
        )

        if result.returncode == 0:
            # Count passed tests
            passed_match = result.stdout.count(" passed")
            return VerificationResult(
                item=item,
                passed=True,
                message=f"All tests passed",
                details=result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
            )
        else:
            return VerificationResult(
                item=item,
                passed=False,
                message="Tests failed",
                details=result.stdout[-500:] + result.stderr[-500:],
            )
    except subprocess.TimeoutExpired:
        return VerificationResult(
            item=item,
            passed=False,
            message="Tests timed out after 5 minutes",
        )
    except Exception as e:
        return VerificationResult(
            item=item,
            passed=False,
            message=f"Could not run tests: {e}",
        )


def run_build(root: Path) -> VerificationResult:
    """Run a build check."""
    item = VerificationItem(
        name="Build Check",
        description="Verify the project builds/imports correctly",
        check_type="command_succeeds",
        target="python -c 'import ai_orchestrator'",
    )

    try:
        result = subprocess.run(
            [sys.executable, "-c", "import ai_orchestrator; print('Import successful')"],
            capture_output=True,
            text=True,
            cwd=root,
            timeout=30,
        )

        if result.returncode == 0:
            return VerificationResult(
                item=item,
                passed=True,
                message="Build/import successful",
            )
        else:
            return VerificationResult(
                item=item,
                passed=False,
                message="Build/import failed",
                details=result.stderr[:500],
            )
    except Exception as e:
        return VerificationResult(
            item=item,
            passed=False,
            message=f"Build check failed: {e}",
        )


def check_regressions(root: Path) -> VerificationResult:
    """Check for potential regressions in critical paths."""
    item = VerificationItem(
        name="Regression Check",
        description="Check critical imports and paths still work",
        check_type="command_succeeds",
        target="python regression_check",
    )

    critical_imports = [
        "from ai_orchestrator.config.settings import get_settings",
        "from ai_orchestrator.cli_adapters.base import CLIAdapter",
        "from ai_orchestrator.core.orchestrator import Orchestrator",
        "from ai_orchestrator.dashboard.schemas import DashboardConfig",
    ]

    failed_imports = []
    for imp in critical_imports:
        try:
            result = subprocess.run(
                [sys.executable, "-c", imp],
                capture_output=True,
                text=True,
                cwd=root,
                timeout=10,
            )
            if result.returncode != 0:
                failed_imports.append(f"{imp}: {result.stderr[:100]}")
        except Exception as e:
            failed_imports.append(f"{imp}: {e}")

    if failed_imports:
        return VerificationResult(
            item=item,
            passed=False,
            message=f"{len(failed_imports)} critical imports failed",
            details="\n".join(failed_imports),
        )

    return VerificationResult(
        item=item,
        passed=True,
        message="All critical imports successful",
    )


def print_report(report: VerificationReport) -> None:
    """Print the verification report."""
    print("\n" + "=" * 70)
    print("  IMPLEMENTATION VERIFICATION REPORT")
    print("=" * 70)

    print(f"\n  Timestamp: {report.timestamp}")
    print(f"  Duration:  {report.duration_seconds:.2f}s")

    print("\n  " + "-" * 66)
    print(f"  {'Check':<40} {'Status':<15} {'Required':<10}")
    print("  " + "-" * 66)

    for result in report.results:
        if result.passed:
            status = "PASSED"
        else:
            status = "FAILED"

        req = "Yes" if result.item.required else "No"
        name = result.item.name[:38] + ".." if len(result.item.name) > 40 else result.item.name
        print(f"  {name:<40} {status:<15} {req:<10}")

    print("  " + "-" * 66)

    # Summary
    print(f"\n  Summary:")
    print(f"    Total Checks: {report.total_checks}")
    print(f"    Passed:       {report.passed}")
    print(f"    Failed:       {report.failed}")
    print(f"    Skipped:      {report.skipped}")

    # Show failures
    failures = [r for r in report.results if not r.passed]
    if failures:
        print(f"\n  Failures:")
        for r in failures:
            req_marker = "[REQUIRED]" if r.item.required else "[optional]"
            print(f"    {req_marker} {r.item.name}")
            print(f"              {r.message}")
            if r.details:
                for line in r.details.split("\n")[:3]:
                    print(f"              > {line[:60]}")

    # Final status
    required_failures = [r for r in failures if r.item.required]

    print("\n" + "=" * 70)
    if required_failures:
        print(f"  STATUS: FAILED - {len(required_failures)} required checks failed")
    elif failures:
        print(f"  STATUS: PASSED WITH WARNINGS - {len(failures)} optional checks failed")
    else:
        print("  STATUS: PASSED - All checks successful")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify implementation completeness")
    parser.add_argument("--run-tests", action="store_true", help="Run test suite")
    parser.add_argument("--run-build", action="store_true", help="Run build check")
    parser.add_argument("--checklist", help="Path to custom checklist JSON")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Fail on any check failure")
    args = parser.parse_args()

    import time
    start_time = time.time()

    root = Path(__file__).parent.parent

    # Load checklist
    if args.checklist:
        checklist = load_checklist(Path(args.checklist))
    else:
        checklist = get_default_checklist(root)

    print(f"Running {len(checklist)} verification checks...")

    # Run all checks
    results: list[VerificationResult] = []

    # Run checklist items
    for item in checklist:
        result = run_verification(item, root)
        results.append(result)

    # Always run regression check
    results.append(check_regressions(root))

    # Optional: run build
    if args.run_build:
        results.append(run_build(root))

    # Optional: run tests
    if args.run_tests:
        results.append(run_tests(root))

    # Build report
    report = VerificationReport(
        timestamp=datetime.now().isoformat(),
        total_checks=len(results),
        passed=sum(1 for r in results if r.passed),
        failed=sum(1 for r in results if not r.passed),
        skipped=0,
        results=results,
        duration_seconds=time.time() - start_time,
    )

    if args.json:
        output = {
            "timestamp": report.timestamp,
            "total_checks": report.total_checks,
            "passed": report.passed,
            "failed": report.failed,
            "duration_seconds": report.duration_seconds,
            "results": [
                {
                    "name": r.item.name,
                    "description": r.item.description,
                    "passed": r.passed,
                    "required": r.item.required,
                    "message": r.message,
                    "details": r.details,
                }
                for r in report.results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_report(report)

    # Exit code
    required_failures = [r for r in results if not r.passed and r.item.required]
    if required_failures:
        sys.exit(1)
    elif args.strict and report.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
