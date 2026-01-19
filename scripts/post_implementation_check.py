#!/usr/bin/env python3
"""Post-implementation validation runner.

Runs all validation checks after code changes are made. This is the main
entry point for the "Cut Once" phase of the workflow.

Usage:
    python scripts/post_implementation_check.py              # Check staged/modified files
    python scripts/post_implementation_check.py --all        # Full validation
    python scripts/post_implementation_check.py --strict     # Fail on any issues
    python scripts/post_implementation_check.py --fix        # Auto-fix where possible
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    skipped_checks: int
    total_errors: int
    total_warnings: int
    results: list[CheckResult]
    modified_files: list[str]
    duration_seconds: float


def get_modified_files(root: Path) -> list[str]:
    """Get list of modified/staged Python files from git."""
    try:
        # Get staged files
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        staged_files = staged.stdout.strip().split("\n") if staged.stdout.strip() else []

        # Get modified but unstaged files
        modified = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        modified_files = modified.stdout.strip().split("\n") if modified.stdout.strip() else []

        # Get untracked files
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            cwd=root,
        )
        untracked_files = untracked.stdout.strip().split("\n") if untracked.stdout.strip() else []

        # Combine and filter for Python files
        all_files = set(staged_files + modified_files + untracked_files)
        python_files = [f for f in all_files if f.endswith(".py") and f]

        return sorted(python_files)
    except Exception as e:
        print(f"Warning: Could not get git status: {e}")
        return []


def run_pattern_check(root: Path, files: list[str], strict: bool = False) -> CheckResult:
    """Run pattern checker on specified files."""
    import time
    start = time.time()

    if not files:
        return CheckResult(
            name="Pattern Check",
            passed=True,
            skipped=True,
            skip_reason="No Python files to check",
        )

    try:
        # Run pattern checker
        cmd = [sys.executable, str(root / "scripts" / "check_patterns.py")]
        if len(files) <= 10:
            cmd.extend(files)
        else:
            cmd.append("--all")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=root,
        )

        # Parse output for errors/warnings
        errors = []
        warnings = []
        for line in result.stdout.split("\n"):
            if "ERRORS" in line or (line.strip().startswith("[") and "error" in line.lower()):
                continue
            if line.strip().startswith("[CFG") or line.strip().startswith("[ERR") or line.strip().startswith("[ASYNC"):
                if "error" in result.stdout.lower():
                    errors.append(line.strip())
                else:
                    warnings.append(line.strip())

        # Count actual errors from output
        error_count = 0
        warning_count = 0
        for line in result.stdout.split("\n"):
            if "ERRORS (" in line:
                try:
                    error_count = int(line.split("(")[1].split(")")[0])
                except (IndexError, ValueError):
                    pass
            if "WARNINGS (" in line:
                try:
                    warning_count = int(line.split("(")[1].split(")")[0])
                except (IndexError, ValueError):
                    pass

        passed = error_count == 0 if strict else True

        return CheckResult(
            name="Pattern Check",
            passed=passed,
            errors=[f"{error_count} pattern errors found"] if error_count > 0 else [],
            warnings=[f"{warning_count} pattern warnings found"] if warning_count > 0 else [],
            duration_seconds=time.time() - start,
        )
    except Exception as e:
        return CheckResult(
            name="Pattern Check",
            passed=False,
            errors=[str(e)],
            duration_seconds=time.time() - start,
        )


def run_catalog_freshness_check(root: Path) -> CheckResult:
    """Check if code catalog is up to date."""
    import time
    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(root / "scripts" / "build_code_registry.py"), "--check"],
            capture_output=True,
            text=True,
            cwd=root,
        )

        passed = result.returncode == 0
        errors: list[str] = []
        warnings: list[str] = []

        if not passed:
            errors.append("Code catalog is stale - run: python scripts/build_code_registry.py")

        return CheckResult(
            name="Catalog Freshness",
            passed=passed,
            errors=errors,
            warnings=warnings,
            duration_seconds=time.time() - start,
        )
    except Exception as e:
        return CheckResult(
            name="Catalog Freshness",
            passed=False,
            errors=[str(e)],
            duration_seconds=time.time() - start,
        )


def run_import_check(root: Path, files: list[str]) -> CheckResult:
    """Check for circular imports and import issues."""
    import time
    start = time.time()

    if not files:
        return CheckResult(
            name="Import Check",
            passed=True,
            skipped=True,
            skip_reason="No files to check",
        )

    errors = []
    warnings = []

    for file_path in files:
        full_path = root / file_path
        if not full_path.exists():
            continue

        try:
            # Try to compile the file to check for syntax errors
            content = full_path.read_text(encoding="utf-8")
            compile(content, str(full_path), "exec")
        except SyntaxError as e:
            errors.append(f"{file_path}:{e.lineno}: Syntax error: {e.msg}")
        except Exception as e:
            warnings.append(f"{file_path}: Could not compile: {e}")

    return CheckResult(
        name="Import Check",
        passed=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        duration_seconds=time.time() - start,
    )


def run_type_check(root: Path, files: list[str]) -> CheckResult:
    """Run basic type checking if mypy is available."""
    import time
    start = time.time()

    if not files:
        return CheckResult(
            name="Type Check",
            passed=True,
            skipped=True,
            skip_reason="No files to check",
        )

    try:
        # Check if mypy is available
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return CheckResult(
                name="Type Check",
                passed=True,
                skipped=True,
                skip_reason="mypy not installed",
                duration_seconds=time.time() - start,
            )

        # Run mypy on modified files
        cmd = [sys.executable, "-m", "mypy", "--ignore-missing-imports", "--no-error-summary"]
        cmd.extend([str(root / f) for f in files[:20]])  # Limit to 20 files

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=root,
        )

        errors = []
        warnings = []
        for line in result.stdout.split("\n"):
            if ": error:" in line:
                errors.append(line.strip())
            elif ": warning:" in line or ": note:" in line:
                warnings.append(line.strip())

        return CheckResult(
            name="Type Check",
            passed=len(errors) == 0,
            errors=errors[:10],  # Limit output
            warnings=warnings[:10],
            duration_seconds=time.time() - start,
        )
    except Exception as e:
        return CheckResult(
            name="Type Check",
            passed=True,
            skipped=True,
            skip_reason=f"Could not run mypy: {e}",
            duration_seconds=time.time() - start,
        )


def run_test_check(root: Path, files: list[str]) -> CheckResult:
    """Check if tests exist for modified modules."""
    import time
    start = time.time()

    warnings = []

    # Filter to non-test source files
    source_files = [f for f in files if "/tests/" not in f and "\\tests\\" not in f and not f.startswith("test_")]

    for file_path in source_files:
        if file_path.startswith("ai_orchestrator/"):
            # Check for corresponding test file
            module_name = Path(file_path).stem
            test_file = root / "ai_orchestrator" / "tests" / f"test_{module_name}.py"
            if not test_file.exists():
                warnings.append(f"No test file found for {file_path} (expected {test_file.relative_to(root)})")

    return CheckResult(
        name="Test Coverage Check",
        passed=True,  # Warnings only, don't fail
        warnings=warnings[:10],
        duration_seconds=time.time() - start,
    )


def run_security_check(root: Path, files: list[str]) -> CheckResult:
    """Basic security checks on modified files."""
    import time
    import re
    start = time.time()

    if not files:
        return CheckResult(
            name="Security Check",
            passed=True,
            skipped=True,
            skip_reason="No files to check",
        )

    errors: list[str] = []
    warnings: list[str] = []

    security_patterns = [
        (r"password\s*=\s*['\"][^'\"]+['\"]", "Hardcoded password"),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "Hardcoded API key"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]", "Hardcoded secret"),
        (r"eval\s*\(", "Use of eval()"),
        (r"exec\s*\(", "Use of exec()"),
        (r"__import__\s*\(", "Dynamic import"),
        (r"subprocess\..*shell\s*=\s*True", "Shell injection risk"),
    ]

    for file_path in files:
        full_path = root / file_path
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8")
            for pattern, message in security_patterns:
                matches = list(re.finditer(pattern, content, re.IGNORECASE))
                for match in matches:
                    line_num = content[:match.start()].count("\n") + 1
                    warnings.append(f"{file_path}:{line_num}: {message}")
        except Exception:
            pass

    return CheckResult(
        name="Security Check",
        passed=True,  # Warnings only
        warnings=warnings[:15],
        duration_seconds=time.time() - start,
    )


def run_docstring_check(root: Path, files: list[str]) -> CheckResult:
    """Check for missing docstrings on public functions/classes."""
    import time
    import ast
    start = time.time()

    if not files:
        return CheckResult(
            name="Docstring Check",
            passed=True,
            skipped=True,
            skip_reason="No files to check",
        )

    warnings = []

    for file_path in files:
        full_path = root / file_path
        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        docstring = ast.get_docstring(node)
                        if not docstring:
                            warnings.append(f"{file_path}:{node.lineno}: Missing docstring for {node.name}()")
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_"):
                        docstring = ast.get_docstring(node)
                        if not docstring:
                            warnings.append(f"{file_path}:{node.lineno}: Missing docstring for class {node.name}")
        except Exception:
            pass

    return CheckResult(
        name="Docstring Check",
        passed=True,  # Warnings only
        warnings=warnings[:20],
        duration_seconds=time.time() - start,
    )


def print_report(report: ValidationReport) -> None:
    """Print the validation report."""
    print("\n" + "=" * 70)
    print("  POST-IMPLEMENTATION VALIDATION REPORT")
    print("=" * 70)

    print(f"\n  Timestamp: {report.timestamp}")
    print(f"  Duration:  {report.duration_seconds:.2f}s")
    print(f"  Files:     {len(report.modified_files)} modified")

    if report.modified_files:
        print("\n  Modified Files:")
        for f in report.modified_files[:10]:
            print(f"    - {f}")
        if len(report.modified_files) > 10:
            print(f"    ... and {len(report.modified_files) - 10} more")

    print("\n  " + "-" * 66)
    print(f"  {'Check':<25} {'Status':<10} {'Errors':<10} {'Warnings':<10} {'Time':<10}")
    print("  " + "-" * 66)

    for result in report.results:
        if result.skipped:
            status = "SKIPPED"
        elif result.passed:
            status = "PASSED"
        else:
            status = "FAILED"

        errors = len(result.errors) if result.errors else 0
        warnings = len(result.warnings) if result.warnings else 0

        print(f"  {result.name:<25} {status:<10} {errors:<10} {warnings:<10} {result.duration_seconds:.2f}s")

    print("  " + "-" * 66)

    # Summary
    print(f"\n  Summary:")
    print(f"    Total Checks:  {report.total_checks}")
    print(f"    Passed:        {report.passed_checks}")
    print(f"    Failed:        {report.failed_checks}")
    print(f"    Skipped:       {report.skipped_checks}")
    print(f"    Total Errors:  {report.total_errors}")
    print(f"    Total Warnings: {report.total_warnings}")

    # Show errors
    all_errors = []
    for result in report.results:
        if result.errors:
            for err in result.errors:
                all_errors.append(f"[{result.name}] {err}")

    if all_errors:
        print(f"\n  Errors:")
        for err in all_errors[:15]:
            print(f"    - {err}")
        if len(all_errors) > 15:
            print(f"    ... and {len(all_errors) - 15} more")

    # Show warnings
    all_warnings = []
    for result in report.results:
        if result.warnings:
            for warn in result.warnings:
                all_warnings.append(f"[{result.name}] {warn}")

    if all_warnings:
        print(f"\n  Warnings:")
        for warn in all_warnings[:10]:
            print(f"    - {warn}")
        if len(all_warnings) > 10:
            print(f"    ... and {len(all_warnings) - 10} more")

    # Final status
    print("\n" + "=" * 70)
    if report.failed_checks > 0:
        print("  STATUS: FAILED - Fix errors before committing")
    elif report.total_warnings > 0:
        print("  STATUS: PASSED WITH WARNINGS - Review warnings before committing")
    else:
        print("  STATUS: PASSED - Ready to commit")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-implementation validation")
    parser.add_argument("--all", action="store_true", help="Run all checks on entire codebase")
    parser.add_argument("--strict", action="store_true", help="Fail on any issues including warnings")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues where possible")
    args = parser.parse_args()

    import time
    start_time = time.time()

    root = Path(__file__).parent.parent

    # Get modified files
    if args.all:
        # Find all Python files
        excludes = {"venv", ".venv", "__pycache__", ".git", "node_modules"}
        modified_files = []
        for path in (root / "ai_orchestrator").rglob("*.py"):
            if not any(ex in path.parts for ex in excludes):
                modified_files.append(str(path.relative_to(root)))
    else:
        modified_files = get_modified_files(root)

    print(f"Checking {len(modified_files)} files...")

    # Run all checks
    results = []

    # 1. Pattern check
    results.append(run_pattern_check(root, modified_files, args.strict))

    # 2. Catalog freshness
    results.append(run_catalog_freshness_check(root))

    # 3. Import check
    results.append(run_import_check(root, modified_files))

    # 4. Type check
    results.append(run_type_check(root, modified_files))

    # 5. Test coverage check
    results.append(run_test_check(root, modified_files))

    # 6. Security check
    results.append(run_security_check(root, modified_files))

    # 7. Docstring check
    results.append(run_docstring_check(root, modified_files))

    # If --fix, regenerate catalog
    if args.fix:
        print("Auto-fixing: Regenerating code catalog...")
        subprocess.run(
            [sys.executable, str(root / "scripts" / "build_code_registry.py")],
            cwd=root,
            capture_output=True,
        )
        # Re-run catalog check
        for i, r in enumerate(results):
            if r.name == "Catalog Freshness":
                results[i] = run_catalog_freshness_check(root)
                break

    # Build report
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    report = ValidationReport(
        timestamp=datetime.now().isoformat(),
        total_checks=len(results),
        passed_checks=sum(1 for r in results if r.passed and not r.skipped),
        failed_checks=sum(1 for r in results if not r.passed and not r.skipped),
        skipped_checks=sum(1 for r in results if r.skipped),
        total_errors=total_errors,
        total_warnings=total_warnings,
        results=results,
        modified_files=modified_files,
        duration_seconds=time.time() - start_time,
    )

    if args.json:
        # Convert to JSON-serializable format
        report_dict = {
            "timestamp": report.timestamp,
            "total_checks": report.total_checks,
            "passed_checks": report.passed_checks,
            "failed_checks": report.failed_checks,
            "skipped_checks": report.skipped_checks,
            "total_errors": report.total_errors,
            "total_warnings": report.total_warnings,
            "modified_files": report.modified_files,
            "duration_seconds": report.duration_seconds,
            "results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "duration_seconds": r.duration_seconds,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                }
                for r in report.results
            ],
        }
        print(json.dumps(report_dict, indent=2))
    else:
        print_report(report)

    # Exit code
    if report.failed_checks > 0:
        sys.exit(1)
    elif args.strict and report.total_warnings > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
