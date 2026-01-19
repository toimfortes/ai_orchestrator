#!/usr/bin/env python3
"""Run all debugging checks in sequence.

This is the master debugging script that runs all validation tools
in the correct order and provides a comprehensive report.

Usage:
    python scripts/debug_everything.py              # Standard check
    python scripts/debug_everything.py --quick      # Quick check (diff only)
    python scripts/debug_everything.py --thorough   # Full check with tests
    python scripts/debug_everything.py --fix        # Auto-fix issues
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class StepResult:
    """Result of a debugging step."""
    name: str
    passed: bool
    duration: float
    output: str
    return_code: int


def run_step(name: str, cmd: list[str], root: Path, timeout: int = 120) -> StepResult:
    """Run a debugging step."""
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=root,
            timeout=timeout,
        )
        return StepResult(
            name=name,
            passed=result.returncode == 0,
            duration=time.time() - start,
            output=result.stdout + result.stderr,
            return_code=result.returncode,
        )
    except subprocess.TimeoutExpired:
        return StepResult(
            name=name,
            passed=False,
            duration=time.time() - start,
            output="Command timed out",
            return_code=-1,
        )
    except Exception as e:
        return StepResult(
            name=name,
            passed=False,
            duration=time.time() - start,
            output=str(e),
            return_code=-1,
        )


def print_step_start(name: str) -> None:
    """Print step start marker."""
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")


def print_step_result(result: StepResult, verbose: bool = False) -> None:
    """Print step result."""
    status = "PASSED" if result.passed else "FAILED"
    print(f"\n  Result: {status} ({result.duration:.2f}s)")

    if verbose or not result.passed:
        # Show output (limited)
        lines = result.output.strip().split("\n")
        if lines:
            print("\n  Output:")
            for line in lines[:30]:
                print(f"    {line[:100]}")
            if len(lines) > 30:
                print(f"    ... ({len(lines) - 30} more lines)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all debugging checks")
    parser.add_argument("--quick", action="store_true", help="Quick check (diff only)")
    parser.add_argument("--thorough", action="store_true", help="Full check with tests")
    parser.add_argument("--fix", action="store_true", help="Auto-fix issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    start_time = time.time()
    results: list[StepResult] = []

    print("\n" + "=" * 60)
    print("  DEBUG EVERYTHING")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Step 1: Build/refresh code catalog
    if not args.quick:
        print_step_start("Build Code Catalog")
        result = run_step(
            "Code Catalog",
            [sys.executable, "scripts/build_code_registry.py"],
            root,
        )
        results.append(result)
        print_step_result(result, args.verbose)

    # Step 2: Check diff patterns (fast feedback)
    print_step_start("Diff Pattern Check")
    result = run_step(
        "Diff Patterns",
        [sys.executable, "scripts/check_diff_patterns.py", "--all"],
        root,
    )
    results.append(result)
    print_step_result(result, args.verbose)

    if args.quick:
        # Quick mode - stop here
        pass
    else:
        # Step 3: Full pattern check
        print_step_start("Full Pattern Check")
        result = run_step(
            "Pattern Check",
            [sys.executable, "scripts/check_patterns.py", "--all"],
            root,
        )
        results.append(result)
        print_step_result(result, args.verbose)

        # Step 4: Post-implementation check
        print_step_start("Post-Implementation Check")
        cmd = [sys.executable, "scripts/post_implementation_check.py"]
        if args.fix:
            cmd.append("--fix")
        result = run_step(
            "Post-Implementation",
            cmd,
            root,
            timeout=180,
        )
        results.append(result)
        print_step_result(result, args.verbose)

        # Step 5: Implementation verification
        print_step_start("Implementation Verification")
        cmd = [sys.executable, "scripts/verify_implementation.py", "--run-build"]
        if args.thorough:
            cmd.append("--run-tests")
        result = run_step(
            "Verification",
            cmd,
            root,
            timeout=300 if args.thorough else 60,
        )
        results.append(result)
        print_step_result(result, args.verbose)

        # Step 6: High-impact module check
        print_step_start("High-Impact Module Analysis")
        result = run_step(
            "Blast Radius",
            [sys.executable, "scripts/measure_blast_radius.py", "--all-critical"],
            root,
        )
        results.append(result)
        print_step_result(result, args.verbose)

    # Summary
    total_duration = time.time() - start_time
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\n  Total Steps:    {len(results)}")
    print(f"  Passed:         {passed}")
    print(f"  Failed:         {failed}")
    print(f"  Total Duration: {total_duration:.2f}s")

    print("\n  Step Details:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"    [{status}] {result.name:<25} ({result.duration:.2f}s)")

    # Failed steps
    failed_steps = [r for r in results if not r.passed]
    if failed_steps:
        print("\n  Failed Steps:")
        for result in failed_steps:
            print(f"    - {result.name}")
            # Show first line of output as hint
            first_line = result.output.strip().split("\n")[0][:60] if result.output else "No output"
            print(f"      Hint: {first_line}")

    print("\n" + "=" * 60)
    if failed:
        print(f"  RESULT: FAILED ({failed} issues)")
    else:
        print("  RESULT: ALL CHECKS PASSED")
    print("=" * 60 + "\n")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
