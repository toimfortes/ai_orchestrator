#!/usr/bin/env python3
"""Check code against best practice patterns.

Validates Python files against the patterns defined in best_practices/patterns.json.
Used as part of the "Measure Twice, Cut Once" workflow.

Usage:
    python scripts/check_patterns.py <file_path>
    python scripts/check_patterns.py ai_orchestrator/core/orchestrator.py
    python scripts/check_patterns.py --all  # Check all files
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def load_patterns(patterns_path: Path) -> dict[str, Any]:
    """Load the patterns configuration."""
    if not patterns_path.exists():
        print(f"Error: Patterns file not found at {patterns_path}")
        sys.exit(1)
    result: dict[str, Any] = json.loads(patterns_path.read_text())
    return result


def is_in_comment_or_docstring(content: str, match_start: int) -> bool:
    """Check if a match position is inside a comment or docstring."""
    lines = content[:match_start].split("\n")
    current_line = lines[-1] if lines else ""

    # Check if on a comment line
    stripped = current_line.lstrip()
    if stripped.startswith("#"):
        return True

    # Check if in a docstring by counting triple quotes before match
    before_match = content[:match_start]
    triple_double = before_match.count('"""')
    triple_single = before_match.count("'''")

    # If odd number of triple quotes, we're inside a docstring
    if triple_double % 2 == 1 or triple_single % 2 == 1:
        return True

    # Check if there's a # before the match on the same line
    if "#" in current_line:
        hash_pos = current_line.find("#")
        match_pos_in_line = len(current_line) - (match_start - content.rfind("\n", 0, match_start) - 1)
        if hash_pos < len(current_line) - match_pos_in_line:
            return True

    return False


def check_hardcoded_values(content: str, file_path: str) -> list[dict[str, Any]]:
    """Check for hardcoded values that should be in settings."""
    violations: list[dict[str, Any]] = []

    # Check for hardcoded model IDs
    model_patterns = [
        r"['\"]claude-\d+[^'\"]*['\"]",
        r"['\"]gpt-\d+[^'\"]*['\"]",
        r"['\"]gemini-[^'\"]*['\"]",
        r"['\"]o\d+-[^'\"]*['\"]",  # OpenAI o1, o3, etc.
    ]

    for pattern in model_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            # Skip if in settings file
            if "settings.py" in file_path:
                continue
            # Skip if in comment or docstring
            if is_in_comment_or_docstring(content, match.start()):
                continue
            line_num = content[:match.start()].count("\n") + 1
            violations.append({
                "rule": "CFG001",
                "file": file_path,
                "line": line_num,
                "message": f"Hardcoded model ID: {match.group()}",
                "severity": "error",
            })

    # Check for hardcoded timeouts (only flag values > 30 seconds)
    timeout_pattern = r"timeout\s*=\s*(\d+)"
    matches = re.finditer(timeout_pattern, content, re.IGNORECASE)
    for match in matches:
        if "settings.py" in file_path:
            continue
        # Skip if in comment or docstring
        if is_in_comment_or_docstring(content, match.start()):
            continue
        # Skip small timeouts (likely health checks)
        timeout_value = int(match.group(1))
        if timeout_value <= 30:
            continue
        # Skip test files
        if "/tests/" in file_path or "\\tests\\" in file_path:
            continue
        line_num = content[:match.start()].count("\n") + 1
        violations.append({
            "rule": "CFG002",
            "file": file_path,
            "line": line_num,
            "message": f"Hardcoded timeout: {match.group()} (consider using settings)",
            "severity": "warning",
        })

    return violations


def check_error_handling(content: str, file_path: str) -> list[dict[str, Any]]:
    """Check for error handling anti-patterns."""
    violations: list[dict[str, Any]] = []

    # Check for bare except: pass
    bare_except = re.finditer(r"except\s*(?:Exception)?\s*:\s*\n\s*pass", content)
    for match in bare_except:
        line_num = content[:match.start()].count("\n") + 1
        violations.append({
            "rule": "ERR001",
            "file": file_path,
            "line": line_num,
            "message": "Bare except with pass - exceptions should be logged or re-raised",
            "severity": "error",
        })

    # Check for error logging without exc_info in except blocks
    # Only flag logger.error() inside except blocks where there's an exception to log
    lines = content.split("\n")
    in_except_block = False
    except_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        current_indent = len(line) - len(stripped)

        # Detect start of except block
        if stripped.startswith("except ") and ":" in stripped:
            in_except_block = True
            except_indent = current_indent
            continue

        # Detect end of except block (line with same or less indentation)
        if in_except_block and stripped and current_indent <= except_indent:
            in_except_block = False

        # Check for logger.error in except block
        if in_except_block and "logger.error(" in line and "exc_info" not in line:
            violations.append({
                "rule": "LOG001",
                "file": file_path,
                "line": i + 1,
                "message": "logger.error() in except block without exc_info=True",
                "severity": "warning",
            })

    return violations


def check_async_patterns(content: str, file_path: str) -> list[dict[str, Any]]:
    """Check for async anti-patterns."""
    violations: list[dict[str, Any]] = []

    # Check for blocking calls in async functions
    async_func_pattern = r"async\s+def\s+\w+[^:]+:"
    async_funcs = list(re.finditer(async_func_pattern, content))

    blocking_calls = [
        (r"open\s*\([^)]*\)\.read", "Blocking file read"),
        (r"time\.sleep\s*\(", "Blocking sleep (use asyncio.sleep)"),
        (r"requests\.(get|post|put|delete)\s*\(", "Blocking HTTP request"),
    ]

    for func_match in async_funcs:
        # Find the function body (approximate - until next top-level function or class)
        func_start = func_match.end()
        next_def = re.search(r"\n(?:async\s+)?def\s+|^class\s+", content[func_start:], re.MULTILINE)
        func_end = func_start + next_def.start() if next_def else len(content)
        func_body = content[func_start:func_end]

        for pattern, message in blocking_calls:
            for block_match in re.finditer(pattern, func_body):
                # Check if this blocking call is inside a nested sync function
                # (which might run in a thread, so blocking is OK)
                code_before_match = func_body[:block_match.start()]
                nested_def_matches = list(re.finditer(r"\n\s+def\s+\w+[^:]*:", code_before_match))
                if nested_def_matches:
                    # Check if there's a nested def that hasn't been closed
                    # by checking indentation - if blocking call is more indented
                    # than the nested def, it's likely inside it
                    last_nested_def = nested_def_matches[-1]
                    nested_def_indent = len(last_nested_def.group()) - len(last_nested_def.group().lstrip())

                    # Get the line with the blocking call
                    lines_before = code_before_match[last_nested_def.end():].split("\n")
                    # If we're still inside the nested function, skip this violation
                    # (simple heuristic: nested functions for threading are usually short)
                    if len(lines_before) < 20:
                        continue

                line_num = content[:func_start + block_match.start()].count("\n") + 1
                violations.append({
                    "rule": "ASYNC001",
                    "file": file_path,
                    "line": line_num,
                    "message": f"{message} in async function",
                    "severity": "error",
                })

    return violations


def check_test_locations(content: str, file_path: str) -> list[dict[str, Any]]:
    """Check that tests are in the correct directory."""
    violations: list[dict[str, Any]] = []

    if "test_" in Path(file_path).name:
        parts = Path(file_path).parts
        if "tests" not in parts:
            violations.append({
                "rule": "TEST001",
                "file": file_path,
                "line": 1,
                "message": "Test file not in tests/ directory",
                "severity": "error",
            })

    return violations


def check_simulated_outputs(content: str, file_path: str) -> list[dict[str, Any]]:
    """Check for simulated/dummy outputs in return statements."""
    violations: list[dict[str, Any]] = []

    # Skip test files - they legitimately use mocks
    if "/tests/" in file_path or "\\tests\\" in file_path:
        return violations

    # Look for return statements with simulated/dummy values
    # This is more targeted than just looking for the words
    simulated_return_patterns = [
        r"return\s*\{[^}]*['\"]simulated['\"][^}]*\}",
        r"return\s*\{[^}]*['\"]dummy['\"][^}]*\}",
        r"return\s*['\"]simulated['\"]",
        r"return\s*['\"]dummy['\"]",
    ]

    for pattern in simulated_return_patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE):
            # Skip if in comment or docstring
            if is_in_comment_or_docstring(content, match.start()):
                continue
            line_num = content[:match.start()].count("\n") + 1
            violations.append({
                "rule": "TEST002",
                "file": file_path,
                "line": line_num,
                "message": "Possible simulated/dummy return value - use real implementation",
                "severity": "warning",
            })

    return violations


def check_pydantic_patterns(content: str, file_path: str) -> list[dict[str, Any]]:
    """Check for Pydantic/dataclass anti-patterns.

    Only checks mutable defaults inside @dataclass or BaseModel classes,
    not local variables in functions.
    """
    violations: list[dict[str, Any]] = []

    # Find dataclass and BaseModel class definitions
    class_pattern = re.compile(
        r"(?:@dataclass[^\n]*\n)?class\s+\w+\s*(?:\([^)]*(?:BaseModel|ABC)[^)]*\)|:)",
        re.MULTILINE,
    )

    lines = content.split("\n")

    for class_match in class_pattern.finditer(content):
        # Find the class body (from class definition to next unindented line)
        class_start_line = content[:class_match.end()].count("\n")

        # Determine base indentation of class body
        class_body_start = class_match.end()
        next_newline = content.find("\n", class_body_start)
        if next_newline == -1:
            continue

        # Find end of class (next line with same or less indentation)
        class_end = len(content)
        for i in range(class_start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() and not line.startswith((" ", "\t")):
                # Found unindented line - class ends here
                class_end = sum(len(lines[j]) + 1 for j in range(i))
                break

        class_body = content[class_match.end():class_end]

        # Check for mutable defaults in this class body
        # Only check lines at class field level (4 spaces), skip method bodies (8+ spaces)
        in_method = False
        for line_offset, line in enumerate(class_body.split("\n")):
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(stripped)

            # Track when we enter/exit a method
            if stripped.startswith("def ") or stripped.startswith("async def "):
                in_method = True
                continue

            # Exit method when we see a line at class level (4 spaces)
            if indent == 4:
                in_method = False

            # Skip if we're inside a method body
            if in_method or indent >= 8:
                continue

            # Check for mutable list default (only at class field level)
            if re.search(r":\s*list\[[^\]]+\]\s*=\s*\[\]", line):
                violations.append({
                    "rule": "PYD001",
                    "file": file_path,
                    "line": class_start_line + line_offset + 2,
                    "message": "Use Field(default_factory=list) instead of []",
                    "severity": "warning",
                })

            # Check for mutable dict default
            if re.search(r":\s*dict\[[^\]]+\]\s*=\s*\{\}", line):
                violations.append({
                    "rule": "PYD001",
                    "file": file_path,
                    "line": class_start_line + line_offset + 2,
                    "message": "Use Field(default_factory=dict) instead of {}",
                    "severity": "warning",
                })

    return violations


def check_file(file_path: Path, root: Path) -> list[dict[str, Any]]:
    """Run all checks on a single file."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as e:
        return [{
            "rule": "FILE",
            "file": str(file_path.relative_to(root)),
            "line": 0,
            "message": f"Could not read file: {e}",
            "severity": "error",
        }]

    rel_path = str(file_path.relative_to(root))
    violations: list[dict[str, Any]] = []

    violations.extend(check_hardcoded_values(content, rel_path))
    violations.extend(check_error_handling(content, rel_path))
    violations.extend(check_async_patterns(content, rel_path))
    violations.extend(check_test_locations(content, rel_path))
    violations.extend(check_simulated_outputs(content, rel_path))
    violations.extend(check_pydantic_patterns(content, rel_path))

    return violations


def find_python_files(root: Path) -> list[Path]:
    """Find all Python files, excluding common excludes."""
    excludes = {"venv", ".venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}
    files = []
    for path in (root / "ai_orchestrator").rglob("*.py"):
        if not any(ex in path.parts for ex in excludes):
            files.append(path)
    return sorted(files)


def print_violations(violations: list[dict[str, Any]]) -> None:
    """Print violations in a readable format."""
    if not violations:
        print("\n  [OK] No pattern violations found.\n")
        return

    errors = [v for v in violations if v["severity"] == "error"]
    warnings = [v for v in violations if v["severity"] == "warning"]

    print("\n" + "=" * 60)
    print("  PATTERN CHECK RESULTS")
    print("=" * 60)

    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for v in errors:
            print(f"    [{v['rule']}] {v['file']}:{v['line']}")
            print(f"           {v['message']}")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for v in warnings:
            print(f"    [{v['rule']}] {v['file']}:{v['line']}")
            print(f"           {v['message']}")

    print("\n" + "=" * 60)
    print(f"  Total: {len(errors)} errors, {len(warnings)} warnings")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check code against best practice patterns")
    parser.add_argument("file_path", nargs="?", help="File path to check")
    parser.add_argument("--all", action="store_true", help="Check all Python files")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--strict", action="store_true", help="Exit with error code if any violations")
    args = parser.parse_args()

    root = Path(__file__).parent.parent

    if args.all:
        files = find_python_files(root)
        print(f"Checking {len(files)} files...")
    elif args.file_path:
        file_path = Path(args.file_path)
        if not file_path.is_absolute():
            file_path = root / file_path
        files = [file_path]
    else:
        parser.print_help()
        sys.exit(1)

    all_violations = []
    for f in files:
        violations = check_file(f, root)
        all_violations.extend(violations)

    if args.json:
        print(json.dumps(all_violations, indent=2))
    else:
        print_violations(all_violations)

    if args.strict and all_violations:
        errors = [v for v in all_violations if v["severity"] == "error"]
        if errors:
            sys.exit(1)


if __name__ == "__main__":
    main()
