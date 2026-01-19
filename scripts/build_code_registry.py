#!/usr/bin/env python3
"""Build code registry/catalog for AI agent context awareness.

Generates a machine-readable map of the codebase architecture that can be
passed to AI agents doing implementation work. This follows the "Measure
Twice, Cut Once" workflow from uk_family_law_ai.

Usage:
    python scripts/build_code_registry.py           # Regenerate catalog
    python scripts/build_code_registry.py --check   # Check freshness
    python scripts/build_code_registry.py --summary # Print summary only
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


def find_python_files(root: Path) -> list[Path]:
    """Find all Python files, excluding tests and venv."""
    excludes = {"venv", ".venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}
    files = []
    for path in root.rglob("*.py"):
        if not any(ex in path.parts for ex in excludes):
            files.append(path)
    return sorted(files)


def extract_module_info(file_path: Path, root: Path) -> dict[str, Any]:
    """Extract information about a Python module."""
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        return {
            "path": str(file_path.relative_to(root)),
            "error": str(e),
        }

    classes = []
    functions = []
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = [
                ast.unparse(base) if hasattr(ast, "unparse") else str(base)
                for base in node.bases
            ]
            methods = [
                n.name for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({
                "name": node.name,
                "bases": bases,
                "methods": methods,
                "line": node.lineno,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Only top-level functions (not methods)
            if isinstance(node, ast.AsyncFunctionDef):
                func_type = "async"
            else:
                func_type = "sync"
            # Check if it's a top-level function
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef) and node in ast.walk(parent):
                    break
            else:
                functions.append({
                    "name": node.name,
                    "type": func_type,
                    "line": node.lineno,
                })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    # Get docstring
    docstring = ast.get_docstring(tree) or ""

    return {
        "path": str(file_path.relative_to(root)),
        "classes": classes,
        "functions": functions,
        "imports": list(set(imports)),
        "docstring": docstring[:200] if docstring else None,
        "lines": len(content.splitlines()),
    }


def categorize_module(path: str) -> str:
    """Categorize a module by its path."""
    parts = Path(path).parts
    if "tests" in parts or path.startswith("test_"):
        return "tests"
    if "cli_adapters" in parts:
        return "cli_adapters"
    if "dashboard" in parts:
        return "dashboard"
    if "config" in parts:
        return "config"
    if "core" in parts:
        return "core"
    if "planning" in parts:
        return "planning"
    if "reviewing" in parts:
        return "reviewing"
    if "research" in parts:
        return "research"
    if "utils" in parts:
        return "utils"
    if "scripts" in parts:
        return "scripts"
    return "other"


def build_import_graph(modules: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Build a graph of module imports."""
    graph = defaultdict(list)
    module_names = {m["path"] for m in modules if "error" not in m}

    for module in modules:
        if "error" in module:
            continue
        for imp in module.get("imports", []):
            # Check if this import is internal
            if imp.startswith("ai_orchestrator"):
                graph[module["path"]].append(imp)

    return dict(graph)


def build_catalog(root: Path) -> dict[str, Any]:
    """Build the complete code catalog."""
    files = find_python_files(root / "ai_orchestrator")
    modules = [extract_module_info(f, root) for f in files]

    # Categorize modules
    by_category = defaultdict(list)
    for module in modules:
        if "error" not in module:
            cat = categorize_module(module["path"])
            by_category[cat].append(module)

    # Build import graph
    import_graph = build_import_graph(modules)

    # Extract key components
    cli_adapters = [
        m for m in modules
        if "cli_adapters" in m.get("path", "") and "error" not in m
    ]

    dashboard_components = [
        m for m in modules
        if "dashboard" in m.get("path", "") and "error" not in m
    ]

    # Count statistics
    total_classes = sum(len(m.get("classes", [])) for m in modules)
    total_functions = sum(len(m.get("functions", [])) for m in modules)
    total_lines = sum(m.get("lines", 0) for m in modules)

    return {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "summary": {
            "total_modules": len(modules),
            "total_classes": total_classes,
            "total_functions": total_functions,
            "total_lines": total_lines,
            "categories": {cat: len(mods) for cat, mods in by_category.items()},
        },
        "cli_adapters": [
            {
                "name": Path(m["path"]).stem,
                "path": m["path"],
                "classes": [c["name"] for c in m.get("classes", [])],
            }
            for m in cli_adapters
        ],
        "dashboard": {
            "api_endpoints": len([
                m for m in dashboard_components
                if "server" in m.get("path", "")
            ]),
            "templates": len([
                f for f in (root / "ai_orchestrator" / "dashboard" / "templates").glob("*.html")
            ]) if (root / "ai_orchestrator" / "dashboard" / "templates").exists() else 0,
            "static_files": {
                "js": len(list((root / "ai_orchestrator" / "dashboard" / "static" / "js").glob("*.js"))) if (root / "ai_orchestrator" / "dashboard" / "static" / "js").exists() else 0,
                "css": len(list((root / "ai_orchestrator" / "dashboard" / "static" / "css").glob("*.css"))) if (root / "ai_orchestrator" / "dashboard" / "static" / "css").exists() else 0,
            },
        },
        "modules": modules,
        "import_graph": import_graph,
        "by_category": {cat: [m["path"] for m in mods] for cat, mods in by_category.items()},
    }


def check_freshness(catalog_path: Path, root: Path) -> bool:
    """Check if the catalog is up to date."""
    if not catalog_path.exists():
        print("Catalog does not exist. Run without --check to generate.")
        return False

    catalog = json.loads(catalog_path.read_text())
    generated_at = datetime.fromisoformat(catalog["generated_at"].replace("Z", "+00:00"))

    # Check if any Python files are newer than the catalog
    files = find_python_files(root / "ai_orchestrator")
    for f in files:
        if datetime.fromtimestamp(f.stat().st_mtime, UTC) > generated_at:
            print(f"Catalog is stale. {f} was modified after catalog generation.")
            return False

    print(f"Catalog is up to date (generated {generated_at.isoformat()}).")
    return True


def print_summary(catalog: dict[str, Any]) -> None:
    """Print a summary of the catalog."""
    summary = catalog["summary"]
    print("\n" + "=" * 60)
    print("  AI Orchestrator Code Registry")
    print("=" * 60)
    print(f"\n  Generated: {catalog['generated_at']}")
    print(f"  Schema:    v{catalog['schema_version']}")
    print(f"\n  Statistics:")
    print(f"    Modules:   {summary['total_modules']}")
    print(f"    Classes:   {summary['total_classes']}")
    print(f"    Functions: {summary['total_functions']}")
    print(f"    Lines:     {summary['total_lines']:,}")
    print(f"\n  By Category:")
    for cat, count in sorted(summary["categories"].items()):
        print(f"    {cat}: {count}")
    print(f"\n  CLI Adapters: {len(catalog['cli_adapters'])}")
    for adapter in catalog["cli_adapters"]:
        print(f"    - {adapter['name']}: {', '.join(adapter['classes'])}")
    print(f"\n  Dashboard:")
    print(f"    Templates: {catalog['dashboard']['templates']}")
    print(f"    JS files:  {catalog['dashboard']['static_files']['js']}")
    print(f"    CSS files: {catalog['dashboard']['static_files']['css']}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build code registry for AI agents")
    parser.add_argument("--check", action="store_true", help="Check if catalog is fresh")
    parser.add_argument("--summary", action="store_true", help="Print summary only")
    parser.add_argument("--output", default="data/code_catalog.json", help="Output path")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    catalog_path = root / args.output

    if args.check:
        success = check_freshness(catalog_path, root)
        sys.exit(0 if success else 1)

    # Build the catalog
    print("Building code catalog...")
    catalog = build_catalog(root)

    # Ensure output directory exists
    catalog_path.parent.mkdir(parents=True, exist_ok=True)

    # Write catalog
    catalog_path.write_text(json.dumps(catalog, indent=2))
    print(f"Catalog written to {catalog_path}")

    # Print summary
    if args.summary or True:  # Always print summary
        print_summary(catalog)


if __name__ == "__main__":
    main()
