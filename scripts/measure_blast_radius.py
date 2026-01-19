#!/usr/bin/env python3
"""Measure blast radius for code changes.

Analyzes which modules would be affected if given files are modified.
Uses the code catalog's import graph to determine direct and transitive
dependencies.

Usage:
    python scripts/measure_blast_radius.py <file_path>
    python scripts/measure_blast_radius.py ai_orchestrator/core/orchestrator.py
    python scripts/measure_blast_radius.py --all-critical  # Show high-impact modules
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_catalog(catalog_path: Path) -> dict[str, Any]:
    """Load the code catalog."""
    if not catalog_path.exists():
        print(f"Error: Catalog not found at {catalog_path}")
        print("Run: python scripts/build_code_registry.py")
        sys.exit(1)
    result: dict[str, Any] = json.loads(catalog_path.read_text())
    return result


def _normalize_catalog_path(path: str) -> str:
    """Normalize a path from the catalog for consistent lookup."""
    return path.replace("\\", "/")


def build_reverse_import_graph(import_graph: dict[str, list[str]], modules: list[dict[str, Any]]) -> dict[str, set[str]]:
    """Build reverse import graph: module -> modules that import it."""
    # Create a mapping from module names to file paths (normalized)
    module_to_path = {}
    for module in modules:
        if "error" not in module:
            path = _normalize_catalog_path(module["path"])
            # Convert path to module name
            module_name = path.replace("/", ".").removesuffix(".py")
            module_to_path[module_name] = path

    # Build reverse graph with normalized paths
    reverse_graph: dict[str, set[str]] = defaultdict(set)

    for importer_path, imports in import_graph.items():
        normalized_importer = _normalize_catalog_path(importer_path)
        for imported_module in imports:
            # Find the file path for the imported module
            # Check various potential matches
            for mod_name, mod_path in module_to_path.items():
                if imported_module in mod_name or mod_name.endswith(imported_module.replace(".", "/")):
                    reverse_graph[mod_path].add(normalized_importer)
                    break

    return dict(reverse_graph)


def find_transitive_dependents(
    module_path: str,
    reverse_graph: dict[str, set[str]],
    visited: set[str] | None = None
) -> set[str]:
    """Find all modules that directly or transitively depend on a module."""
    if visited is None:
        visited = set()

    if module_path in visited:
        return set()

    visited.add(module_path)
    dependents = set()

    direct_dependents = reverse_graph.get(module_path, set())
    dependents.update(direct_dependents)

    for dep in direct_dependents:
        transitive = find_transitive_dependents(dep, reverse_graph, visited)
        dependents.update(transitive)

    return dependents


def categorize_risk(dependent_count: int, total_modules: int) -> tuple[str, str]:
    """Categorize the blast radius risk level."""
    percentage = (dependent_count / total_modules) * 100 if total_modules > 0 else 0

    if percentage > 30:
        return "CRITICAL", "[!!!]"
    elif percentage > 15:
        return "HIGH", "[!!]"
    elif percentage > 5:
        return "MEDIUM", "[!]"
    elif dependent_count > 0:
        return "LOW", "[.]"
    else:
        return "MINIMAL", "[-]"


def normalize_path(path: str) -> str:
    """Normalize a file path for consistent comparison."""
    # Use forward slashes consistently
    normalized = path.replace("\\", "/")
    # Extract relative path if absolute
    if "ai_orchestrator" in normalized:
        parts = normalized.split("/")
        try:
            idx = parts.index("ai_orchestrator")
            normalized = "/".join(parts[idx:])
        except ValueError:
            pass
    return normalized


def analyze_blast_radius(file_path: str, catalog: dict[str, Any]) -> dict[str, Any]:
    """Analyze the blast radius for a given file."""
    modules = catalog["modules"]
    import_graph = catalog.get("import_graph", {})

    # Normalize file path
    normalized_path = normalize_path(file_path)

    # Build reverse import graph
    reverse_graph = build_reverse_import_graph(import_graph, modules)

    # Find the module in the catalog
    module_info = None
    for m in modules:
        catalog_path = _normalize_catalog_path(m["path"])
        if catalog_path == normalized_path or catalog_path.endswith(normalized_path):
            module_info = m
            normalized_path = catalog_path
            break

    # Find all dependents
    direct_dependents = reverse_graph.get(normalized_path, set())
    all_dependents = find_transitive_dependents(normalized_path, reverse_graph)

    # Calculate risk
    total_modules = len([m for m in modules if "error" not in m])
    risk_level, risk_icon = categorize_risk(len(all_dependents), total_modules)

    return {
        "file": normalized_path,
        "module_info": module_info,
        "direct_dependents": sorted(direct_dependents),
        "transitive_dependents": sorted(all_dependents - direct_dependents),
        "total_affected": len(all_dependents),
        "total_modules": total_modules,
        "percentage_affected": round((len(all_dependents) / total_modules) * 100, 1) if total_modules > 0 else 0,
        "risk_level": risk_level,
        "risk_icon": risk_icon,
    }


def find_critical_modules(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Find all modules with high blast radius."""
    modules = catalog["modules"]
    import_graph = catalog.get("import_graph", {})
    reverse_graph = build_reverse_import_graph(import_graph, modules)

    results: list[dict[str, Any]] = []
    for module in modules:
        if "error" in module:
            continue

        path = _normalize_catalog_path(module["path"])
        all_dependents = find_transitive_dependents(path, reverse_graph)
        total_modules = len([m for m in modules if "error" not in m])
        risk_level, risk_icon = categorize_risk(len(all_dependents), total_modules)

        if len(all_dependents) > 0:
            results.append({
                "file": path,
                "dependents": len(all_dependents),
                "percentage": round((len(all_dependents) / total_modules) * 100, 1),
                "risk_level": risk_level,
                "risk_icon": risk_icon,
            })

    # Sort by number of dependents (highest first)
    results.sort(key=lambda x: int(x["dependents"]), reverse=True)
    return results


def print_blast_radius_report(analysis: dict[str, Any]) -> None:
    """Print a formatted blast radius report."""
    print("\n" + "=" * 60)
    print("  BLAST RADIUS ANALYSIS")
    print("=" * 60)

    print(f"\n  File: {analysis['file']}")
    print(f"  Risk: {analysis['risk_icon']} {analysis['risk_level']}")
    print(f"  Total Affected: {analysis['total_affected']} / {analysis['total_modules']} modules ({analysis['percentage_affected']}%)")

    if analysis["module_info"]:
        info = analysis["module_info"]
        print(f"\n  Module Info:")
        print(f"    Classes: {len(info.get('classes', []))}")
        print(f"    Functions: {len(info.get('functions', []))}")
        print(f"    Lines: {info.get('lines', 0)}")

    if analysis["direct_dependents"]:
        print(f"\n  Direct Dependents ({len(analysis['direct_dependents'])}):")
        for dep in analysis["direct_dependents"][:10]:
            print(f"    -> {dep}")
        if len(analysis["direct_dependents"]) > 10:
            print(f"    ... and {len(analysis['direct_dependents']) - 10} more")

    if analysis["transitive_dependents"]:
        print(f"\n  Transitive Dependents ({len(analysis['transitive_dependents'])}):")
        for dep in analysis["transitive_dependents"][:10]:
            print(f"    --> {dep}")
        if len(analysis["transitive_dependents"]) > 10:
            print(f"    ... and {len(analysis['transitive_dependents']) - 10} more")

    print("\n" + "=" * 60 + "\n")


def print_critical_modules_report(critical: list[dict[str, Any]]) -> None:
    """Print a report of critical/high-impact modules."""
    print("\n" + "=" * 60)
    print("  HIGH-IMPACT MODULES (by blast radius)")
    print("=" * 60 + "\n")

    if not critical:
        print("  No modules with dependents found.")
        return

    print(f"  {'Module':<50} {'Deps':>6} {'%':>6} {'Risk':>10}")
    print("  " + "-" * 76)

    for item in critical[:20]:
        name = item["file"]
        if len(name) > 48:
            name = "..." + name[-45:]
        print(f"  {name:<50} {item['dependents']:>6} {item['percentage']:>5.1f}% {item['risk_icon']} {item['risk_level']:<8}")

    if len(critical) > 20:
        print(f"\n  ... and {len(critical) - 20} more modules with dependents")

    print("\n" + "=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure blast radius for code changes")
    parser.add_argument("file_path", nargs="?", help="File path to analyze")
    parser.add_argument("--all-critical", action="store_true", help="Show all high-impact modules")
    parser.add_argument("--catalog", default="data/code_catalog.json", help="Path to code catalog")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    catalog_path = root / args.catalog
    catalog = load_catalog(catalog_path)

    if args.all_critical:
        critical = find_critical_modules(catalog)
        if args.json:
            print(json.dumps(critical, indent=2))
        else:
            print_critical_modules_report(critical)
    elif args.file_path:
        analysis = analyze_blast_radius(args.file_path, catalog)
        if args.json:
            print(json.dumps(analysis, indent=2))
        else:
            print_blast_radius_report(analysis)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
