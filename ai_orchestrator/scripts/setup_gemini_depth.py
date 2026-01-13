#!/usr/bin/env python3
"""Setup Gemini CLI for deep, thorough responses.

This script installs GEMINI.md and optional system.md templates
to enable Claude-level depth from Gemini CLI.

Usage:
    python -m ai_orchestrator.scripts.setup_gemini_depth [--global] [--project PATH] [--system]

Options:
    --global        Install to ~/.gemini/GEMINI.md (applies to all projects)
    --project PATH  Install to PATH/GEMINI.md (project-specific)
    --system        Also install system.md override template
    --force         Overwrite existing files
    --show          Just show what would be installed
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Template locations relative to this script
SCRIPT_DIR = Path(__file__).parent
TEMPLATE_DIR = SCRIPT_DIR.parent / "templates" / "gemini"


def get_templates() -> dict[str, Path]:
    """Get available template files."""
    return {
        "GEMINI.md": TEMPLATE_DIR / "GEMINI.md",
        "system.md": TEMPLATE_DIR / "system.md",
    }


def install_global(force: bool = False, include_system: bool = False) -> None:
    """Install GEMINI.md to global ~/.gemini/ directory."""
    gemini_dir = Path.home() / ".gemini"
    gemini_dir.mkdir(exist_ok=True)

    templates = get_templates()

    # Install GEMINI.md
    target = gemini_dir / "GEMINI.md"
    if target.exists() and not force:
        print(f"  [SKIP] {target} already exists (use --force to overwrite)")
    else:
        shutil.copy(templates["GEMINI.md"], target)
        print(f"  [OK] Installed {target}")

    # Optionally install system.md
    if include_system:
        target = gemini_dir / "system.md"
        if target.exists() and not force:
            print(f"  [SKIP] {target} already exists (use --force to overwrite)")
        else:
            shutil.copy(templates["system.md"], target)
            print(f"  [OK] Installed {target}")
            print("  [INFO] Set GEMINI_SYSTEM_MD=true to enable system override")


def install_project(project_path: Path, force: bool = False, include_system: bool = False) -> None:
    """Install GEMINI.md to project directory."""
    if not project_path.exists():
        print(f"  [ERROR] Project path does not exist: {project_path}")
        sys.exit(1)

    templates = get_templates()

    # Install GEMINI.md at project root
    target = project_path / "GEMINI.md"
    if target.exists() and not force:
        print(f"  [SKIP] {target} already exists (use --force to overwrite)")
    else:
        shutil.copy(templates["GEMINI.md"], target)
        print(f"  [OK] Installed {target}")

    # Optionally install system.md in .gemini/
    if include_system:
        gemini_dir = project_path / ".gemini"
        gemini_dir.mkdir(exist_ok=True)
        target = gemini_dir / "system.md"
        if target.exists() and not force:
            print(f"  [SKIP] {target} already exists (use --force to overwrite)")
        else:
            shutil.copy(templates["system.md"], target)
            print(f"  [OK] Installed {target}")
            print("  [INFO] Set GEMINI_SYSTEM_MD=true to enable system override")


def show_templates() -> None:
    """Display template contents."""
    templates = get_templates()

    for name, path in templates.items():
        print(f"\n{'='*60}")
        print(f"Template: {name}")
        print(f"{'='*60}")
        if path.exists():
            print(path.read_text())
        else:
            print(f"[ERROR] Template not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup Gemini CLI for deep, thorough responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Install globally (all projects)
    python -m ai_orchestrator.scripts.setup_gemini_depth --global

    # Install to specific project
    python -m ai_orchestrator.scripts.setup_gemini_depth --project /path/to/project

    # Include system.md override
    python -m ai_orchestrator.scripts.setup_gemini_depth --global --system

    # Show templates without installing
    python -m ai_orchestrator.scripts.setup_gemini_depth --show
        """,
    )

    parser.add_argument(
        "--global",
        dest="install_global",
        action="store_true",
        help="Install to ~/.gemini/ (global config)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="Install to project directory",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Also install system.md override template",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show template contents without installing",
    )

    args = parser.parse_args()

    print("Gemini CLI Depth Enhancement Setup")
    print("=" * 40)

    if args.show:
        show_templates()
        return

    if not args.install_global and not args.project:
        print("\nNo installation target specified.")
        print("Use --global for global config or --project PATH for project-specific.")
        print("Use --show to preview templates.")
        parser.print_help()
        sys.exit(1)

    # Verify templates exist
    templates = get_templates()
    for name, path in templates.items():
        if not path.exists():
            print(f"[ERROR] Template not found: {path}")
            print("Run from the ai_orchestrator package directory.")
            sys.exit(1)

    if args.install_global:
        print("\nInstalling global configuration...")
        install_global(force=args.force, include_system=args.system)

    if args.project:
        print(f"\nInstalling project configuration to {args.project}...")
        install_project(args.project, force=args.force, include_system=args.system)

    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nGemini CLI will now provide more thorough, detailed responses.")
    print("\nVerify with: gemini -p 'Explain how to review code' ")
    print("Check loaded context: /memory show")


if __name__ == "__main__":
    main()
