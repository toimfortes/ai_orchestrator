"""Registry of foundation files to scaffold into new projects.

Uses LIVE source files from this repository to avoid template drift.
"""

from __future__ import annotations

from pathlib import Path


class FoundationRegistry:
    """
    Registry of foundation files to scaffold into new projects.

    These are copied from the LIVE source in this repository, not from
    static templates, to ensure they stay up-to-date.
    """

    # Workflow scripts to copy to new projects
    SCRIPTS = [
        "scripts/build_code_registry.py",
        "scripts/check_patterns.py",
        "scripts/check_diff_patterns.py",
        "scripts/measure_blast_radius.py",
        "scripts/post_implementation_check.py",
        "scripts/verify_implementation.py",
        "scripts/debug_everything.py",
    ]

    # Best practices files to copy
    BEST_PRACTICES = [
        "best_practices/patterns.json",
    ]

    # Template files that need parameterization
    TEMPLATES = {
        "CLAUDE.md": "ai_orchestrator/templates/CLAUDE.md.template",
    }

    # Directories to create in new projects
    DIRECTORIES = [
        "scripts",
        "best_practices",
        "data",
        "tests",
    ]

    @classmethod
    def get_orchestrator_root(cls) -> Path:
        """
        Get the root directory of the AI orchestrator installation.

        Returns:
            Path to the orchestrator root (contains scripts/, ai_orchestrator/, etc.)
        """
        # This file is at ai_orchestrator/project/foundation_registry.py
        # So root is 3 levels up
        return Path(__file__).parent.parent.parent.resolve()

    @classmethod
    def get_script_paths(cls) -> list[tuple[Path, str]]:
        """
        Get list of (source_path, destination_filename) for scripts.

        Returns:
            List of tuples with source path and destination filename.
        """
        root = cls.get_orchestrator_root()
        result = []

        for script_rel in cls.SCRIPTS:
            src = root / script_rel
            dst_name = Path(script_rel).name
            if src.exists():
                result.append((src, dst_name))

        return result

    @classmethod
    def get_best_practices_paths(cls) -> list[tuple[Path, str]]:
        """
        Get list of (source_path, destination_filename) for best practices.

        Returns:
            List of tuples with source path and destination filename.
        """
        root = cls.get_orchestrator_root()
        result = []

        for bp_rel in cls.BEST_PRACTICES:
            src = root / bp_rel
            dst_name = Path(bp_rel).name
            if src.exists():
                result.append((src, dst_name))

        return result

    @classmethod
    def get_template_path(cls, template_name: str) -> Path | None:
        """
        Get path to a template file.

        Args:
            template_name: Name of the template (e.g., "CLAUDE.md").

        Returns:
            Path to template file, or None if not found.
        """
        if template_name not in cls.TEMPLATES:
            return None

        root = cls.get_orchestrator_root()
        template_rel = cls.TEMPLATES[template_name]
        template_path = root / template_rel

        return template_path if template_path.exists() else None

    @classmethod
    def validate_installation(cls) -> dict[str, list[str]]:
        """
        Validate that all foundation files exist in the installation.

        Returns:
            Dict with 'missing' and 'present' lists of files.
        """
        root = cls.get_orchestrator_root()
        missing = []
        present = []

        for script in cls.SCRIPTS:
            path = root / script
            if path.exists():
                present.append(script)
            else:
                missing.append(script)

        for bp in cls.BEST_PRACTICES:
            path = root / bp
            if path.exists():
                present.append(bp)
            else:
                missing.append(bp)

        for name, template_rel in cls.TEMPLATES.items():
            path = root / template_rel
            if path.exists():
                present.append(template_rel)
            else:
                missing.append(template_rel)

        return {"missing": missing, "present": present}
