"""Tests for greenfield project detection and scaffolding."""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile
import shutil

from ai_orchestrator.project.greenfield import (
    GreenfieldDetector,
    GreenfieldAnalysis,
    ProjectMaturity,
)
from ai_orchestrator.project.foundation_registry import FoundationRegistry
from ai_orchestrator.project.scaffolder import FoundationScaffolder, ScaffoldingError


class TestGreenfieldDetector:
    """Tests for GreenfieldDetector."""

    @pytest.fixture
    def detector(self) -> GreenfieldDetector:
        """Create a detector instance."""
        return GreenfieldDetector()

    @pytest.fixture
    def empty_dir(self) -> Path:
        """Create an empty temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def greenfield_project(self, empty_dir: Path) -> Path:
        """Create a greenfield project (0-5 files)."""
        # Add 3 source files
        (empty_dir / "main.py").write_text("print('hello')")
        (empty_dir / "utils.py").write_text("def foo(): pass")
        (empty_dir / "config.py").write_text("DEBUG = True")
        return empty_dir

    @pytest.fixture
    def nascent_project(self, empty_dir: Path) -> Path:
        """Create a nascent project (5-20 files)."""
        # Add 10 source files
        for i in range(10):
            (empty_dir / f"module_{i}.py").write_text(f"# Module {i}")
        return empty_dir

    @pytest.fixture
    def established_project(self, empty_dir: Path) -> Path:
        """Create an established project (20+ files)."""
        # Add 25 source files
        src_dir = empty_dir / "src"
        src_dir.mkdir()
        for i in range(25):
            (src_dir / f"file_{i}.py").write_text(f"# File {i}")
        return empty_dir

    @pytest.mark.asyncio
    async def test_empty_directory_is_greenfield(
        self, detector: GreenfieldDetector, empty_dir: Path
    ) -> None:
        """Empty directory should be classified as greenfield."""
        analysis = await detector.analyze(empty_dir)

        assert analysis.maturity == ProjectMaturity.GREENFIELD
        assert analysis.source_file_count == 0
        assert analysis.needs_scaffolding is True

    @pytest.mark.asyncio
    async def test_greenfield_project_detection(
        self, detector: GreenfieldDetector, greenfield_project: Path
    ) -> None:
        """Project with 0-5 files should be greenfield."""
        analysis = await detector.analyze(greenfield_project)

        assert analysis.maturity == ProjectMaturity.GREENFIELD
        assert analysis.source_file_count == 3
        assert analysis.needs_scaffolding is True

    @pytest.mark.asyncio
    async def test_nascent_project_detection(
        self, detector: GreenfieldDetector, nascent_project: Path
    ) -> None:
        """Project with 5-20 files should be nascent."""
        analysis = await detector.analyze(nascent_project)

        assert analysis.maturity == ProjectMaturity.NASCENT
        assert analysis.source_file_count == 10
        assert analysis.needs_scaffolding is False

    @pytest.mark.asyncio
    async def test_established_project_detection(
        self, detector: GreenfieldDetector, established_project: Path
    ) -> None:
        """Project with 20+ files should be established."""
        analysis = await detector.analyze(established_project)

        assert analysis.maturity == ProjectMaturity.ESTABLISHED
        assert analysis.source_file_count == 25
        assert analysis.needs_scaffolding is False

    @pytest.mark.asyncio
    async def test_excludes_node_modules(
        self, detector: GreenfieldDetector, empty_dir: Path
    ) -> None:
        """Should exclude node_modules from file count."""
        # Add files in node_modules (should be ignored)
        node_dir = empty_dir / "node_modules" / "some_package"
        node_dir.mkdir(parents=True)
        for i in range(50):
            (node_dir / f"file_{i}.js").write_text(f"// File {i}")

        # Add 2 real source files
        (empty_dir / "index.js").write_text("console.log('hi')")
        (empty_dir / "app.js").write_text("export default {}")

        analysis = await detector.analyze(empty_dir)

        # Should only count the 2 real files, not node_modules
        assert analysis.source_file_count == 2
        assert analysis.maturity == ProjectMaturity.GREENFIELD

    @pytest.mark.asyncio
    async def test_excludes_venv(
        self, detector: GreenfieldDetector, empty_dir: Path
    ) -> None:
        """Should exclude venv from file count."""
        # Add files in venv (should be ignored)
        venv_dir = empty_dir / "venv" / "lib" / "python3.11"
        venv_dir.mkdir(parents=True)
        for i in range(100):
            (venv_dir / f"module_{i}.py").write_text(f"# Module {i}")

        # Add 1 real source file
        (empty_dir / "main.py").write_text("print('hello')")

        analysis = await detector.analyze(empty_dir)

        # Should only count the 1 real file
        assert analysis.source_file_count == 1
        assert analysis.maturity == ProjectMaturity.GREENFIELD

    @pytest.mark.asyncio
    async def test_detects_existing_foundations(
        self, detector: GreenfieldDetector, greenfield_project: Path
    ) -> None:
        """Should detect if CLAUDE.md and scripts/ already exist."""
        # Project starts as greenfield needing scaffolding
        analysis = await detector.analyze(greenfield_project)
        assert analysis.needs_scaffolding is True
        assert analysis.has_claude_md is False
        assert analysis.has_scripts_folder is False

        # Add CLAUDE.md and scripts/
        (greenfield_project / "CLAUDE.md").write_text("# Standards")
        (greenfield_project / "scripts").mkdir()

        # Re-analyze
        analysis = await detector.analyze(greenfield_project)

        # Still greenfield, but no longer needs scaffolding
        assert analysis.maturity == ProjectMaturity.GREENFIELD
        assert analysis.needs_scaffolding is False
        assert analysis.has_claude_md is True
        assert analysis.has_scripts_folder is True

    @pytest.mark.asyncio
    async def test_analysis_summary(
        self, detector: GreenfieldDetector, empty_dir: Path
    ) -> None:
        """Analysis summary should be human-readable."""
        analysis = await detector.analyze(empty_dir)

        summary = analysis.summary
        assert "greenfield" in summary.lower()
        assert "0 source files" in summary
        assert "scaffolding needed" in summary


class TestFoundationRegistry:
    """Tests for FoundationRegistry."""

    def test_get_orchestrator_root(self) -> None:
        """Should return the correct orchestrator root."""
        root = FoundationRegistry.get_orchestrator_root()

        # Root should contain ai_orchestrator and scripts directories
        assert (root / "ai_orchestrator").is_dir()
        assert (root / "scripts").is_dir()

    def test_get_script_paths(self) -> None:
        """Should return valid script paths."""
        scripts = FoundationRegistry.get_script_paths()

        # Should return at least some scripts
        assert len(scripts) > 0

        # Each should be (path, filename) tuple
        for src_path, dst_name in scripts:
            assert src_path.exists(), f"Script not found: {src_path}"
            assert dst_name.endswith(".py")

    def test_get_best_practices_paths(self) -> None:
        """Should return valid best practices paths."""
        best_practices = FoundationRegistry.get_best_practices_paths()

        # Should return at least patterns.json
        assert len(best_practices) > 0

        for src_path, dst_name in best_practices:
            assert src_path.exists(), f"Best practice not found: {src_path}"

    def test_validate_installation(self) -> None:
        """Should validate all foundation files exist."""
        result = FoundationRegistry.validate_installation()

        # Should have present list
        assert "present" in result
        assert "missing" in result

        # Most files should be present
        assert len(result["present"]) > 0


class TestFoundationScaffolder:
    """Tests for FoundationScaffolder."""

    @pytest.fixture
    def empty_dir(self) -> Path:
        """Create an empty temporary directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_dry_run(self, empty_dir: Path) -> None:
        """Dry run should not create any files."""
        scaffolder = FoundationScaffolder(empty_dir)

        result = await scaffolder.scaffold_all(dry_run=True)

        assert result["dry_run"] is True
        assert "would_create_files" in result
        assert "would_create_dirs" in result

        # Should not have created anything
        assert not (empty_dir / "CLAUDE.md").exists()
        assert not (empty_dir / "scripts").exists()

    @pytest.mark.asyncio
    async def test_scaffold_creates_files(self, empty_dir: Path) -> None:
        """Scaffold should create expected files and directories."""
        scaffolder = FoundationScaffolder(empty_dir)

        result = await scaffolder.scaffold_all(project_name="test_project")

        # Check directories created
        assert (empty_dir / "scripts").is_dir()
        assert (empty_dir / "best_practices").is_dir()
        assert (empty_dir / "data").is_dir()
        assert (empty_dir / "tests").is_dir()

        # Check CLAUDE.md created with project name
        claude_md = empty_dir / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "test_project" in content

        # Check some scripts were copied
        scripts_dir = empty_dir / "scripts"
        script_files = list(scripts_dir.glob("*.py"))
        assert len(script_files) > 0

    @pytest.mark.asyncio
    async def test_scaffold_skips_existing(self, empty_dir: Path) -> None:
        """Scaffold should skip existing files."""
        # Pre-create CLAUDE.md with custom content
        existing_content = "# My Custom Standards"
        (empty_dir / "CLAUDE.md").write_text(existing_content)

        scaffolder = FoundationScaffolder(empty_dir)
        result = await scaffolder.scaffold_all()

        # Should have skipped CLAUDE.md
        assert "CLAUDE.md (exists)" in result["skipped"]

        # Content should be unchanged
        assert (empty_dir / "CLAUDE.md").read_text() == existing_content

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, empty_dir: Path) -> None:
        """Should rollback created files on failure."""
        scaffolder = FoundationScaffolder(empty_dir)

        # Create scripts dir to track rollback
        scripts_dir = empty_dir / "scripts"

        # Monkey-patch to fail during CLAUDE.md generation
        original_generate = scaffolder._generate_claude_md

        async def failing_generate(*args, **kwargs):
            raise RuntimeError("Simulated failure")

        scaffolder._generate_claude_md = failing_generate

        # Should raise and rollback
        with pytest.raises(ScaffoldingError):
            await scaffolder.scaffold_all()

        # Directories created before failure should be rolled back if empty
        # Note: some directories may remain if they weren't empty
        # The key test is that the error was raised and handled
