"""Project context dataclass for discovered conventions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_orchestrator.project.greenfield import ProjectMaturity


class VerificationConfig(BaseModel):
    """Configuration for verification gates."""

    debug_script: str | None = None
    blast_radius: str | None = None
    static_analysis: list[str] = Field(default_factory=lambda: ["ruff check .", "mypy ."])
    unit_tests: str = "pytest tests/ -v"
    integration_tests: str | None = None
    security_scan: str | None = None


class ContextInjection(BaseModel):
    """Configuration for context injection into prompts."""

    include_files: list[str] = Field(default_factory=list)
    summarize_dirs: list[str] = Field(default_factory=list)


class ScoringWeights(BaseModel):
    """Weights for plan scoring criteria."""

    completeness: int = 20
    correctness: int = 25
    security: int = 20
    testability: int = 15
    maintainability: int = 15
    blast_radius: int = 10
    integration_friction: int = 10
    performance: int = 10


class ProjectConfig(BaseModel):
    """Explicit project configuration from .ai_orchestrator.yaml."""

    version: str = "1.0"
    name: str | None = None
    instructions: str | None = None
    patterns_dir: str | None = None
    registry: str | None = None
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    context_injection: ContextInjection = Field(default_factory=ContextInjection)
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)


class ProjectContext(BaseModel):
    """Discovered or configured project conventions."""

    # Root directory
    root: Path

    # Discovered paths (or None if not found)
    instructions_path: Path | None = None  # CLAUDE.md, agents.md, etc.
    debug_script_path: Path | None = None
    patterns_dir: Path | None = None
    registry_path: Path | None = None
    blast_radius_script: Path | None = None
    pre_commit_config: Path | None = None
    test_config_path: Path | None = None  # pytest.ini, pyproject.toml

    # Loaded content (optional, for caching)
    instructions_content: str | None = None
    registry_content: dict[str, Any] | None = None

    # Explicit configuration (if .ai_orchestrator.yaml exists)
    config: ProjectConfig = Field(default_factory=ProjectConfig)

    # Discovery metadata
    discovery_method: str = "auto"  # "auto" or "config"
    discovered_at: str | None = None

    # Greenfield detection
    maturity: str = "established"  # "greenfield", "nascent", "established"
    is_greenfield: bool = False
    needs_foundation_scaffold: bool = False

    class Config:
        arbitrary_types_allowed = True

    @property
    def has_instructions(self) -> bool:
        """Whether coding instructions were discovered."""
        return self.instructions_path is not None

    @property
    def has_debug_script(self) -> bool:
        """Whether a debug script was discovered."""
        return self.debug_script_path is not None

    @property
    def has_patterns(self) -> bool:
        """Whether a patterns library was discovered."""
        return self.patterns_dir is not None

    @property
    def has_registry(self) -> bool:
        """Whether a code registry was discovered."""
        return self.registry_path is not None

    @property
    def has_security_scan(self) -> bool:
        """Whether security scanning is configured."""
        return self.config.verification.security_scan is not None

    def get_verification_commands(self) -> dict[str, str | list[str] | None]:
        """Get all verification commands."""
        v = self.config.verification
        return {
            "debug_script": v.debug_script,
            "blast_radius": v.blast_radius,
            "static_analysis": v.static_analysis,
            "unit_tests": v.unit_tests,
            "integration_tests": v.integration_tests,
            "security_scan": v.security_scan,
        }

    def get_context_files(self) -> list[Path]:
        """Get list of files to include in prompts."""
        files = []

        # Always include instructions if available
        if self.instructions_path and self.instructions_path.exists():
            files.append(self.instructions_path)

        # Add configured files
        for rel_path in self.config.context_injection.include_files:
            path = self.root / rel_path
            if path.exists():
                files.append(path)

        return files

    def summary(self) -> str:
        """Generate a human-readable summary of discovered conventions."""
        lines = [f"Project: {self.root.name}"]
        lines.append(f"Discovery method: {self.discovery_method}")
        lines.append(f"Maturity: {self.maturity}")
        if self.needs_foundation_scaffold:
            lines.append("  [!] Needs foundation scaffolding")

        if self.instructions_path:
            lines.append(f"  instructions: {self.instructions_path.name}")
        if self.patterns_dir:
            lines.append(f"  patterns_dir: {self.patterns_dir.name}/")
        if self.registry_path:
            lines.append(f"  registry: {self.registry_path.name}")
        if self.debug_script_path:
            lines.append(f"  debug_script: {self.debug_script_path.name}")
        if self.blast_radius_script:
            lines.append(f"  blast_radius: {self.blast_radius_script.name}")
        if self.pre_commit_config:
            lines.append(f"  pre_commit: {self.pre_commit_config.name}")
        if self.test_config_path:
            lines.append(f"  tests: pytest (from {self.test_config_path.name})")

        return "\n".join(lines)
