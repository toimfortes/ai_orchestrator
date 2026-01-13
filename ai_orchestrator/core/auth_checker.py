"""CLI authentication status checker."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from ai_orchestrator.cli_adapters.base import CLIAdapter

logger = logging.getLogger(__name__)


@dataclass
class AuthResult:
    """Authentication check result for a single CLI."""

    authenticated: bool
    config_exists: bool
    cli_responsive: bool
    auth_command: str
    error_message: str | None = None


@dataclass
class AuthStatus:
    """Authentication status for all CLIs."""

    results: dict[str, AuthResult] = field(default_factory=dict)

    @property
    def all_authenticated(self) -> bool:
        """Whether all CLIs are authenticated."""
        if not self.results:
            return False
        return all(r.authenticated for r in self.results.values())

    @property
    def any_authenticated(self) -> bool:
        """Whether at least one CLI is authenticated."""
        return any(r.authenticated for r in self.results.values())

    @property
    def unauthenticated(self) -> list[str]:
        """List of unauthenticated CLI names."""
        return [name for name, r in self.results.items() if not r.authenticated]


class AuthChecker:
    """
    Verify CLI authentication status before workflow.

    All CLIs use browser-based OAuth - no API keys needed.
    """

    CLI_AUTH_CHECKS: dict[str, dict[str, str | Path]] = {
        "claude": {
            "check_cmd": "claude --version",
            "auth_cmd": "claude login",
            "config_path": Path.home() / ".claude" / "config.json",
        },
        "codex": {
            "check_cmd": "codex --version",
            "auth_cmd": "codex login",
            "config_path": Path.home() / ".codex" / "config.json",
        },
        "gemini": {
            "check_cmd": "gemini --version",
            "auth_cmd": "gemini auth",
            "config_path": Path.home() / ".gemini" / "config.json",
        },
        "kilocode": {
            "check_cmd": "kilocode --version",
            "auth_cmd": "kilocode auth",
            "config_path": Path.home() / ".kilocode" / "secrets.json",
        },
    }

    async def check_all_auth(self, enabled_clis: list[str]) -> AuthStatus:
        """
        Check authentication for all enabled CLIs.

        Args:
            enabled_clis: List of CLI names to check.

        Returns:
            AuthStatus with results for each CLI.
        """
        results: dict[str, AuthResult] = {}

        for cli in enabled_clis:
            if cli not in self.CLI_AUTH_CHECKS:
                logger.warning("Unknown CLI for auth check: %s", cli)
                continue

            check = self.CLI_AUTH_CHECKS[cli]
            result = await self._check_single_cli(cli, check)
            results[cli] = result

        return AuthStatus(results=results)

    async def _check_single_cli(
        self,
        cli_name: str,
        check_config: dict[str, str | Path],
    ) -> AuthResult:
        """Check authentication for a single CLI."""
        config_path = Path(check_config["config_path"])
        auth_cmd = str(check_config["auth_cmd"])

        # Method 1: Check if config file exists
        config_exists = config_path.exists()

        # Method 2: Try running CLI
        cli_responsive = False
        error_message = None

        try:
            check_cmd = str(check_config["check_cmd"]).split()
            proc = await asyncio.create_subprocess_exec(
                *check_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            cli_responsive = proc.returncode == 0

            if not cli_responsive:
                error_message = stderr.decode("utf-8", errors="replace").strip()

        except asyncio.TimeoutError:
            error_message = "CLI check timed out"
        except FileNotFoundError:
            error_message = "CLI not installed"
        except Exception as e:
            error_message = str(e)

        # Authenticated if both config exists and CLI works
        authenticated = config_exists and cli_responsive

        return AuthResult(
            authenticated=authenticated,
            config_exists=config_exists,
            cli_responsive=cli_responsive,
            auth_command=auth_cmd,
            error_message=error_message,
        )

    def print_auth_status(self, status: AuthStatus) -> None:
        """Print human-readable auth status to console."""
        print("\n┌─────────────────────────────────────┐")
        print("│  CLI Authentication Status          │")
        print("├─────────────┬───────────────────────┤")

        for cli, result in status.results.items():
            icon = "+" if result.authenticated else "X"
            status_text = "Ready" if result.authenticated else "Not authenticated"
            print(f"│ {cli:<11} │ [{icon}] {status_text:<17} │")

        print("└─────────────┴───────────────────────┘")

        # Show auth commands for unauthenticated CLIs
        unauthenticated = [
            (cli, r) for cli, r in status.results.items() if not r.authenticated
        ]

        if unauthenticated:
            print("\n[!] Run these commands to authenticate:")
            for cli, result in unauthenticated:
                print(f"    $ {result.auth_command}")
                if result.error_message:
                    print(f"      ({result.error_message})")


async def check_cli_auth(cli_name: str) -> bool:
    """
    Quick check if a single CLI is authenticated.

    Args:
        cli_name: Name of the CLI to check.

    Returns:
        True if authenticated, False otherwise.
    """
    checker = AuthChecker()
    status = await checker.check_all_auth([cli_name])
    return cli_name in status.results and status.results[cli_name].authenticated
