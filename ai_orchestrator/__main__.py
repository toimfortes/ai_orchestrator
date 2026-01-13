"""Entry point for AI Orchestrator CLI."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from ai_orchestrator.config.settings import get_settings, load_settings_from_yaml
from ai_orchestrator.core.orchestrator import Orchestrator
from ai_orchestrator.project.discovery import discover_project
from ai_orchestrator.project.loader import create_default_config


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="ai-orchestrator",
        description="Multi-AI Code Orchestration System",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Main run command (default)
    run_parser = subparsers.add_parser("run", help="Run orchestration workflow")
    run_parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Task prompt for the AI",
    )
    run_parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project directory (default: current directory)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from saved state",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    run_parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Stop after planning phase",
    )
    run_parser.add_argument(
        "--iterations",
        type=int,
        help="Override max iterations",
    )
    run_parser.add_argument(
        "--planners",
        type=str,
        help="Comma-separated list of CLIs for planning (e.g., 'claude,codex,gemini')",
    )
    run_parser.add_argument(
        "--reviewers",
        type=str,
        help="Comma-separated list of CLIs for reviewing (e.g., 'claude,gemini')",
    )
    run_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent CLI invocations (default: 5)",
    )

    # Incremental review options
    run_parser.add_argument(
        "--incremental-review",
        action="store_true",
        help="Enable micro-reviews during implementation (catches issues early)",
    )
    run_parser.add_argument(
        "--review-agent",
        type=str,
        default="gemini",
        help="Agent for incremental reviews (default: gemini - uses generous free limits)",
    )
    run_parser.add_argument(
        "--review-granularity",
        type=str,
        choices=["file", "commit", "hunk", "time"],
        default="file",
        help="Granularity of incremental reviews (default: file)",
    )
    run_parser.add_argument(
        "--review-threshold",
        type=int,
        default=10,
        help="Minimum lines changed to trigger incremental review (default: 10)",
    )

    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug output",
    )

    # Discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Show discovered project conventions",
    )
    discover_parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project directory",
    )

    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize .ai_orchestrator.yaml config",
    )
    init_parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project directory",
    )

    # Also support running without subcommand (default to run)
    parser.add_argument(
        "-p", "--prompt",
        help="Task prompt (shortcut for 'run -p')",
    )
    parser.add_argument(
        "--project",
        type=Path,
        help="Project directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug output",
    )

    return parser.parse_args()


async def cmd_run(args: argparse.Namespace) -> int:
    """Run orchestration workflow."""
    project_path = args.project or Path.cwd()
    max_concurrent = getattr(args, "max_concurrent", 5)

    print(f"AI Orchestrator - Project: {project_path}")
    print("=" * 50)

    # Get planner/reviewer overrides
    planners = None
    if getattr(args, "planners", None):
        planners = [p.strip() for p in args.planners.split(",")]
        print(f"Planners: {', '.join(planners)}")

    reviewers = None
    if getattr(args, "reviewers", None):
        reviewers = [r.strip() for r in args.reviewers.split(",")]
        print(f"Reviewers: {', '.join(reviewers)}")

    print(f"Max concurrent: {max_concurrent}")

    # Get incremental review settings
    incremental_review = getattr(args, "incremental_review", False)
    review_agent = getattr(args, "review_agent", "gemini")
    review_granularity = getattr(args, "review_granularity", "file")
    review_threshold = getattr(args, "review_threshold", 10)

    if incremental_review:
        print(f"Incremental review: enabled ({review_agent} at {review_granularity} level)")

    orchestrator = Orchestrator(
        project_path,
        max_concurrent_clis=max_concurrent,
        incremental_review=incremental_review,
        review_agent=review_agent,
        review_granularity=review_granularity,
        review_threshold=review_threshold,
    )

    # Override fallback chains if specified
    if planners:
        orchestrator.FALLBACK_CHAINS["planning"] = planners
    if reviewers:
        orchestrator.FALLBACK_CHAINS["reviewing"] = reviewers

    try:
        state = await orchestrator.run(
            prompt=args.prompt,
            resume=getattr(args, "resume", False),
            dry_run=args.dry_run,
            plan_only=getattr(args, "plan_only", False),
            max_iterations=getattr(args, "iterations", None),
        )

        print(f"\nWorkflow completed: {state.current_phase.value}")
        print(f"Iterations: {state.current_iteration}")
        print(f"Plans generated: {len(state.plans)}")

        if state.errors:
            print(f"\nErrors ({len(state.errors)}):")
            for error in state.errors[-5:]:
                print(f"  - {error}")

        return 0 if state.current_phase.value == "completed" else 1

    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


async def cmd_discover(args: argparse.Namespace) -> int:
    """Discover and display project conventions."""
    project_path = args.project or Path.cwd()

    context = await discover_project(project_path)

    print(f"\nDiscovered project conventions in {project_path}:")
    print("-" * 50)
    print(context.summary())

    return 0


async def cmd_init(args: argparse.Namespace) -> int:
    """Initialize .ai_orchestrator.yaml config."""
    project_path = args.project or Path.cwd()

    config_path = create_default_config(project_path)

    print(f"Created config file: {config_path}")
    print("\nEdit this file to customize orchestrator behavior.")

    return 0


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    verbose = getattr(args, "verbose", False)
    debug = getattr(args, "debug", False)
    setup_logging(verbose=verbose, debug=debug)

    # Handle default command (no subcommand)
    if args.command is None:
        if args.prompt:
            # Run with prompt
            args.command = "run"
            if args.project is None:
                args.project = Path.cwd()
        else:
            # No prompt provided, show help
            print("Usage: python -m ai_orchestrator -p 'Your task prompt'")
            print("       python -m ai_orchestrator discover --project /path/to/project")
            print("       python -m ai_orchestrator init --project /path/to/project")
            return 0

    # Dispatch to command handler
    if args.command == "run":
        return await cmd_run(args)
    elif args.command == "discover":
        return await cmd_discover(args)
    elif args.command == "init":
        return await cmd_init(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def cli_main() -> None:
    """CLI entry point (synchronous wrapper)."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli_main()
