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

    # Research command
    research_parser = subparsers.add_parser(
        "research",
        help="Run deep research on code with AI providers",
    )
    research_parser.add_argument(
        "question",
        nargs="?",
        help="Research question (or use --template for predefined questions)",
    )
    research_parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project directory",
    )
    research_parser.add_argument(
        "--files",
        type=str,
        help="Comma-separated list of files to analyze (default: auto-detect key files)",
    )
    research_parser.add_argument(
        "--focus",
        type=str,
        choices=["general", "security", "performance", "architecture", "reliability"],
        default="general",
        help="Focus area for the research (default: general)",
    )
    research_parser.add_argument(
        "--template",
        type=str,
        choices=["architecture_review", "security_review", "reliability_review", "performance_review"],
        help="Use a predefined question template",
    )

    # Tier selection
    tier_group = research_parser.add_mutually_exclusive_group()
    tier_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick research: single provider, fast (~2-5 min)",
    )
    tier_group.add_argument(
        "--standard",
        action="store_true",
        default=True,
        help="Standard research: 2 providers with comparison (~8-12 min) [default]",
    )
    tier_group.add_argument(
        "--council",
        action="store_true",
        help="Council research: 4 providers with full consensus (~20-30 min)",
    )

    research_parser.add_argument(
        "--providers",
        type=str,
        help="Override providers (comma-separated, e.g., 'gemini,openai')",
    )
    research_parser.add_argument(
        "--output",
        type=Path,
        help="Save result to file (JSON format)",
    )
    research_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    research_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug output",
    )

    # Dashboard command
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start the web-based control dashboard",
    )
    dashboard_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    dashboard_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
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


async def cmd_research(args: argparse.Namespace) -> int:
    """Run deep research on code."""
    from ai_orchestrator.research import (
        StandardResearch,
        ResearchFocus,
        quick_research,
        format_code_context,
        get_question_template,
    )
    import json

    project_path = args.project or Path.cwd()

    # Get question
    question = args.question
    if args.template:
        question = get_question_template(args.template)
    if not question:
        print("Error: Must provide a question or --template")
        return 1

    # Get focus
    focus = ResearchFocus(args.focus)

    # Determine tier
    if args.quick:
        tier = "quick"
    elif args.council:
        tier = "council"
        print("Warning: Council tier not yet implemented, falling back to standard")
        tier = "standard"
    else:
        tier = "standard"

    # Get files to analyze
    if args.files:
        file_paths = [project_path / f.strip() for f in args.files.split(",")]
    else:
        # Auto-detect key files (look for orchestrator, state_manager, etc.)
        file_paths = _discover_key_files(project_path)

    if not file_paths:
        print("Error: No files found to analyze")
        return 1

    # Load file contents
    files: dict[str, str] = {}
    for path in file_paths:
        if path.exists():
            try:
                files[str(path.relative_to(project_path))] = path.read_text(encoding="utf-8")
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")

    if not files:
        print("Error: Could not read any files")
        return 1

    # Format code context
    code_context = format_code_context(files)

    print(f"\nAI Orchestrator Research")
    print("=" * 60)
    print(f"Tier: {tier}")
    print(f"Focus: {focus.value}")
    print(f"Files: {len(files)}")
    print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
    print("=" * 60)

    try:
        if tier == "quick":
            # Quick: single provider
            provider = "gemini"
            if args.providers:
                provider = args.providers.split(",")[0].strip()

            print(f"\nRunning quick research with {provider}...")
            result = await quick_research(code_context, question, focus, provider)

            print(f"\nResult: {'Success' if result.success else 'Failed'}")
            print(f"Duration: {result.duration_seconds:.1f}s")
            print(f"Findings: {len(result.findings)}")

            if result.success:
                print("\n" + "-" * 60)
                print(result.raw_output)
            else:
                print(f"\nError: {result.error}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\nSaved to: {args.output}")

        else:
            # Standard: 2 providers
            providers = None
            if args.providers:
                providers = [p.strip() for p in args.providers.split(",")]

            print("\nRunning standard research (2 providers)...")
            research = StandardResearch(providers=providers)
            result = await research.run(code_context, question, focus)

            # Print formatted report
            print("\n" + result.format_report())

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                print(f"\nSaved to: {args.output}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


def _discover_key_files(project_path: Path) -> list[Path]:
    """Discover key files in the project for analysis."""
    key_patterns = [
        "core/orchestrator.py",
        "core/state_manager.py",
        "core/workflow_phases.py",
        "cli_adapters/base.py",
        "reviewing/feedback_classifier.py",
        "utils/json_parser.py",
    ]

    files = []
    for pattern in key_patterns:
        path = project_path / pattern
        if path.exists():
            files.append(path)

    # Also look for main module files
    for py_file in project_path.glob("*.py"):
        if py_file.name not in ("__init__.py", "__main__.py", "conftest.py"):
            files.append(py_file)

    return files[:10]  # Limit to 10 files


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Start the web dashboard server."""
    try:
        import uvicorn
        from ai_orchestrator.dashboard.server import app
    except ImportError:
        print("Dashboard dependencies not installed.")
        print("Install with: pip install 'ai-orchestrator[dashboard]'")
        print("Or: pip install fastapi uvicorn websockets")
        return 1

    host = args.host
    port = args.port
    open_browser = not args.no_browser

    print(f"\n{'=' * 60}")
    print("  AI Orchestrator Control Dashboard")
    print(f"{'=' * 60}")
    print(f"\n  Starting server at http://{host}:{port}")
    print("  Press Ctrl+C to stop\n")

    # Open browser if requested
    if open_browser:
        import webbrowser
        import threading

        def open_browser_delayed():
            import time
            time.sleep(1.5)  # Wait for server to start
            url = f"http://{'localhost' if host == '0.0.0.0' else host}:{port}"
            webbrowser.open(url)

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
        return 0
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
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
            print("       python -m ai_orchestrator research 'Review circuit breaker' --standard")
            print("       python -m ai_orchestrator research --template security_review --focus security")
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
    elif args.command == "research":
        return await cmd_research(args)
    elif args.command == "dashboard":
        return cmd_dashboard(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


def cli_main() -> None:
    """CLI entry point (synchronous wrapper)."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    cli_main()
