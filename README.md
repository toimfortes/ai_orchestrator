# AI Orchestrator

Multi-AI Code Orchestration System for automated code generation and review workflows.

## Overview

AI Orchestrator is a **project-agnostic** infrastructure for orchestrating multiple AI coding CLIs (Claude, Codex, Gemini, Kilocode) to generate, review, and implement code changes.

## Features

- **Multi-CLI Support**: Unified interface for Claude, Codex, Gemini, and Kilocode CLIs
- **Project Discovery**: Auto-discovers project conventions (CLAUDE.md, debug scripts, patterns)
- **Adaptive Iteration**: DDI-based convergence detection stops when reviews converge
- **Atomic State**: Crash-safe persistence with rolling backups
- **Graceful Degradation**: Human handoff when all CLIs fail

## Installation

```bash
# From source
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run on current directory
python -m ai_orchestrator -p "Add user authentication"

# Run on a specific project
python -m ai_orchestrator -p "Add logging" --project /path/to/project

# Discover project conventions
python -m ai_orchestrator discover --project /path/to/project

# Initialize config file
python -m ai_orchestrator init --project /path/to/project
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `run -p "prompt"` | Run orchestration workflow |
| `discover` | Show discovered project conventions |
| `init` | Create .ai_orchestrator.yaml config |

## Configuration

Create `.ai_orchestrator.yaml` in your project root:

```yaml
version: "1.0"

project:
  name: "My Project"
  instructions: "CLAUDE.md"
  patterns_dir: "best_practices/"

verification:
  debug_script: "scripts/debug.py --quick"
  static_analysis: ["ruff check .", "mypy ."]
  unit_tests: "pytest tests/ -v"
```

## Architecture

```
INIT → PLANNING → REVIEWING → FIXING → IMPLEMENTING → POST_CHECKS → COMPLETED
```

## Subscription Model

All primary CLIs use **subscription-based authentication** with usage limits:

| CLI | Subscription | Limit Type |
|-----|-------------|------------|
| Claude | Pro / Max | Token/message limits |
| Codex | OpenAI Plus/Pro | Usage limits |
| Gemini | Google One AI Premium | Usage limits |
| Kilocode | Free + OpenRouter | Free models unlimited |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type check
mypy ai_orchestrator

# Lint
ruff check .
```

## License

MIT
