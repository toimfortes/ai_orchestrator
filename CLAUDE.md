# AI Coding Standards - Multi-AI Code Orchestrator

## Purpose
- Governs code generation and engineering conduct for the AI Orchestrator project
- Enforces: SOLID principles, composition + DI, modularity, tests, observability, and configuration management
- Audience: AI coding assistants and developers

## Non-Negotiables (Architecture & Safety)

### Configuration Management
- **No hardcoded values** outside of `settings.py` and `.env` files
- All model IDs, API endpoints, timeouts, and feature flags must be configurable
- Use Pydantic settings with environment variable support
- Validate configuration at startup (fail fast)

### Code Organization
- SOLID by default; composition over inheritance; one responsibility per module
- Ports/adapters pattern for external dependencies
- Dependency injection at composition root; avoid global singletons
- Explicit interfaces; typed errors with causes and remediation hints

### Logging & Observability
- Structured logging with context + exc_info for errors
- Never log errors without traceback
- Basic metrics where applicable (counters, latency, error rates)

### Error Handling
- Never return mock/fallback data to hide failures; fail visibly
- Return real errors with context; let callers handle gracefully
- Circuit breakers for external service calls

## Quick Rules

### Never
- Hardcoded model IDs, API keys, or magic constants in code files
- Tests in root or module root directories
- Log errors without traceback
- Commit secrets or missing .env.example entries
- Block event loops with sync calls in async code
- Mock to hide errors in production paths
- Duplicate model/class definitions

### Always
- Check settings.py before adding new configuration
- Use structured logging with context
- Type hints on all public functions
- Pydantic validation for inputs
- Circuit breakers for external API calls
- Document new environment variables in .env.example

## Project Structure

```
ai_orchestrator/
├── config/           # Settings and configuration (settings.py)
├── core/             # Core orchestration logic
├── cli_adapters/     # Adapters for AI CLIs (Claude, Gemini, etc.)
├── dashboard/        # Web dashboard (FastAPI)
├── research/         # Deep research capabilities
├── project/          # Project discovery and loading
├── reviewing/        # Review and feedback logic
└── utils/            # Shared utilities
```

## Configuration Locations

| Type | Location | Example |
|------|----------|---------|
| Model IDs | `config/settings.py` | `AvailableModels` class |
| Timeouts | `config/settings.py` | `CLITimeouts` class |
| Feature flags | `config/settings.py` | Settings attributes |
| Secrets | `.env` file | `AI_ORCHESTRATOR_*` prefix |
| Dashboard state | `~/.ai_orchestrator/` | JSON configs |

## Adding New Models

1. Add to `AvailableModels` in `config/settings.py`:
```python
class AvailableModels(BaseModel):
    claude: ProviderModels = Field(default_factory=lambda: ProviderModels(models=[
        ModelInfo(id="new-model-id", name="Display Name", tier="standard"),
    ]))
```

2. Models are automatically available in the dashboard API

## Decision Gate (approval required)

Before implementing, get approval for:
- New frameworks, datastores, or external services
- Public API shape changes
- New environment variables
- Data model or migration changes
- Performance or cost-impacting patterns

## Output Contract

Every deliverable should include:
- **S** (Summary): What was done
- **D** (Decisions): Key choices made
- **R** (Risks): Potential issues and mitigations
- **N** (Next steps): Follow-up actions with commands/files
- **T** (Tests): How to verify the changes

## Measure Twice, Cut Once Workflow

Before implementing changes, use these tools to understand the codebase architecture and impact:

### 1. Build/Refresh Code Catalog

```bash
python scripts/build_code_registry.py
```

Generates `data/code_catalog.json` with:
- All modules, classes, functions
- Import graph (what depends on what)
- Module categorization
- Statistics (lines of code, etc.)

### 2. Measure Blast Radius

Before modifying a file, check what will be affected:

```bash
# Check impact of modifying a specific file
python scripts/measure_blast_radius.py ai_orchestrator/core/orchestrator.py

# See all high-impact modules
python scripts/measure_blast_radius.py --all-critical
```

Risk levels:
- `[!!!] CRITICAL`: >30% of codebase affected
- `[!!] HIGH`: >15% affected
- `[!] MEDIUM`: >5% affected
- `[.] LOW`: Some dependents
- `[-] MINIMAL`: No dependents

### 3. Check Patterns Before Implementation

Validate code against best practices:

```bash
# Check a specific file
python scripts/check_patterns.py ai_orchestrator/core/orchestrator.py

# Check all files
python scripts/check_patterns.py --all

# Strict mode (fails on errors)
python scripts/check_patterns.py --all --strict
```

### Implementation Workflow

1. **Before coding**: Run blast radius check on files you'll modify
2. **During coding**: Follow patterns in `best_practices/patterns.json`
3. **After coding**: Run pattern checker on modified files
4. **Before commit**: Regenerate code catalog

### Key Files (Pre-Implementation)

| Tool | Location | Purpose |
|------|----------|---------|
| Code Catalog | `scripts/build_code_registry.py` | Generate codebase map |
| Blast Radius | `scripts/measure_blast_radius.py` | Impact analysis |
| Pattern Checker | `scripts/check_patterns.py` | Validate against best practices |
| Patterns | `best_practices/patterns.json` | Pattern definitions |
| Catalog Output | `data/code_catalog.json` | Machine-readable codebase map |

## Post-Implementation Checks (Cut Once)

After implementing changes, run these validation checks before committing:

### 1. Full Post-Implementation Check

```bash
# Basic check on modified files
python scripts/post_implementation_check.py

# Full check with tests
python scripts/post_implementation_check.py --run-tests

# Auto-fix issues (regenerate catalog)
python scripts/post_implementation_check.py --fix

# Strict mode (fail on warnings too)
python scripts/post_implementation_check.py --strict
```

This runs:
- Pattern violations check
- Code catalog freshness
- Import/syntax validation
- Type checking (if mypy installed)
- Test coverage check
- Security scan
- Docstring check

### 2. Diff-Based Pattern Check

For faster feedback, check only changed lines:

```bash
# Check staged changes (pre-commit)
python scripts/check_diff_patterns.py

# Check unstaged changes
python scripts/check_diff_patterns.py --unstaged

# Check all changes
python scripts/check_diff_patterns.py --all

# Check specific commit
python scripts/check_diff_patterns.py --commit HEAD~1
```

### 3. Implementation Verification

Verify implementation completeness:

```bash
# Basic verification
python scripts/verify_implementation.py

# With test run
python scripts/verify_implementation.py --run-tests

# With build check
python scripts/verify_implementation.py --run-build

# Use custom checklist
python scripts/verify_implementation.py --checklist my_checklist.json
```

### Key Files (Post-Implementation)

| Tool | Location | Purpose |
|------|----------|---------|
| Post-Check | `scripts/post_implementation_check.py` | Full validation suite |
| Diff Checker | `scripts/check_diff_patterns.py` | Check only changed lines |
| Verifier | `scripts/verify_implementation.py` | Completeness verification |

### Complete Workflow

```
1. MEASURE (Before)
   └── Build catalog → Check blast radius → Review patterns

2. CUT (During)
   └── Follow patterns → Keep changes minimal → Update tests

3. VERIFY (After)
   └── Run diff check → Post-implementation check → Verify implementation

4. COMMIT (Final)
   └── Regenerate catalog → Final verification → Commit
```

### Pre-Commit Hook (Optional)

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
python scripts/check_diff_patterns.py --strict
if [ $? -ne 0 ]; then
    echo "Pattern violations found. Fix before committing."
    exit 1
fi
python scripts/post_implementation_check.py
```
