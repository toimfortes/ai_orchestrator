# Debug Everything Process

This document describes the comprehensive debugging process for the AI Orchestrator codebase.

## Quick Start

```bash
# Run all debugging checks
python scripts/post_implementation_check.py --all

# Quick diff-based check (fastest)
python scripts/check_diff_patterns.py --all
```

## Debugging Workflow

### 1. Pre-Implementation Analysis

Before making changes, understand the impact:

```bash
# Build fresh code catalog
python scripts/build_code_registry.py

# Check blast radius of files you'll modify
python scripts/measure_blast_radius.py <file_path>

# Example: Check impact of modifying orchestrator
python scripts/measure_blast_radius.py ai_orchestrator/core/orchestrator.py

# See all high-impact modules (modify with care)
python scripts/measure_blast_radius.py --all-critical
```

### 2. During Development

Check patterns as you code:

```bash
# Check single file
python scripts/check_patterns.py ai_orchestrator/path/to/file.py

# Check only your changes (fastest feedback)
python scripts/check_diff_patterns.py --unstaged
```

### 3. Pre-Commit Validation

Before committing, run full validation:

```bash
# Full check on all modified files
python scripts/post_implementation_check.py

# Auto-fix issues (regenerates catalog)
python scripts/post_implementation_check.py --fix

# Strict mode (fails on warnings too)
python scripts/post_implementation_check.py --strict

# Check staged changes specifically
python scripts/check_diff_patterns.py
```

### 4. Post-Implementation Verification

Verify implementation completeness:

```bash
# Basic verification
python scripts/verify_implementation.py

# With test run
python scripts/verify_implementation.py --run-tests

# With build check
python scripts/verify_implementation.py --run-build
```

## Tool Reference

### build_code_registry.py

Generates `data/code_catalog.json` containing:
- All modules, classes, functions
- Import graph (dependencies)
- Module statistics

```bash
python scripts/build_code_registry.py           # Regenerate catalog
python scripts/build_code_registry.py --check   # Check freshness
python scripts/build_code_registry.py --summary # Print summary
```

### measure_blast_radius.py

Analyzes impact of modifying files:

```bash
python scripts/measure_blast_radius.py <file>         # Analyze single file
python scripts/measure_blast_radius.py --all-critical # Show high-impact modules
python scripts/measure_blast_radius.py <file> --json  # JSON output
```

Risk levels:
- `[!!!] CRITICAL`: >30% of codebase affected
- `[!!] HIGH`: >15% affected
- `[!] MEDIUM`: >5% affected
- `[.] LOW`: Some dependents
- `[-] MINIMAL`: No dependents

### check_patterns.py

Validates code against best practices:

```bash
python scripts/check_patterns.py <file>       # Check single file
python scripts/check_patterns.py --all        # Check all files
python scripts/check_patterns.py --strict     # Fail on any violation
python scripts/check_patterns.py --json       # JSON output
```

Pattern rules:
- `CFG001`: Hardcoded model IDs
- `CFG002`: Hardcoded timeouts (>30s)
- `ERR001`: Bare except with pass
- `ASYNC001`: Blocking calls in async
- `LOG001`: logger.error() without exc_info
- `PYD001`: Mutable default values
- `TEST001`: Tests in wrong directory
- `TEST002`: Simulated/dummy outputs

### check_diff_patterns.py

Fast check on only changed lines:

```bash
python scripts/check_diff_patterns.py              # Check staged
python scripts/check_diff_patterns.py --unstaged   # Check unstaged
python scripts/check_diff_patterns.py --all        # Check all changes
python scripts/check_diff_patterns.py --commit X   # Check specific commit
```

### post_implementation_check.py

Comprehensive validation suite:

```bash
python scripts/post_implementation_check.py         # Basic check
python scripts/post_implementation_check.py --all   # Check all files
python scripts/post_implementation_check.py --fix   # Auto-fix issues
python scripts/post_implementation_check.py --json  # JSON output
```

Runs these checks:
1. Pattern violations
2. Catalog freshness
3. Import/syntax validation
4. Type checking (mypy)
5. Test coverage check
6. Security scan
7. Docstring check

### verify_implementation.py

Verify implementation is complete:

```bash
python scripts/verify_implementation.py             # Basic verification
python scripts/verify_implementation.py --run-tests # Include tests
python scripts/verify_implementation.py --run-build # Include build check
python scripts/verify_implementation.py --checklist FILE  # Custom checklist
```

## Common Issues and Fixes

### CFG001: Hardcoded Model ID

**Bad:**
```python
model = "claude-3-opus"
```

**Good:**
```python
from ai_orchestrator.config.settings import get_settings
model = get_settings().models.claude.default
```

### PYD001: Mutable Default

**Bad:**
```python
class Config(BaseModel):
    items: list[str] = []
```

**Good:**
```python
class Config(BaseModel):
    items: list[str] = Field(default_factory=list)
```

### ERR001: Bare Except

**Bad:**
```python
try:
    risky_call()
except:
    pass
```

**Good:**
```python
try:
    risky_call()
except Exception as e:
    logger.error("Failed", exc_info=True)
    raise
```

### LOG001: Missing exc_info

**Bad:**
```python
logger.error(f"Failed: {e}")
```

**Good:**
```python
logger.error("Operation failed", exc_info=True)
```

## CI/CD Integration

### Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
set -e

echo "Running diff pattern check..."
python scripts/check_diff_patterns.py --strict

echo "Running post-implementation check..."
python scripts/post_implementation_check.py

echo "All checks passed!"
```

### GitHub Actions

```yaml
- name: Debug Check
  run: |
    python scripts/post_implementation_check.py --all --strict
```

## Debugging Specific Issues

### Import Errors

```bash
# Check if module imports correctly
python -c "from ai_orchestrator.module import Class"

# Run regression check
python scripts/verify_implementation.py
```

### Type Errors

```bash
# Run mypy on specific file
python -m mypy ai_orchestrator/path/to/file.py

# Run full type check
python scripts/post_implementation_check.py
```

### Finding Affected Code

```bash
# Find all files that import a module
python scripts/measure_blast_radius.py ai_orchestrator/utils/json_parser.py

# Generate fresh import graph
python scripts/build_code_registry.py
```

## File Locations

| File | Purpose |
|------|---------|
| `scripts/build_code_registry.py` | Generate code catalog |
| `scripts/measure_blast_radius.py` | Impact analysis |
| `scripts/check_patterns.py` | Pattern validation |
| `scripts/check_diff_patterns.py` | Diff-based validation |
| `scripts/post_implementation_check.py` | Full validation suite |
| `scripts/verify_implementation.py` | Implementation verification |
| `best_practices/patterns.json` | Pattern definitions |
| `data/code_catalog.json` | Generated code catalog |
