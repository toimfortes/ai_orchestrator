# CLI Prompting Comparison Guide

> Quick reference for prompting differences across Claude, Codex, Gemini, and Kilocode CLIs.

## Summary Matrix

| Aspect | Claude | Codex | Gemini | Kilocode |
|--------|--------|-------|--------|----------|
| Default Verbosity | Thorough | Moderate | Concise | Varies by model |
| Implicit Reasoning | Excellent | Good | Needs explicit | Varies by model |
| Vague Prompts | Handles well | Handles OK | Takes literally | Varies by model |
| Session Memory | Yes (--resume) | Yes (resume) | Via GEMINI.md | No |
| Persistent Context | CLAUDE.md | None | GEMINI.md hierarchy | None |
| Planning Mode | Built-in | Via prompt | Needs explicit depth | Via prompt |

## Claude Code CLI

**Strengths**: Best at implicit reasoning, handles vague prompts well, thorough by default.

**Prompting Style**: Natural language works great.

```
# This works well - Claude understands implicit intent
"Do a deep critique of this implementation plan"

# Claude infers you want security, performance, maintainability, etc.
"Review this code"
```

**Context File**: `CLAUDE.md` (or `.claude/instructions.md`)
- Automatically loaded from project root
- Supports hierarchical loading

**Session Management**:
```bash
claude -p "prompt"           # Single prompt
claude --resume SESSION_ID   # Continue session
```

## OpenAI Codex CLI

**Strengths**: Strong code generation, good at following specifications.

**Prompting Style**: Be specific about requirements.

```
# Good - explicit about what you want
"Generate a Python function that validates email addresses
using regex. Include type hints and docstrings."

# Less effective - too vague
"Write an email validator"
```

**Context File**: None built-in (use prompt prefix)

**Session Management**:
```bash
codex exec "prompt"              # Single execution
codex exec resume SESSION_ID     # Continue session
codex exec resume --last         # Resume last session
```

**Key Flags**:
- `--json`: JSON output
- `--full-auto`: No confirmations
- `--output-schema schema.json`: Structured output

## Google Gemini CLI

**Strengths**: Fast, good at structured tasks, excellent multimodal support.

**Weaknesses**: Less verbose by default, needs explicit depth instructions.

**Prompting Style**: Be explicit about depth and reasoning.

```
# Won't work well - too vague
"Do a deep critique"

# Works better - explicit instructions
"Before responding, break down the problem into sub-tasks.
For each sub-task, provide detailed analysis.
After your response, self-critique: did you answer the intent?

Now critique this implementation plan..."
```

**Context Files**: `GEMINI.md` (hierarchical)
- `~/.gemini/GEMINI.md` - Global
- `<project>/GEMINI.md` - Project-level
- `<project>/subdir/GEMINI.md` - Directory-specific

**System Override**:
```bash
export GEMINI_SYSTEM_MD=true  # Use .gemini/system.md
```

**Key Flags**:
- `-p "prompt"`: Non-interactive mode
- `--output-format json`: JSON output
- `--yolo`: Auto-approve actions
- `-m model`: Model selection

## Kilocode CLI (OpenRouter)

**Strengths**: Access to many models via OpenRouter, free tier available.

**Note**: Behavior varies significantly by underlying model.

**Prompting Style**: Depends on model - follow that model's best practices.

```bash
# Uses whatever model is configured
kilocode --auto --json "Your prompt"
```

**Model-Specific Tips**:
- `anthropic/claude-*`: Use Claude prompting style
- `openai/gpt-*`: Use Codex prompting style
- `google/gemini-*`: Use explicit depth instructions
- `mistralai/devstral-*`: Be specific, good at coding tasks

**Key Flags**:
- `--auto`: Required for non-interactive
- `--json`: JSON output (requires --auto)
- `--timeout N`: Timeout in seconds

## Depth Enhancement Patterns

### For Gemini (Required)

```
Before responding, follow these reasoning steps:
1. Parse the goal into distinct sub-tasks
2. Check if you have complete information for each sub-task
3. Create a structured outline of your response
4. Provide thorough, detailed analysis for each section
5. After drafting, self-critique: Did you answer the intent?

Provide comprehensive, in-depth responses rather than concise summaries.

[Your actual prompt]
```

### For Codex (Recommended)

```
Provide a detailed, comprehensive response. Include:
- Step-by-step reasoning
- Edge cases and error handling
- Alternative approaches considered
- Potential issues and mitigations

[Your actual prompt]
```

### For Claude (Usually Not Needed)

Claude typically provides thorough responses without explicit depth requests.
Only add depth instructions if getting unexpectedly brief responses.

## Planning Mode Prompts

### Claude
```
Create a detailed implementation plan for: [task]
```

### Codex
```
Create a comprehensive implementation plan with:
1. All files to modify
2. Step-by-step changes
3. Testing strategy
4. Potential risks

Task: [task]
```

### Gemini
```
You are creating a detailed implementation plan. Be thorough.

Before providing your plan:
1. Identify all components that need modification
2. Consider security, testing, and risks
3. Break down into ordered steps
4. Identify non-obvious considerations

Provide complete sections, not brief summaries.

Task: [task]
```

## Code Review Prompts

### Claude
```
Review this code thoroughly, checking for security,
performance, and maintainability issues.

[code]
```

### Codex
```
Perform a detailed code review covering:
- Security vulnerabilities (OWASP Top 10)
- Performance issues
- Error handling gaps
- Code quality and maintainability
- Test coverage recommendations

[code]
```

### Gemini
```
Perform a comprehensive code review. For each issue found:
1. Identify the severity (CRITICAL/HIGH/MEDIUM/LOW)
2. Explain why it's a problem
3. Provide a specific fix

Check for:
- Security vulnerabilities
- Performance bottlenecks
- Error handling gaps
- Maintainability issues
- Missing edge cases

Self-critique: What did you miss?

[code]
```

## Quick Reference

| If You Want... | Claude | Codex | Gemini |
|----------------|--------|-------|--------|
| Deep critique | "Do a deep critique" | "Provide detailed critique covering..." | [Use depth prefix] + "Critique..." |
| Code review | "Review this code" | "Review checking for X, Y, Z..." | [Use depth prefix] + "Review..." |
| Implementation plan | "Create a plan for..." | "Create comprehensive plan with sections for..." | [Use planning prefix] + "Plan..." |
| Quick answer | Works fine | Works fine | Works fine (default) |

## Adapter Usage

The `ai_orchestrator` handles these differences automatically:

```python
from ai_orchestrator.cli_adapters import (
    ClaudeAdapter,
    CodexAdapter,
    GeminiAdapter,  # Auto-adds depth enhancement
    KilocodeAdapter,
)

# Gemini adapter adds depth prefixes by default
gemini = GeminiAdapter()  # enhance_depth=True by default

# Disable if using GEMINI.md for persistent config
gemini = GeminiAdapter(enhance_depth=False)
```
