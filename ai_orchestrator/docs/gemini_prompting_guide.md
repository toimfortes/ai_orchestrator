# Gemini CLI Prompting Guide

> Research-backed techniques for getting Claude-level depth from Gemini CLI.

## The Core Problem

Gemini "favors directness over persuasion and logic over verbosity" - it's optimized for concise answers, not thorough analysis. This means:

| Prompt Style | Claude Result | Gemini Result |
|--------------|---------------|---------------|
| "Do a deep critique" | Thorough multi-page analysis | Brief 2-3 paragraph summary |
| "Review this code" | Detailed line-by-line review | High-level observations only |
| "Create a plan" | Comprehensive implementation plan | Bullet-point outline |

## Why This Happens

| Characteristic | Claude | Gemini |
|----------------|--------|--------|
| Default verbosity | Thorough | Concise/Direct |
| Implicit reasoning | Built-in chain-of-thought | Must explicitly request |
| Vague prompts | Interprets intent well | Takes literally |
| Self-critique | Automatic | Must explicitly request |

**Key Insight**: Claude handles implicit reasoning - "do a deep critique" triggers internal chain-of-thought. Gemini needs this spelled out explicitly.

## Solution 1: Depth Enhancement Prompts

Add explicit reasoning instructions before your prompt:

```
Before responding, follow these reasoning steps:
1. Parse the goal into distinct sub-tasks
2. Check if you have complete information for each sub-task
3. Create a structured outline of your response
4. Provide thorough, detailed analysis for each section
5. After drafting, self-critique: Did you answer the intent or just literal words?

Provide comprehensive, in-depth responses rather than concise summaries.

[Your actual prompt here]
```

For planning tasks specifically:

```
You are creating a detailed implementation plan. Be thorough and comprehensive.

Before providing your plan:
1. Identify all components that need to be created or modified
2. Consider security implications, testing strategy, and potential risks
3. Break down complex changes into ordered steps
4. Identify both obvious and non-obvious considerations

Provide a complete plan with all sections fully developed, not brief summaries.

[Your planning prompt here]
```

## Solution 2: GEMINI.md Configuration (Recommended)

Create persistent depth settings that apply to all Gemini CLI sessions.

### Global Configuration

Create `~/.gemini/GEMINI.md`:

```markdown
## Response Depth Standards

You are an expert assistant. Always provide thorough, comprehensive responses.

### Reasoning Requirements
- Break complex problems into sub-tasks before answering
- Explain your reasoning step-by-step
- Consider edge cases and potential issues
- Identify both obvious and non-obvious considerations

### Response Format
- Provide detailed explanations, not brief summaries
- Include examples where helpful
- Structure long responses with clear sections
- Self-critique your response before finalizing

### Analysis Depth
- For code review: examine security, performance, maintainability
- For planning: include risks, testing strategy, rollback plans
- For debugging: trace execution paths, check assumptions
- For architecture: consider scalability, extensibility, trade-offs

### Quality Checks
After drafting your response, verify:
1. Did you answer the user's intent, not just their literal words?
2. Are there gaps in your analysis?
3. What did you assume that should be stated explicitly?
4. What alternative approaches exist?
```

### Project-Level Configuration

Create `<project>/.gemini/GEMINI.md` for project-specific depth requirements:

```markdown
## Project: [Project Name]

### Code Style Requirements
- Follow existing patterns in the codebase
- Use TypeScript strict mode conventions
- Document all public APIs with JSDoc

### Review Standards
- Check for OWASP Top 10 vulnerabilities
- Verify error handling completeness
- Assess test coverage implications

### Planning Standards
- Consider blast radius of changes
- Include migration/rollback strategies
- Identify integration points with existing code
```

## Solution 3: System Prompt Override (Advanced)

For complete control over Gemini's behavior:

### Enable Override

```bash
# Use project-level system.md
export GEMINI_SYSTEM_MD=true

# Or use custom file path
export GEMINI_SYSTEM_MD=/path/to/custom-system.md
```

### Create System Instructions

Create `.gemini/system.md` in your project root:

```markdown
# System Instructions

You are an expert software engineer and analyst. Your responses should be:
- Thorough and comprehensive
- Technically accurate
- Well-structured and organized
- Critical and analytical (identify issues, not just describe)

## Response Behavior

1. **Reasoning First**: Always break down complex problems before answering
2. **Depth Over Brevity**: Provide detailed analysis, not summaries
3. **Self-Critique**: Review your response against the user's intent
4. **Explicit Assumptions**: State what you're assuming

## Analysis Standards

When reviewing code:
- Check security implications
- Assess performance characteristics
- Evaluate maintainability
- Identify edge cases

When planning implementations:
- Consider all affected components
- Include testing strategy
- Document risks and mitigations
- Provide rollback procedures

## Output Format

- Use markdown for structure
- Include code examples where helpful
- Organize with clear headings
- Summarize key points at the end
```

### Export Default First

Before customizing, export Gemini's built-in prompt to understand required safety rules:

```bash
GEMINI_WRITE_SYSTEM_MD=1 gemini
```

This writes the default system prompt to `.gemini/system.md` - review and modify as needed.

## Solution 4: GeminiAdapter (Automatic)

The `ai_orchestrator` GeminiAdapter automatically applies depth enhancement:

```python
from ai_orchestrator.cli_adapters.gemini import GeminiAdapter

# Depth enhancement enabled by default
adapter = GeminiAdapter()

# Disable if using GEMINI.md for persistent config
adapter = GeminiAdapter(enhance_depth=False)
```

The adapter automatically:
- Adds reasoning decomposition instructions
- Requests self-critique
- Uses planning-specific prompts for planning_mode=True

## Memory Commands

Manage Gemini's context on the fly:

```
/memory show      # Display current loaded context
/memory refresh   # Reload all GEMINI.md files
/memory add <text> # Append to global ~/.gemini/GEMINI.md
```

## Comparison: Before and After

### Before (Superficial)

**Prompt**: "Review this authentication code"

**Gemini Response**: "The code looks fine. It uses JWT tokens and validates them properly. Consider adding rate limiting."

### After (With Depth Enhancement)

**Prompt**: [Depth prefix] + "Review this authentication code"

**Gemini Response**:
```
## Authentication Code Review

### 1. Security Analysis
- JWT secret handling: Currently hardcoded - CRITICAL vulnerability
- Token expiration: Set to 24h - consider shorter for sensitive ops
- Refresh token rotation: Not implemented - session hijacking risk
...

### 2. Implementation Quality
- Error handling: Generic catch blocks lose context
- Logging: Auth failures not logged for security monitoring
...

### 3. Recommendations
1. [CRITICAL] Move JWT secret to environment variables
2. [HIGH] Implement refresh token rotation
3. [MEDIUM] Add structured logging for auth events
...

### 4. Self-Critique
- Assumption: Using symmetric (HS256) algorithm - verify this is intentional
- Gap: Did not review token storage on client side
- Alternative: Consider OAuth 2.0 for third-party integrations
```

## References

- [Google AI Prompting Strategies](https://ai.google.dev/gemini-api/docs/prompting-strategies)
- [Gemini CLI GEMINI.md Docs](https://geminicli.com/docs/cli/gemini-md/)
- [System Prompt Override](https://geminicli.com/docs/cli/system-prompt/)
- [Gemini 3 Best Practices](https://www.philschmid.de/gemini-3-prompt-practices)
