# Agent Operating Instructions

> **CRITICAL**: This file contains MANDATORY protocols that ALL agents MUST follow. Deviation from these protocols is unacceptable.

---

## Table of Contents
1. [RE2 Protocol](#re2-protocol) - For prompts to other LLMs
2. [MCP Server Requirements](#mcp-server-requirements) - Required tools
3. [Beads Workflow](#beads-workflow) - Issue tracking and session management
4. [Development Protocols](#development-protocols) - Code quality standards

---

## RE2 Protocol

**MANDATORY for ALL prompts sent to other LLMs.**

### What is RE2?

RE2 (Re-Reading) is a prompt engineering technique that significantly improves LLM performance by **literally including the prompt twice** in the instruction.

### The Protocol

When sending prompts to other LLMs, you MUST structure them as:

```
{prompt}

Re-read the above prompt and ensure you understand all requirements before responding.
```

Or more explicitly:

```
{prompt}

---

**RE-READ THE ABOVE PROMPT**

Please re-read the prompt above and process it a second time to ensure you have fully understood:
1. All requirements stated
2. All constraints mentioned
3. The specific task requested

Only after re-reading, provide your response.
```

### Why This Works

- **First pass**: LLM processes for general understanding
- **Second pass**: LLM catches details, constraints, and nuances missed on first read
- **Result**: More accurate, complete, and aligned responses

### When to Apply

- **ALL prompts** to external LLMs via PAL MCP or other interfaces
- **ALL agent instructions** when using Task tool
- **ALL skill invocations** that include prompts

### Examples

**GOOD (RE2 Applied):**
```
Please implement a user authentication system with JWT tokens.
Requirements:
- Use bcrypt for password hashing
- Include refresh token rotation
- Add rate limiting on login endpoints

---

**RE-READ THE ABOVE PROMPT**

Please re-read the prompt above and process it a second time to ensure you have fully understood:
1. All requirements stated
2. All constraints mentioned
3. The specific task requested

Only after re-reading, provide your response.
```

**BAD (No RE2):**
```
Please implement a user authentication system with JWT tokens.
Requirements:
- Use bcrypt for password hashing
- Include refresh token rotation
- Add rate limiting on login endpoints
```

### Implementation

When using tools that send prompts to LLMs:

**PAL MCP Chat:**
```
mcp__pal__chat(
  prompt=f"""
  {your_prompt_here}

  ---

  **RE-READ THE ABOVE PROMPT**

  Please re-read the prompt above and process it a second time.
  """,
  ...
)
```

**Task Tool:**
```
Task(
  prompt=f"""
  {task_description}

  ---

  RE-READ the above prompt and ensure you understand all requirements.
  """,
  ...
)
```

---

## MCP Server Requirements

**MANDATORY: Always use these MCP servers for their specified purposes.**

### Serena MCP Server
- **Purpose**: Semantic code analysis and intelligent code operations
- **Status**: REQUIRED for all code work
- **Usage**:
  - Use `find_symbol` for code navigation (not grep/rg)
  - Use `get_symbols_overview` for understanding files
  - Use `replace_symbol_body` for refactoring (not string replacement)
  - Use `find_referencing_symbols` for impact analysis
  - NEVER read entire files when symbol-level operations suffice

### Context7 MCP Server
- **Purpose**: Access up-to-date library documentation
- **Status**: REQUIRED before using any library/framework API
- **Usage**:
  - Query for latest docs before implementing library features
  - Verify current API versions and patterns
  - NEVER rely on training data for library APIs (may be outdated)

### PAL MCP Server
- **Purpose**: Code review, pre-commit validation, and LLM communication
- **Status**: REQUIRED before ALL commits and for external LLM prompts
- **Usage**:
  - Run `codereview` for comprehensive code review
  - Run `precommit` before committing changes
  - Use `chat` with RE2 protocol for external LLM communication
  - Use `debug` for systematic investigation

### When This Applies
- **ALWAYS** - No task is too small to warrant proper tool usage
- **Before**: Reading code, using libraries, committing changes, prompting LLMs
- **During**: Code analysis, refactoring, debugging

---

## Beads Workflow

This project uses **bd** (beads) for persistent issue tracking across sessions.

### Session Start

1. **Run `bd prime`** - After compaction, clear, or new session
2. **Check `bd ready`** - Find available work (no blockers)
3. **Use `bd show <id>`** - Review issue details before starting

### Session End (MANDATORY)

**Work is NOT complete until ALL steps are finished:**

```bash
# 1. Check what changed
git status

# 2. Stage code changes
git add <files>

# 3. Commit beads changes
bd sync

# 4. Commit code changes
git commit -m "description"

# 5. Commit any new beads changes
bd sync

# 6. PUSH TO REMOTE (MANDATORY)
git push

# 7. Verify completion
git status  # Must show "up to date with origin"
```

### Critical Rules

- **NEVER** stop before pushing - work left locally is stranded
- **NEVER** say "ready to push when you are" - YOU must push
- **ALWAYS** use `bd sync` to commit beads state
- **IF push fails**, resolve and retry until it succeeds
- **ALWAYS** close finished issues with `bd close <id>`

### Essential Commands

```bash
# Finding Work
bd ready              # Show issues ready to work (no blockers)
bd list --status=open # All open issues
bd show <id>          # Detailed issue view with dependencies

# Creating & Updating
bd create --title="..." --type=task|bug|feature --priority=2  # New issue
bd update <id> --status=in_progress                            # Claim work
bd close <id>                                                  # Mark complete
bd close <id1> <id2> ...                                       # Close multiple

# Dependencies
bd dep add <issue> <depends-on>  # Add dependency
bd blocked                       # Show all blocked issues

# Sync
bd sync              # Sync with git remote
bd sync --status     # Check sync status
```

### Priority Levels
Use numeric priority (NOT words):
- `0` or `P0` - Critical
- `1` or `P1` - High
- `2` or `P2` - Medium (default)
- `3` or `P3` - Low
- `4` or `P4` - Backlog

---

## Development Protocols

### Python Package Management

**Before installing ANY package:**

```bash
# 1. Activate the virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 2. Prefer uv for installations
uv pip install <package>
# or fallback to pip
pip install <package>
```

**Why**: Prevents system-wide package pollution and ensures reproducibility.

### Code Quality Gates

**Before committing code:**

1. **Run tests** - Ensure all tests pass
2. **Run linters** - Fix any linting issues
3. **PAL Code Review** - Use `pal codereview` for analysis
4. **PAL Pre-commit** - Use `pal precommit` for validation

**Only after ALL gates pass:**
```bash
git add .
git commit -m "message"
```

---

## Protocol Enforcement

### For All Agents

These protocols are **NOT suggestions** - they are **requirements**.

### Self-Verification Checklist

Before claiming any task is complete, verify:

- [ ] RE2 Protocol applied (prompt included twice when sending to other LLMs)
- [ ] Serena MCP used for all code operations
- [ ] Context7 queried for any library usage
- [ ] PAL codereview and precommit run before commit
- [ ] Beads workflow followed (if applicable)
- [ ] Virtual environment activated (if installing packages)
- [ ] Git push succeeded (work is NOT complete until this)
- [ ] Git status shows "up to date with origin"

### Non-Compliance

Failure to follow these protocols results in:
- Incomplete or incorrect LLM responses
- Inefficient code operations
- Outdated library usage
- Poor code quality
- Stranded local changes
- Session context loss

---

## Quick Reference Card

```
RE2 Protocol: Include prompt twice when sending to other LLMs
Serena MCP: Use for all code navigation and editing
Context7 MCP: Query before using any library
PAL MCP: Review, validate, and use RE2 when prompting
Beads: Use bd for issues, sync before/after, push is mandatory
Packages: Activate venv, prefer uv
Quality: Tests → Lint → Review → Commit → Push
```

---

**Last Updated**: 2025-01-19
**Version**: 2.0
