# Task Completion Checklist

## Before claiming work is complete:

### 1. Code Quality
- [ ] Run `ruff check .` - no lint errors
- [ ] Run `ruff format .` - code formatted
- [ ] Run `pytest` - all tests pass

### 2. Git
- [ ] Stage all changes: `git add .`
- [ ] Commit with descriptive message
- [ ] `git pull --rebase` - sync with remote
- [ ] `git push` - **MANDATORY - work NOT complete until push succeeds**

### 3. Beads (Issue Tracking)
- [ ] Close completed issues: `bd close <id>`
- [ ] Update in-progress items as needed
- [ ] Run `bd sync --flush-only`

### 4. Cleanup
- [ ] Clear any git stashes
- [ ] Verify `git status` shows clean or staged only

### 5. Verification
```bash
git status  # MUST show "up to date with origin" or clean working dir
```

## CRITICAL RULES
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- If push fails, resolve conflicts and retry until it succeeds
