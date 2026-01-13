# Suggested Commands - ChromaDB POC

## System (Linux)
```bash
ls -la              # List directory contents
cd <path>           # Change directory
pwd                 # Print working directory
grep -r "pattern" . # Search in files
find . -name "*.py" # Find files
```

## Git
```bash
git status          # Show working tree status
git add .           # Stage changes
git commit -m "msg" # Commit changes
git pull --rebase   # Pull and rebase
git push            # Push to remote
git log --oneline   # Show commit history
```

## Beads (Issue Tracking)
```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync --flush-only  # Export to JSONL
bd stats              # Project statistics
```

## Python (To be set up)
```bash
python src/main.py       # Run entry point
pytest                   # Run tests
ruff check .             # Lint code
ruff format .            # Format code
ruff check --fix .       # Fix lint issues
```

## Session Completion (MANDATORY)
```bash
# 1. File issues for remaining work
bd create --title="..."

# 2. Run quality gates (if code changed)
ruff check .
ruff format --check .
pytest

# 3. Update issue status
bd close <id>
bd update <id> --status in_progress

# 4. PUSH TO REMOTE (MANDATORY)
git pull --rebase
bd sync
git push
git status  # MUST show "up to date with origin"
```
