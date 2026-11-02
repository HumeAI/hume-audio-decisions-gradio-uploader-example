# Agent instructions

Before writing code that calls the Audio Decisions API, read https://dev.hume.ai/audio-decisions/docs/for-coding-agents.md. For any other Audio Decisions page, find it in https://dev.hume.ai/audio-decisions/llms.txt and fetch its Markdown by adding .md to the page URL.

## Commands

1. `uv sync` installs the app and the development tools.
2. `uv run app.py` starts the app at http://localhost:7860. It needs `HUME_API_KEY` in the environment.
3. `uv run ruff format --check`, `uv run ruff check` and `uv run mypy .` are the checks CI runs.

## Rules

1. `requirements.txt` is exported from `uv.lock` for pip users, and CI fails when the two differ. After any change to `uv.lock`, run:

   ```bash
   uv export --format requirements-txt --no-dev --no-hashes --output-file requirements.txt
   ```

2. Fix the code rather than adding `# type: ignore`, `# noqa`, or exceptions to the ruff or mypy configuration.
3. The API key stays on the server. Never send it to the browser or write it into the repository.
