# Research Assistant

A command-line tool where a team of AI agents researches a question and delivers a cited markdown report.

## Setup

    mise install
    uv sync
    uv run pre-commit install
    cp .env.example .env   # then fill in KILO_API_KEY and TAVILY_API_KEY

## Run

    uv run research-assistant --help
    uv run research-assistant check-config

API keys in `.env` are loaded by mise, so run commands from the project directory in a
mise-activated shell, or prefix them with `mise exec --`.

    uv run research-assistant run "your question" --auto-approve
    uv run research-assistant run "your question" --real planner   # planner on a real model
    uv run research-assistant index ./my-docs      # optional: let researchers use your own files
    uv run research-assistant board

Agents without a real implementation yet run as stubs. `index` is added as it is built.

## Checks

    mise run check             # format, lint, types, tests, config
    mise run secrets:history   # scan git history for secrets

