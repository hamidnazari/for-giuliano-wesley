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

Commands `index`, `run` and `board` are added as they are built.

## Checks

    mise run check             # format, lint, types, tests, config
    mise run secrets:history   # scan git history for secrets

