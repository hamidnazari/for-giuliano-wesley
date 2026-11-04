# Research Assistant

A command-line tool where a team of AI agents researches a question using the web and your own
documents, checks its own work, and delivers a cited report as markdown and PDF.

Planner (asks a few questions, you approve the plan) -> Researchers (one per card, in parallel)
-> Critic (a different model family; checks every citation against the cited page) -> Writer
(approved cards only; every claim cited). Claims the Critic cannot support are removed and listed
under Limitations.

## Setup

    mise install
    uv sync
    uv run pre-commit install
    cp .env.example .env   # then fill in KILO_API_KEY and TAVILY_API_KEY

PDF output needs pango. macOS: `brew install pango`. Debian/Ubuntu:
`apt install libpango-1.0-0 libpangoft2-1.0-0`. Without it the markdown report is still written
and the run notes that the PDF was not produced.

API keys in `.env` are loaded by mise, so run commands from the project directory in a
mise-activated shell, or prefix them with `mise exec --`.

## Run

    mise exec -- uv run research-assistant run "your question" \
      --real planner --real researcher --real critic --real writer

Add `--auto-approve` to skip the prompts. Without `--real <agent>` that agent is a stub, which
is how the tests run. Reports land in `data/projects/<run-id>/` as `report.md` and `report.pdf`,
next to `board.json` and `audit.jsonl` (every tool call and model call).

    uv run research-assistant index ./my-docs    # let researchers use your own .md/.txt/.pdf files
    uv run research-assistant board              # print the latest board
    uv run research-assistant check-config

Options: `--search live|record|replay` with `--cassette <file>` to record or replay web results,
`--kanzo` to mirror the board into a Kanzo kanban.

## Checks

    mise run check             # format, lint, types, tests, config. No keys needed.
    mise run secrets:history   # scan git history for secrets

## Evals

Live-model gates (need `KILO_API_KEY`). Thresholds are in `evals/thresholds.yaml`; each command
exits 1 if a gate fails and `--out file.json` saves the full results.

    mise run eval:critic       # catch rate on planted defects, false positives
    mise run eval:injection    # agents must ignore instructions planted in pages, documents, notes
    mise run eval:golden       # end to end over a frozen corpus: citation validity, cost, cards

CI runs the checks on every push and the three evals on pull requests from this repository
(add a `KILO_API_KEY` repository secret). Results upload as the `eval-results` artifact.
