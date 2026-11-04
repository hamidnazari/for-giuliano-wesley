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

## Watching a run on a kanban board (Kanzo)

[Kanzo](https://github.com/emanuele-em/kanzo) is a self-hosted kanban board. `mise install`
installs the pinned version. It is optional: the run works the same without it.

1. Start the board. It opens http://localhost:4242 (the port is set in `~/kanzo/config.toml`):

        mise exec -- kanzo                      # or: kanzo --port 4300 --no-open
        mise exec -- kanzo --home ~/some/dir    # use a different data folder

2. Run with `--kanzo`, in another terminal. The board can stay open: cards appear and move as
   the run progresses.

        mise exec -- uv run research-assistant run "your question" --kanzo \
          --real planner --real researcher --real critic --real writer

   Add `--kanzo-home <dir>` if the board uses a data folder other than `~/kanzo`.

3. In the browser, each run is one project named `Research <run-id>`, with one task per card.
   Open a task to see its sub-question, notes, sources with credibility and the
   researcher and Critic comments.

| Our card status | Kanzo column |
| --- | --- |
| Backlog | todo |
| In progress | doing |
| In review | review |
| Done | done |
| Blocked / Rejected | review, tagged `blocked` or `rejected` |

The board is a read-only mirror. The run decides what moves where; changes made in Kanzo do not
flow back, and a Kanzo failure is logged as `sink_error` without stopping the run.

Other commands:

    mise exec -- kanzo export -p <run-id>      # dump one run's board as Markdown (-f json for JSON)
    mise exec -- kanzo service                 # start the board automatically at login
    uv run research-assistant board            # the same board as text, without Kanzo

Kanzo has no logins and listens on 127.0.0.1 only. Keep it that way unless you trust your
network.

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
