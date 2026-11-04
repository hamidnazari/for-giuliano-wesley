import argparse
import sys
from pathlib import Path

from research_assistant.agents.build import REAL_AGENTS, agents_builder
from research_assistant.agents.factory import MissingCredential
from research_assistant.board.render import render_board
from research_assistant.board.sink import load_snapshot
from research_assistant.graph.run import run_research
from research_assistant.interaction import AutoApproveIO, ConsoleIO
from research_assistant.settings import ConfigError, load_settings
from research_assistant.tools.search import make_backend


def _check_config(config_dir: Path) -> int:
    try:
        load_settings(config_dir)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"config ok: {config_dir}")
    return 0


def _run(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    stubs = sorted({"planner", "researcher", "critic", "writer"} - set(args.real))
    print(f"note: stub agents in use: {', '.join(stubs)}", file=sys.stderr)
    try:
        backend = None
        if "researcher" in args.real:
            cassette = args.cassette or args.data / "search_cassette.json"
            backend = make_backend(args.search, cassette)
        make_agents = agents_builder(
            settings, args.config / "prompts", args.real, search_backend=backend
        )
    except MissingCredential as e:
        print(f"{e}. Put it in .env and run through mise (mise exec -- ...).", file=sys.stderr)
        return 1
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    try:
        result = run_research(
            args.question,
            settings,
            args.data,
            AutoApproveIO() if args.auto_approve else ConsoleIO(),
            make_agents,
            kanzo=args.kanzo,
            kanzo_home=args.kanzo_home,
        )
    except MissingCredential as e:
        print(f"{e}. Put it in .env and run through mise (mise exec -- ...).", file=sys.stderr)
        return 1
    print(render_board(result.board))
    print(f"\nrun {result.run_id}: {result.status}")
    if result.stop_reason:
        print(f"stopped early: {result.stop_reason}")
    if result.report_path:
        print(f"report: {result.report_path}")
    return 2 if result.status == "stopped" else 0


def _board(data_root: Path, run_id: str | None) -> int:
    runs = sorted((data_root / "projects").glob("*/board.json"))
    if run_id:
        runs = [p for p in runs if p.parent.name == run_id]
    if not runs:
        print("no board found" + (f" for run {run_id}" if run_id else ""), file=sys.stderr)
        return 1
    print(render_board(load_snapshot(runs[-1])))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Multi-agent research assistant that delivers a cited report.",
    )
    sub = parser.add_subparsers(dest="command")
    check = sub.add_parser("check-config", help="validate the config directory")
    check.add_argument("--config", type=Path, default=Path("config"))
    run = sub.add_parser("run", help="research a question and write a cited report")
    run.add_argument("question")
    run.add_argument("--config", type=Path, default=Path("config"))
    run.add_argument("--data", type=Path, default=Path("data"))
    run.add_argument("--auto-approve", action="store_true", help="skip the plan approval prompt")
    run.add_argument(
        "--real",
        action="append",
        default=[],
        choices=REAL_AGENTS,
        metavar="AGENT",
        help="run this agent on a real model (repeatable); the rest stay stubs",
    )
    run.add_argument(
        "--search",
        choices=["live", "record", "replay"],
        default="live",
        help="web search mode; record and replay use a cassette file",
    )
    run.add_argument("--cassette", type=Path, default=None, help="search cassette (JSON)")
    run.add_argument("--kanzo", action="store_true", help="mirror the board into Kanzo")
    run.add_argument("--kanzo-home", type=Path, default=None, help="Kanzo data folder")
    board = sub.add_parser("board", help="print a run's board (latest run by default)")
    board.add_argument("--data", type=Path, default=Path("data"))
    board.add_argument("--run", default=None, help="run ID")
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config(args.config)
    if args.command == "run":
        return _run(args)
    if args.command == "board":
        return _board(args.data, args.run)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
