import argparse
import sys
from pathlib import Path

from research_assistant.agents.protocols import Agents
from research_assistant.agents.stubs import StubCritic, StubPlanner, StubResearcher, StubWriter
from research_assistant.board.render import render_board
from research_assistant.board.sink import load_snapshot
from research_assistant.graph.run import run_research
from research_assistant.guardrails.budget import Budget
from research_assistant.interaction import AutoApproveIO, ConsoleIO
from research_assistant.settings import ConfigError, load_settings


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
    print("note: using stub agents; real agents are not built yet", file=sys.stderr)

    def make_agents(budget: Budget) -> Agents:
        return Agents(StubPlanner(), StubResearcher(budget), StubCritic(), StubWriter())

    result = run_research(
        args.question,
        settings,
        args.data,
        AutoApproveIO() if args.auto_approve else ConsoleIO(),
        make_agents,
        kanzo=args.kanzo,
        kanzo_home=args.kanzo_home,
    )
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
