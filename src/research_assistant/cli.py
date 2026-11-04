import argparse
import sys
from pathlib import Path

from research_assistant.board.render import render_board
from research_assistant.board.sink import load_snapshot
from research_assistant.settings import ConfigError, load_settings


def _check_config(config_dir: Path) -> int:
    try:
        load_settings(config_dir)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"config ok: {config_dir}")
    return 0


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
    board = sub.add_parser("board", help="print a run's board (latest run by default)")
    board.add_argument("--data", type=Path, default=Path("data"))
    board.add_argument("--run", default=None, help="run ID")
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config(args.config)
    if args.command == "board":
        return _board(args.data, args.run)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
