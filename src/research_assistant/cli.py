import argparse
import sys
from pathlib import Path

from research_assistant.settings import ConfigError, load_settings


def _check_config(config_dir: Path) -> int:
    try:
        load_settings(config_dir)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"config ok: {config_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Multi-agent research assistant that delivers a cited report.",
    )
    sub = parser.add_subparsers(dest="command")
    check = sub.add_parser("check-config", help="validate the config directory")
    check.add_argument("--config", type=Path, default=Path("config"))
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config(args.config)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
