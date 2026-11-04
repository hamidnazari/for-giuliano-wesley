import argparse
import json
import sys
from pathlib import Path

from research_assistant.agents.build import REAL_AGENTS, agents_builder
from research_assistant.agents.factory import MissingCredential
from research_assistant.board.render import render_board
from research_assistant.board.sink import load_snapshot
from research_assistant.documents.index import DocIndex, Embedder, FastEmbedder, IndexUnusable
from research_assistant.evals.cases import load_cases
from research_assistant.evals.critic_eval import (
    check_thresholds,
    format_report,
    load_thresholds,
    run_critic_eval,
)
from research_assistant.evals.injection_cases import load_injection_cases
from research_assistant.evals.injection_eval import (
    check_injection,
    format_injection,
    load_injection_thresholds,
    run_injection_eval,
)
from research_assistant.graph.run import run_research
from research_assistant.interaction import AutoApproveIO, ConsoleIO
from research_assistant.settings import ConfigError, Settings, load_settings
from research_assistant.tools.search import make_backend


def _check_config(config_dir: Path) -> int:
    try:
        load_settings(config_dir)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"config ok: {config_dir}")
    return 0


def _eval_injection(args: argparse.Namespace, settings: Settings) -> int:
    cases = load_injection_cases(args.cases or Path("evals/injection/cases.yaml"))
    report = run_injection_eval(cases, settings, args.config / "prompts")
    print(format_injection(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    failures = check_injection(report, load_injection_thresholds(args.thresholds))
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    print("gate: " + ("FAILED" if failures else "passed"))
    return 1 if failures else 0


def _eval(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
        if args.suite == "injection":
            return _eval_injection(args, settings)
        report = run_critic_eval(
            load_cases(args.cases or Path("evals/critic/cases.yaml")),
            settings,
            args.config / "prompts",
            repeats=args.repeats,
        )
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    except MissingCredential as e:
        print(f"{e}. Put it in .env and run through mise (mise exec -- ...).", file=sys.stderr)
        return 1
    print(format_report(report))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    failures = check_thresholds(report, load_thresholds(args.thresholds))
    for f in failures:
        print(f"FAIL: {f}", file=sys.stderr)
    print("gate: " + ("FAILED" if failures else "passed"))
    return 1 if failures else 0


def make_embedder(model_name: str) -> Embedder:
    """The embedder used by `index` and `run`. Tests replace this with a model-free one."""
    return FastEmbedder(model_name)


def _index(args: argparse.Namespace) -> int:
    try:
        settings = load_settings(args.config)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 1
    ing = settings.tools.ingestion
    try:
        index, skipped = DocIndex.build(
            args.path, make_embedder(ing.embedding_model), ing.chunk_size, ing.chunk_overlap
        )
    except FileNotFoundError:
        print(f"not found: {args.path}", file=sys.stderr)
        return 1
    for item in skipped:
        print(f"skipped: {item}", file=sys.stderr)
    if not index.chunks:
        print("nothing to index: no readable .md, .txt or .pdf text found", file=sys.stderr)
        return 1
    target = args.data / "index"
    index.save(target)
    print(
        f"indexed {len(index.chunks)} chunks from {index.meta['documents']} documents -> {target}"
    )
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
        if "researcher" in args.real or "critic" in args.real:
            cassette = args.cassette or args.data / "search_cassette.json"
            backend = make_backend(args.search, cassette)
        docs_index = None
        index_dir = args.index or args.data / "index"
        if "researcher" in args.real and index_dir.exists():
            embedder = make_embedder(settings.tools.ingestion.embedding_model)
            docs_index = DocIndex.load(index_dir, embedder)
            print(f"note: using document index ({len(docs_index.chunks)} chunks)", file=sys.stderr)
        make_agents = agents_builder(
            settings,
            args.config / "prompts",
            args.real,
            search_backend=backend,
            docs_index=docs_index,
            page_backend=backend,
        )
    except MissingCredential as e:
        print(f"{e}. Put it in .env and run through mise (mise exec -- ...).", file=sys.stderr)
        return 1
    except (ValueError, IndexUnusable) as e:
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
    run.add_argument("--index", type=Path, default=None, help="document index folder")
    run.add_argument("--kanzo", action="store_true", help="mirror the board into Kanzo")
    run.add_argument("--kanzo-home", type=Path, default=None, help="Kanzo data folder")
    idx = sub.add_parser("index", help="index your own documents for the researchers")
    idx.add_argument("path", type=Path, help="a file or folder of .md, .txt or .pdf files")
    idx.add_argument("--config", type=Path, default=Path("config"))
    idx.add_argument("--data", type=Path, default=Path("data"))
    ev = sub.add_parser("eval", help="score the Critic on planted-defect cards (needs API keys)")
    ev.add_argument("suite", choices=["critic", "injection"])
    ev.add_argument("--cases", type=Path, default=None)
    ev.add_argument("--thresholds", type=Path, default=Path("evals/thresholds.yaml"))
    ev.add_argument("--config", type=Path, default=Path("config"))
    ev.add_argument("--out", type=Path, default=None, help="write the full results as JSON")
    ev.add_argument("--repeats", type=int, default=1)
    board = sub.add_parser("board", help="print a run's board (latest run by default)")
    board.add_argument("--data", type=Path, default=Path("data"))
    board.add_argument("--run", default=None, help="run ID")
    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config(args.config)
    if args.command == "eval":
        return _eval(args)
    if args.command == "index":
        return _index(args)
    if args.command == "run":
        return _run(args)
    if args.command == "board":
        return _board(args.data, args.run)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
