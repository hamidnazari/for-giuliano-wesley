import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from research_assistant.settings import ConfigError, load_settings

REPO_CONFIG = Path(__file__).parent.parent / "config"

Edit = Callable[[dict[str, Any]], None]


def make_config(tmp_path: Path, file: str | None = None, edit: Edit | None = None) -> Path:
    dest = tmp_path / "config"
    shutil.copytree(REPO_CONFIG, dest)
    if file and edit:
        path = dest / file
        data = yaml.safe_load(path.read_text())
        edit(data)
        path.write_text(yaml.safe_dump(data))
    return dest


def test_shipped_config_loads() -> None:
    settings = load_settings(REPO_CONFIG)
    critic, researcher = settings.agents.critic.model, settings.agents.researcher.model
    assert critic.split("/")[0] != researcher.split("/")[0]


CASES: list[tuple[str, str, Edit, str]] = [
    (
        "unknown tool",
        "agents.yaml",
        lambda d: d["researcher"]["tools"].append("nope"),
        "unknown tool 'nope'",
    ),
    (
        "write tool to researcher",
        "agents.yaml",
        lambda d: d["researcher"]["tools"].append("render_report"),
        "write-capable",
    ),
    (
        "planner with tools",
        "agents.yaml",
        lambda d: d["planner"].update(tools=["web_search"]),
        "no tools",
    ),
    (
        "critic open search",
        "agents.yaml",
        lambda d: d["critic"]["tools"].append("web_search"),
        "only",
    ),
    (
        "same model",
        "agents.yaml",
        lambda d: d["critic"].update(model=d["researcher"]["model"]),
        "different model",
    ),
    (
        "bad model id",
        "agents.yaml",
        lambda d: d["writer"].update(model="no-provider"),
        "writer.model",
    ),
    ("missing prompt", "agents.yaml", lambda d: d["writer"].update(prompt="gone.md"), "not found"),
    ("zero cap", "guardrails.yaml", lambda d: d.update(max_critic_rounds=0), "max_critic_rounds"),
    ("negative cap", "guardrails.yaml", lambda d: d.update(max_searches=-1), "max_searches"),
    (
        "search cap cannot cover the plan",
        "guardrails.yaml",
        lambda d: d.update(max_searches=10),
        "cannot cover",
    ),
    ("unknown key", "guardrails.yaml", lambda d: d.update(surprise=1), "surprise"),
    (
        "overlap too big",
        "tools.yaml",
        lambda d: d["ingestion"].update(chunk_overlap=900),
        "chunk_overlap",
    ),
]


@pytest.mark.parametrize(("name", "file", "edit", "expected"), CASES, ids=[c[0] for c in CASES])
def test_bad_config_fails_loudly(
    tmp_path: Path, name: str, file: str, edit: Edit, expected: str
) -> None:
    with pytest.raises(ConfigError, match=expected):
        load_settings(make_config(tmp_path, file, edit))


def test_missing_file_fails(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    (cfg / "guardrails.yaml").unlink()
    with pytest.raises(ConfigError, match="guardrails.yaml: file not found"):
        load_settings(cfg)


def test_all_problems_reported_together(tmp_path: Path) -> None:
    def edit(d: dict[str, Any]) -> None:
        d["planner"]["tools"] = ["web_search"]
        d["writer"]["prompt"] = "gone.md"

    with pytest.raises(ConfigError) as exc:
        load_settings(make_config(tmp_path, "agents.yaml", edit))
    assert len(exc.value.problems) >= 2
