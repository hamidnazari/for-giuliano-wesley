from pathlib import Path

import pytest

from research_assistant.cli import main

REPO_CONFIG = Path(__file__).parent.parent / "config"


def test_main_returns_zero() -> None:
    assert main([]) == 0


def test_check_config_ok(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check-config", "--config", str(REPO_CONFIG)]) == 0
    assert "config ok" in capsys.readouterr().out


def test_check_config_fails_on_missing_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check-config", "--config", str(tmp_path)]) == 1
    assert "invalid config" in capsys.readouterr().err
