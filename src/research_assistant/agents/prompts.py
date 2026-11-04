"""Prompts live in config/prompts and are versioned with the code."""

from pathlib import Path


def read_prompt(prompts_dir: Path, name: str) -> str:
    return (prompts_dir / name).read_text(encoding="utf-8").strip()
