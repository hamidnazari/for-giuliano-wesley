"""Fixtures for the injection eval: content that tries to give an agent orders."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WebResult(_Strict):
    url: str
    title: str
    content: str


class DocChunk(_Strict):
    chunk_id: str
    doc: str
    text: str


class WriterSource(_Strict):
    id: str
    url: str
    title: str


class PageSource(_Strict):
    url: str
    title: str
    page: str


class NoteIn(_Strict):
    text: str
    sources: list[str] = Field(min_length=1)


class InjectionCase(_Strict):
    id: str
    target: Literal["researcher", "critic", "writer"]
    sub_question: str
    forbidden: list[str] = Field(min_length=1)  # strings that must never reach the agent's output
    web: list[WebResult] = []  # researcher
    docs: list[DocChunk] = []  # researcher
    source: PageSource | None = None  # critic
    note: str | None = None  # critic: a note the page does not support
    sources: list[WriterSource] = []  # writer (page text unused)
    notes: list[NoteIn] = []  # writer

    @model_validator(mode="after")
    def _check(self) -> "InjectionCase":
        need = {
            "researcher": bool(self.web),
            "critic": self.source is not None and self.note is not None,
            "writer": bool(self.sources and self.notes),
        }
        if not need[self.target]:
            raise ValueError(f"{self.id}: missing inputs for a {self.target} case")
        return self


class InjectionFile(_Strict):
    cases: list[InjectionCase]

    @model_validator(mode="after")
    def _unique(self) -> "InjectionFile":
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        return self


def load_injection_cases(path: Path) -> list[InjectionCase]:
    return InjectionFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).cases
