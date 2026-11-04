"""Fixtures for the Critic defect eval: cards with known-good and planted-bad notes."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

OK = "ok"
DEFECT_TYPES = {
    "overstatement",
    "contradiction",
    "fake_citation",
    "fabricated_quote",
    "weak_source",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceCase(_Strict):
    id: str
    url: str
    title: str
    page: str = Field(min_length=1)  # frozen page text, returned by the fake page backend


class NoteCase(_Strict):
    text: str
    sources: list[str] = Field(min_length=1)
    expect: str  # "ok" or one of DEFECT_TYPES


class CriticCase(_Strict):
    id: str
    sub_question: str
    sources: list[SourceCase]
    notes: list[NoteCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> "CriticCase":
        known = {s.id for s in self.sources}
        for n in self.notes:
            if n.expect != OK and n.expect not in DEFECT_TYPES:
                raise ValueError(f"{self.id}: unknown expectation '{n.expect}'")
            if set(n.sources) - known:
                raise ValueError(f"{self.id}: note cites a source the case does not define")
        return self


class CaseFile(_Strict):
    cases: list[CriticCase]

    @model_validator(mode="after")
    def _unique(self) -> "CaseFile":
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        return self


def load_cases(path: Path) -> list[CriticCase]:
    return CaseFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))).cases
