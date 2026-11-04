from pathlib import Path

import pytest

from research_assistant.rendering.pdf import (
    PdfUnavailable,
    _deny_fetch,  # pyright: ignore[reportPrivateUsage]
    load_weasyprint,
    markdown_to_html,
    markdown_to_pdf,
)
from research_assistant.tools.render import PDF_NAME, REPORT_NAME, make_render_report

REPORT = """# Is pgvector enough?

## Findings

- pgvector supports HNSW and IVFFlat. [c1-s1]
- A table follows.

| a | b |
|---|---|
| 1 | 2 |

## Sources

- [c1-s1] pgvector README: https://docs.example/pgvector
"""


def pdf_works() -> bool:
    try:
        load_weasyprint()
    except PdfUnavailable:
        return False
    return True


needs_pango = pytest.mark.skipif(not pdf_works(), reason="pango is not installed")


def test_headings_lists_and_tables_become_html() -> None:
    html = markdown_to_html(REPORT)
    assert "<h1>" in html and "<h2>" in html and "<li>" in html and "<table>" in html


def test_raw_html_in_a_report_is_escaped_not_passed_through() -> None:
    evil = (
        "# T\n\n<script>alert(1)</script>\n\n"
        '<img src="https://evil.example/pixel.png">\n\n'
        '<link rel="stylesheet" href="https://evil.example/x.css">\n'
    )
    html = markdown_to_html(evil)
    assert "<script>" not in html and "<img" not in html and "<link" not in html
    assert "&lt;img" in html  # shown as text instead


def test_the_title_cannot_inject_markup() -> None:
    assert "<script>" not in markdown_to_html("x", title="<script>alert(1)</script>")


def test_the_renderer_refuses_every_url() -> None:
    with pytest.raises(ValueError, match="not allowed"):
        _deny_fetch("https://evil.example/pixel.png")


@needs_pango
def test_a_pdf_is_written_and_nothing_is_fetched(tmp_path: Path) -> None:
    target = tmp_path / "r.pdf"
    markdown_to_pdf(REPORT + '\n<img src="https://evil.example/p.png">\n', target)
    data = target.read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1000
    assert not list(tmp_path.glob("*.tmp"))


@needs_pango
def test_the_render_tool_writes_both_files_and_returns_the_markdown_path(tmp_path: Path) -> None:
    errors: list[str] = []
    path = make_render_report(tmp_path, errors.append)(markdown=REPORT)
    assert Path(path) == (tmp_path / REPORT_NAME).resolve()
    assert (tmp_path / PDF_NAME).read_bytes().startswith(b"%PDF") and errors == []


def test_a_pdf_failure_keeps_the_markdown_and_reports_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(markdown: str, target: Path) -> None:
        raise PdfUnavailable("no pango")

    monkeypatch.setattr("research_assistant.tools.render.markdown_to_pdf", boom)
    (tmp_path / PDF_NAME).write_bytes(b"stale")
    errors: list[str] = []
    path = make_render_report(tmp_path, errors.append)(markdown=REPORT)
    assert Path(path).read_text() == REPORT and errors == ["no pango"]
    assert not (tmp_path / PDF_NAME).exists()  # a stale PDF is removed
