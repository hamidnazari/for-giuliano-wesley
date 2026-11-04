"""Markdown report to PDF with WeasyPrint.

The report holds text that came from web pages, so two things are locked down:
- raw HTML in the markdown is not passed through (it is shown as text), and
- the renderer is not allowed to fetch any URL, so no image or stylesheet can phone home.
Links stay clickable but nothing is requested while rendering.
"""

import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from markdown_it import MarkdownIt

CSS = """
@page {
  size: A4; margin: 22mm 20mm;
  @bottom-center { content: counter(page); font-size: 9pt; color: #666; }
}
body {
  font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt;
  line-height: 1.5; color: #1a1a1a;
}
h1 { font-size: 20pt; margin: 0 0 12pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
p, li { orphans: 2; widows: 2; }
a { color: #1a4fa0; text-decoration: none; word-break: break-all; }
code { font-family: Menlo, monospace; font-size: 9pt; background: #f2f2f2; padding: 0 2pt; }
blockquote { margin: 8pt 0; padding-left: 10pt; border-left: 3px solid #ccc; color: #444; }
table { border-collapse: collapse; margin: 8pt 0; }
th, td { border: 1px solid #ccc; padding: 3pt 6pt; text-align: left; }
"""


class PdfUnavailable(RuntimeError):
    """WeasyPrint or its system libraries (pango) are missing, or rendering failed."""


def markdown_to_html(markdown: str, title: str = "Research report") -> str:
    """HTML for a report. Raw HTML in the input is escaped, never passed through."""
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    body = md.render(markdown)
    safe_title = md.renderInline(title)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{safe_title}</title></head><body>{body}</body></html>"
    )


def _deny_fetch(url: str, *args: Any, **kwargs: Any) -> Any:
    raise ValueError(f"fetching {url} is not allowed while rendering a report")


HOMEBREW_LIB = Path("/opt/homebrew/lib")


def load_weasyprint() -> ModuleType:
    """Import WeasyPrint, or raise PdfUnavailable if pango is missing.

    On macOS, Homebrew's libraries are not on the default search path, and the dyld variable
    must be set inside this process because system shells strip it from the environment.
    """
    if sys.platform == "darwin" and HOMEBREW_LIB.is_dir():
        current = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if str(HOMEBREW_LIB) not in current.split(":"):
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                p for p in (current, str(HOMEBREW_LIB)) if p
            )
    try:
        import weasyprint  # pyright: ignore[reportMissingTypeStubs]
    except OSError as e:  # the python package is there, the pango libraries are not
        raise PdfUnavailable(
            "WeasyPrint cannot load its system libraries (pango). "
            "macOS: brew install pango. "
            "Debian/Ubuntu: apt install libpango-1.0-0 libpangoft2-1.0-0."
        ) from e
    return weasyprint


def markdown_to_pdf(markdown: str, target: Path) -> None:
    """Write a PDF to target (atomically). Raises PdfUnavailable if it cannot."""
    weasyprint = load_weasyprint()
    WeasyCSS, HTML = weasyprint.CSS, weasyprint.HTML  # noqa: N806
    tmp = target.with_suffix(".pdf.tmp")
    try:
        html = HTML(string=markdown_to_html(markdown), url_fetcher=_deny_fetch)
        html.write_pdf(str(tmp), stylesheets=[WeasyCSS(string=CSS)])  # pyright: ignore[reportUnknownMemberType]
        tmp.replace(target)
    except Exception as e:  # noqa: BLE001  any rendering failure must not lose the markdown
        tmp.unlink(missing_ok=True)
        raise PdfUnavailable(f"PDF rendering failed: {e}") from e
