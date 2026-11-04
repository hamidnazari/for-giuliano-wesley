"""render_report: writes report.md and report.pdf inside the project folder.

It takes no path argument. The markdown is always written first, so a PDF problem (for example
missing pango) never loses the report; it is reported through on_pdf_error instead.
"""

from collections.abc import Callable
from pathlib import Path

from research_assistant.rendering.pdf import PdfUnavailable, markdown_to_pdf

REPORT_NAME = "report.md"
PDF_NAME = "report.pdf"


def make_render_report(
    project_dir: Path, on_pdf_error: Callable[[str], None] | None = None
) -> Callable[..., str]:
    root = project_dir.resolve()

    def render_report(markdown: str) -> str:
        target = (root / REPORT_NAME).resolve()
        pdf = (root / PDF_NAME).resolve()
        if target.parent != root or pdf.parent != root:  # sandbox guard; names are fixed
            raise PermissionError("report must stay inside the project folder")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(markdown, encoding="utf-8")
        tmp.replace(target)  # atomic and idempotent: writing twice leaves one file
        try:
            markdown_to_pdf(markdown, pdf)
        except PdfUnavailable as e:
            pdf.unlink(missing_ok=True)  # never leave a stale PDF next to a newer report
            if on_pdf_error:
                on_pdf_error(str(e))
        return str(target)

    return render_report
