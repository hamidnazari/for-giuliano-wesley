"""render_report: writes report.md inside the project folder. It takes no path argument."""

from collections.abc import Callable
from pathlib import Path

REPORT_NAME = "report.md"


def make_render_report(project_dir: Path) -> Callable[..., str]:
    root = project_dir.resolve()

    def render_report(markdown: str) -> str:
        target = (root / REPORT_NAME).resolve()
        if target.parent != root:  # cannot happen with a fixed name; kept as a sandbox guard
            raise PermissionError("report must stay inside the project folder")
        tmp = target.with_suffix(".tmp")
        tmp.write_text(markdown, encoding="utf-8")
        tmp.replace(target)  # atomic and idempotent: writing twice leaves one file
        return str(target)

    return render_report
