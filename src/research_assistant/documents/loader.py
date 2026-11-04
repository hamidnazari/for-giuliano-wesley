"""Read the user's own files into plain text pages."""

from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

SUPPORTED = {".md", ".markdown", ".txt", ".pdf"}


@dataclass(frozen=True)
class Page:
    doc: str  # path relative to the indexed folder
    page: int | None  # 1-based for PDFs, None for plain text
    text: str


@dataclass
class Loaded:
    pages: list[Page] = field(default_factory=lambda: list[Page]())
    skipped: list[str] = field(default_factory=lambda: list[str]())  # unsupported or unreadable


def _files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))


def load_documents(root: Path) -> Loaded:
    if not root.exists():
        raise FileNotFoundError(root)
    base = root.parent if root.is_file() else root
    out = Loaded()
    for path in _files(root):
        rel = str(path.relative_to(base))
        if path.suffix.lower() not in SUPPORTED:
            out.skipped.append(f"{rel} (unsupported type)")
            continue
        try:
            if path.suffix.lower() == ".pdf":
                for n, pdf_page in enumerate(PdfReader(path).pages, start=1):
                    text = (pdf_page.extract_text() or "").strip()
                    if text:
                        out.pages.append(Page(rel, n, text))
            else:
                text = path.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    out.pages.append(Page(rel, None, text))
        except Exception as e:  # a broken file must not stop the whole index
            out.skipped.append(f"{rel} (unreadable: {type(e).__name__})")
    return out
