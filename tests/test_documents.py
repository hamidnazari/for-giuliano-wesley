import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import research_assistant.cli as cli
from research_assistant.agents.factory import AgentModel
from research_assistant.agents.researcher import LLMResearcher
from research_assistant.documents.chunker import chunk_text
from research_assistant.documents.index import DocIndex, Embedder, HashingEmbedder, IndexUnusable
from research_assistant.documents.loader import load_documents
from research_assistant.guardrails.budget import Budget
from research_assistant.models import Card, CardStatus
from research_assistant.observability.audit import AuditLog
from research_assistant.settings import load_settings
from research_assistant.sources import SourceStore
from research_assistant.tools.docs import make_docs_retrieve
from research_assistant.tools.registry import Tool, ToolRegistry

CONFIG = Path(__file__).parent.parent / "config"


def make_pdf(path: Path, pages: list[str]) -> None:
    """A tiny valid PDF with one line of text per page."""
    objs: list[str] = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>")
    font = 3 + 2 * len(pages)
    for i, text in enumerate(pages):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {4 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
        )
        objs.append(f"<< /Length {len(content)} >>\nstream\n{content}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = "%PDF-1.4\n"
    offsets: list[int] = []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


# ---- chunker ----


def test_chunks_respect_size_overlap_and_cover_the_text() -> None:
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, 200, 40)
    assert all(0 < len(c) <= 200 for c in chunks) and len(chunks) > 5
    assert chunks[0].startswith("word0") and chunks[-1].endswith("word399")
    for a, b in zip(chunks, chunks[1:], strict=False):  # consecutive chunks overlap
        assert set(a.split()[-3:]) & set(b.split()[:8])
    joined = " ".join(chunks)
    assert all(f"word{i}" in joined for i in range(400))


def test_chunker_handles_edge_cases() -> None:
    assert chunk_text("", 100, 10) == [] and chunk_text("   \n ", 100, 10) == []
    assert chunk_text("short", 100, 10) == ["short"]
    blob = "x" * 1000  # no whitespace to break on
    assert all(len(c) <= 100 for c in chunk_text(blob, 100, 20))
    with pytest.raises(ValueError):
        chunk_text("a", 10, 10)


# ---- loader ----


def test_loader_reads_md_txt_pdf_and_skips_others(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.md").write_text("# Title\n\nAlpha text.")
    (tmp_path / "sub" / "b.txt").write_text("Beta text.")
    (tmp_path / ".hidden.md").write_text("ignored")
    (tmp_path / "pic.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("   ")
    make_pdf(tmp_path / "c.pdf", ["Gamma page one", "Delta page two"])
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    got = load_documents(tmp_path)
    by = {(p.doc, p.page): p.text for p in got.pages}
    assert by[("a.md", None)].endswith("Alpha text.") and by[("sub/b.txt", None)] == "Beta text."
    assert "Gamma page one" in by[("c.pdf", 1)] and "Delta page two" in by[("c.pdf", 2)]
    assert not any(d == ".hidden.md" or d == "empty.md" for d, _ in by)
    assert any("pic.png (unsupported" in s for s in got.skipped)
    assert any("broken.pdf (unreadable" in s for s in got.skipped)


def test_loader_single_file_and_missing_path(tmp_path: Path) -> None:
    f = tmp_path / "one.md"
    f.write_text("hello")
    assert [p.doc for p in load_documents(f).pages] == ["one.md"]
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "nope")


# ---- index ----


def build(tmp_path: Path) -> DocIndex:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "vec.md").write_text(
        "Vector databases store embeddings and run nearest neighbour search."
    )
    (tmp_path / "docs" / "sql.md").write_text(
        "Relational databases store rows in tables and use SQL joins."
    )
    (tmp_path / "docs" / "cook.txt").write_text("Bake the bread at high heat for forty minutes.")
    idx, skipped = DocIndex.build(tmp_path / "docs", HashingEmbedder(), 200, 20)
    assert skipped == []
    return idx


def test_search_ranks_the_relevant_chunk_first(tmp_path: Path) -> None:
    idx = build(tmp_path)
    hit = idx.search("how do embeddings and nearest neighbour search work", k=2)[0]
    assert hit.chunk.doc == "vec.md" and hit.chunk.chunk_id == "vec.md#c1" and hit.score > 0.3
    assert idx.search("SQL joins and tables", k=1)[0].chunk.doc == "sql.md"
    assert idx.search("bread", k=3, min_score=0.3)[0].chunk.doc == "cook.txt"
    assert idx.search("zzz qqq", k=3, min_score=0.5) == []


def test_pdf_chunks_carry_page_numbers(tmp_path: Path) -> None:
    make_pdf(tmp_path / "r.pdf", ["Quantum computing uses qubits", "Bananas are yellow fruit"])
    idx, _ = DocIndex.build(tmp_path, HashingEmbedder(), 200, 20)
    hit = idx.search("qubits", k=1)[0]
    assert hit.chunk.page == 1 and hit.chunk.title == "r.pdf (page 1)"


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    idx = build(tmp_path)
    idx.save(tmp_path / "index")
    loaded = DocIndex.load(tmp_path / "index", HashingEmbedder())
    assert loaded.chunks == idx.chunks
    assert loaded.search("bread", k=1)[0].chunk.doc == "cook.txt"
    assert loaded.meta["model"] == "hashing-test" and loaded.meta["documents"] == 3


def test_load_rejects_wrong_model_and_damaged_index(tmp_path: Path) -> None:
    build(tmp_path).save(tmp_path / "index")

    class Other(HashingEmbedder):
        name = "another-model"

    with pytest.raises(IndexUnusable, match="another-model"):
        DocIndex.load(tmp_path / "index", Other())
    (tmp_path / "index" / "vectors.npy").unlink()
    with pytest.raises(IndexUnusable, match="cannot read"):
        DocIndex.load(tmp_path / "index", HashingEmbedder())
    meta = json.loads((tmp_path / "index" / "meta.json").read_text())
    meta["version"] = 99
    (tmp_path / "index" / "meta.json").write_text(json.dumps(meta))


# ---- the tool and the researcher ----


def test_docs_retrieve_tool(tmp_path: Path) -> None:
    idx = build(tmp_path)
    out = make_docs_retrieve(idx, 2, 0.0)(query="bread")
    assert out[0]["doc"] == "cook.txt" and {"chunk_id", "page", "text", "score"} <= out[0].keys()
    assert make_docs_retrieve(None, 2, 0.0)(query="bread") == []


def test_source_store_doc_ids_share_the_card_counter() -> None:
    store = SourceStore()
    a = store.add_doc("c1", "vec.md#c1", "vec.md", "text")
    b = store.add_doc("c1", "vec.md#c1", "vec.md", "text")
    c = store.add_doc("c1", "sql.md#c1", "sql.md", "text")
    assert (a.source_id, b.source_id, c.source_id) == ("c1-s1", "c1-s1", "c1-s2")
    assert a.kind == "doc" and a.locator == "vec.md#c1" and a.snippet == "text"


def test_researcher_cites_documents_alongside_the_web(tmp_path: Path) -> None:
    settings = load_settings(CONFIG)
    audit = AuditLog(tmp_path / "audit.jsonl", "r1", secrets=[])
    budget = Budget(settings.guardrails)
    idx = build(tmp_path)
    reg = ToolRegistry(settings, audit)

    def web(query: str) -> list[dict[str, Any]]:
        return [{"url": "https://w.example/1", "title": "Web", "content": "web text", "score": 0.5}]

    def noop(**_: Any) -> list[Any]:
        return []

    reg.register(Tool("web_search", web))
    reg.register(Tool("docs_retrieve", make_docs_retrieve(idx, 1, 0.0)))
    reg.register(Tool("fetch_cited_passage", noop))
    reg.register(Tool("render_report", noop))
    replies = [
        json.dumps({"queries": ["embeddings search"]}),
        json.dumps(
            {
                "notes": [
                    {"text": "Vectors support nearest neighbour search.", "source_ids": ["c1-s2"]},
                    {"text": "Web view.", "source_ids": ["c1-s1"]},
                ],
                "credibility": {"c1-s2": 0.6, "c1-s1": 0.5},
            }
        ),
    ]
    fake = GenericFakeChatModel(messages=iter([AIMessage(content=r) for r in replies]))
    model = AgentModel("researcher", "anthropic/claude-haiku-4.5", fake, budget, audit)
    researcher = LLMResearcher(
        model, "SYS", reg.for_agent("researcher"), SourceStore(), audit, 6, use_documents=True
    )
    card = Card(id="c1", sub_question="How?", status=CardStatus.IN_PROGRESS)
    work = researcher.research(card)
    kinds = {s.source_id: s.kind for s in work.sources}
    assert kinds == {"c1-s1": "web", "c1-s2": "doc"}
    doc = next(s for s in work.sources if s.kind == "doc")
    assert doc.locator == "vec.md#c1" and "nearest neighbour" in doc.snippet
    assert budget.searches == 0  # web_search here is a fake; the local document lookup is free


# ---- CLI ----


def hashing(_name: str) -> Embedder:
    return HashingEmbedder()


@pytest.fixture
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_embedder", hashing)


def test_cli_index_builds_and_reports(
    tmp_path: Path, fake_embedder: None, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Alpha beta gamma.")
    (docs / "x.png").write_bytes(b"png")
    code = cli.main(["index", str(docs), "--config", str(CONFIG), "--data", str(tmp_path / "data")])
    out = capsys.readouterr()
    assert code == 0 and "indexed 1 chunks from 1 documents" in out.out
    assert "x.png (unsupported" in out.err and (tmp_path / "data" / "index" / "meta.json").exists()


def test_cli_index_errors(
    tmp_path: Path, fake_embedder: None, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--config", str(CONFIG), "--data", str(tmp_path / "data")]
    assert cli.main(["index", str(tmp_path / "missing"), *argv]) == 1
    assert "not found" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["index", str(empty), *argv]) == 1
    assert "nothing to index" in capsys.readouterr().err


def test_cli_run_refuses_an_index_built_with_another_model(
    tmp_path: Path,
    fake_embedder: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Alpha beta gamma.")
    data = tmp_path / "data"
    cli.main(["index", str(docs), "--config", str(CONFIG), "--data", str(data)])
    cassette = tmp_path / "c.json"
    cassette.write_text(json.dumps({"version": 1, "entries": {}}))

    class Bge(HashingEmbedder):
        name = "BAAI/bge-small-en-v1.5"

    def bge(_name: str) -> Embedder:
        return Bge()

    monkeypatch.setattr(cli, "make_embedder", bge)
    code = cli.main(
        ["run", "Q", "--auto-approve", "--real", "researcher", "--search", "replay",
         "--cassette", str(cassette), "--config", str(CONFIG), "--data", str(data)]
    )  # fmt: skip
    assert code == 1 and "run the index command again" in capsys.readouterr().err
