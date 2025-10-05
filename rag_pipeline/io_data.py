from pathlib import Path
from typing import List, Dict
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from .utils import slugify

def doc_id_from_path(p: Path, regex: str | None = r"^(paper_\d+)") -> str:
    stem = p.stem
    if regex:
        m = re.match(regex, stem, flags=re.I)
        if m: return m.group(1).lower()
    return slugify(stem)

def title_from_path(p: Path) -> str:
    m = re.match(r"^(paper_\d+)_", p.stem, flags=re.I)
    return p.stem[m.end():] if m else p.stem

def load_pdfs(data_dir: Path=None, pdf_path: Path=None, doc_id_regex: str | None = r"^(paper_\d+)") -> List[Document]:
    files = []
    if data_dir:
        files = sorted([f for f in data_dir.glob("*.pdf") if f.is_file()])
        if not files: raise FileNotFoundError(f"No PDFs under {data_dir}")
    elif pdf_path:
        if not pdf_path.is_file(): raise FileNotFoundError(f"PDF not found: {pdf_path}")
        files = [pdf_path]
    else:
        raise ValueError("Either data_dir or pdf must be provided.")
    pages: List[Document] = []
    for f in files:
        pid = doc_id_from_path(f, doc_id_regex)
        title = title_from_path(f)
        for d in PyPDFLoader(str(f)).load():
            md = dict(d.metadata or {})
            md.update({"doc": pid, "title": title, "source": str(f), "page": int(md.get("page", 0))})
            pages.append(Document(page_content=d.page_content, metadata=md))
    return pages
