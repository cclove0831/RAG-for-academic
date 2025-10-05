import re
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .config import KEEP_FIRST_PAGES

NOISE_RE = re.compile(r"(acknowledg(e)?ments?|references?|funding|conflict of interest|author contributions?|ethics statement|data availability|appendix)", re.I)
KEEP_RE  = re.compile(r"(abstract|introduction|conclusion|discussion|results?|methods?)", re.I)

def section_aware_clean(docs: List[Document]) -> List[Document]:
    MIN_CHARS = 120
    bydoc = {}
    for d in docs:
        bydoc.setdefault(d.metadata.get("doc","NA"), []).append(d)
    cleaned = []
    for _, pages in bydoc.items():
        pages.sort(key=lambda x: int(x.metadata.get("page", 1e9)))
        keep = {int(p.metadata.get("page", 0)) for p in pages[:KEEP_FIRST_PAGES]}
        for d in pages:
            text = (d.page_content or "").strip()
            pg = int(d.metadata.get("page", 0))
            if pg in keep: cleaned.append(d); continue
            if len(text) < MIN_CHARS and not KEEP_RE.search(text[:300]): continue
            if NOISE_RE.search(text) and not KEEP_RE.search(text[:300]): continue
            cleaned.append(d)
    return cleaned

def split_docs(page_docs: List[Document]) -> List[Document]:
    sp = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = sp.split_documents(page_docs)
    for c in chunks:
        md = c.metadata or {}
        try: md["page"] = int(md.get("page", 0))
        except: md["page"] = 0
        c.metadata = md
    return chunks
