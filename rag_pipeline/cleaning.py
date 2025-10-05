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

def sanitize_metadata(d: Document) -> Document:
    """确保每个 chunk 都有规范的 doc / page，避免后续评测出现 ('NA', …) 或 ('paper', …) 之类的异常。"""
    md = dict(d.metadata or {})

    # doc 兜底：优先已有 doc；否则从 file_name / source 中提取；再不行用 'NA'
    doc = md.get("doc")
    if not doc:
        src = (md.get("file_name") or md.get("source") or "").strip()
        if src:
            import os, re
            stem = os.path.splitext(os.path.basename(src))[0]
            # 若文件名以 paper_数字 开头，优先取这个（与你评测集的 gold 对齐）
            m = re.match(r"^(paper_\d+)", stem, flags=re.I)
            doc = m.group(1).lower() if m else stem.lower()
        else:
            doc = "NA"
    md["doc"] = doc

    # page 兜底：强制为 int；失败则 -1
    try:
        page = int(md.get("page", -1))
    except Exception:
        page = -1
    md["page"] = page

    d.metadata = md
    return d

def split_docs(page_docs: List[Document]) -> List[Document]:
    sp = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = sp.split_documents(page_docs)
    for c in chunks:
        md = c.metadata or {}
        try:
            md["page"] = int(md.get("page", 0))
        except:
            md["page"] = 0
        c.metadata = md
        # === 新增：把 docid 与页码永久写入文本前缀 ===
        docid = str(md.get("doc", "NA"))
        page = int(md.get("page", 0))
        prefix = f"[{docid} p.{page}] "
        if not c.page_content.startswith(prefix):
            c.page_content = prefix + (c.page_content or "").strip()
    return chunks

