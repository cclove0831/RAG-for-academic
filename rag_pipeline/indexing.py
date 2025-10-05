from pathlib import Path
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
import torch
from langchain_community.cross_encoders import HuggingFaceCrossEncoder as LCHFCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.schema import Document
from .config import (
    EMB_MODEL_NAME, DENSE_K, BM25_K, MMR_FETCH_K, MMR_LAMBDA, RERANK_TOPN,
    PAGE_CLIP_CHARS, DOC_ROUTE_K, PERSIST_DIRNAME,DEVICE,CE_DEVICE,EMB_BATCH)

def build_doc_bm25_from_chunks(chunks: List[Document]) -> BM25Retriever:
    # 文档级BM25（按页排序，拼前几段）
    bydoc = {}
    for d in chunks: bydoc.setdefault(d.metadata.get("doc","NA"), []).append(d)
    docs: List[Document] = []
    for docid, items in bydoc.items():
        items.sort(key=lambda x: int(x.metadata.get("page", 1e9)))
        buf, total = [], 0
        for s in items:
            seg = (s.page_content or "").strip()
            if seg: buf.append(seg[:800]); total += len(seg)
            if len(buf) >= 3 or total >= 3000: break
        title = items[0].metadata.get("title",""); src = items[0].metadata.get("source","")
        content = (title + "\n" + "\n".join(buf))[:4000] if buf else title
        docs.append(Document(page_content=content, metadata={"doc": docid, "title": title, "source": src}))
    r = BM25Retriever.from_documents(docs); r.k = DOC_ROUTE_K
    return r

def build_indexes(chunks: List[Document]):
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL_NAME,
                                model_kwargs={"device":DEVICE},
                                encode_kwargs={"normalize_embeddings": True,"batch_size": EMB_BATCH})
    vs = Chroma.from_documents(chunks, emb, persist_directory=str(Path(PERSIST_DIRNAME)))
    dense = vs.as_retriever(search_type="mmr",
                            search_kwargs={"k": DENSE_K, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA})
    bm25_chunk = BM25Retriever.from_documents(chunks); bm25_chunk.k = BM25_K
    hybrid = (dense, bm25_chunk)

    # 页级BM25（截断到 PAGE_CLIP_CHARS）
    page_docs = [Document(page_content=(d.page_content or "")[:PAGE_CLIP_CHARS], metadata=d.metadata) for d in chunks]
    page_bm25 = BM25Retriever.from_documents(page_docs); page_bm25.k = 100

    dev = CE_DEVICE
    if dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"
    lc_ce = LCHFCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        model_kwargs={"device": dev}
    )
    reranker = CrossEncoderReranker(model=lc_ce, top_n=RERANK_TOPN)
    doc_bm25 = build_doc_bm25_from_chunks(chunks)
    return vs, hybrid, doc_bm25, page_bm25, reranker
