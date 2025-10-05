from typing import List, Dict
from collections import defaultdict
import os
from langchain.schema import Document
from .config import (
    MAX_TOTAL_CTX,
    PER_PAGE_LIMIT,
    POOL_CAP,
    PER_DOC_PAGES,
    RERANK_CLIP,
)
from .utils import clip


LIST_TOKENS = ("type", "types", "category", "categories", "类别", "种类", "类型")
def is_list_query(q: str) -> bool:
    ql = (q or "").lower()
    return any(tok in ql for tok in LIST_TOKENS)

def _fix_doc_meta(d: Document) -> Document:
    """若 doc 缺失/异常，用 source 文件名兜底恢复为 paper_xxx。"""
    docid = d.metadata.get("doc", None)
    if docid in (None, "", "NA", "paper"):
        src = d.metadata.get("source", "")
        if src:
            stem = os.path.splitext(os.path.basename(src))[0]
            if stem:
                d.metadata["doc"] = stem
    return d

def fmt_docs(docs: List[Document]) -> str:
    rows = []
    for i, d in enumerate(docs, 1):
        pg = int(d.metadata.get("page", 0))
        docid = d.metadata.get("doc", "NA")
        src = d.metadata.get("source", "")
        rows.append(f"[{i}] ({docid}, p.{pg}) {d.page_content.strip().replace(chr(10),' ')}\n<source:{src}>")
    return "\n\n".join(rows)

# ------------------------------------------------------------
# 双索引混合检索：doc-BM25 路由 + page-BM25 页挑选
# 候选池：dense & bm25_chunk → 页内 Top-1 → 邻页预取
# 重排：交叉编码器 → 邻页补全（兜底现取）
# ------------------------------------------------------------
def two_stage_retrieve_multi(
    query: str,
    vectorstore,
    hybrid,            # (dense_retriever, bm25_chunk)
    doc_bm25,
    page_bm25,
    reranker,
    per_doc_pages: int = PER_DOC_PAGES,
    max_total: int = MAX_TOTAL_CTX,
    per_page_limit: int = PER_PAGE_LIMIT,
    pool_cap: int = POOL_CAP,
) -> List[Document]:

    dense, bm25_chunk = hybrid

    # 每文档可选页数：列表型问句至少取 4 页
    P = per_doc_pages if per_doc_pages is not None else PER_DOC_PAGES
    if is_list_query(query) and P < 4:
        P = 4

    # 1) 文档级路由（doc BM25）
    try:
        allowed_docs_bm25 = [d.metadata.get("doc", "NA") for d in doc_bm25.invoke(query)]
    except Exception:
        allowed_docs_bm25 = []

    # 2) 页级路由（page BM25）：对路由到的文档，各取前 P 页
    per_doc_pages_map: Dict[str, List[int]] = defaultdict(list)
    try:
        all_hits = page_bm25.invoke(query)
    except Exception:
        all_hits = []

    for d in all_hits:
        d = _fix_doc_meta(d)
        docid = d.metadata.get("doc", "NA")
        # 若有路由文档，先限定在路由集内
        if allowed_docs_bm25 and (docid not in allowed_docs_bm25):
            continue
        if len(per_doc_pages_map[docid]) >= P:
            continue
        pg = int(d.metadata.get("page", 0))
        if pg not in per_doc_pages_map[docid]:
            per_doc_pages_map[docid].append(pg)
        if allowed_docs_bm25 and all(len(per_doc_pages_map[x]) >= P for x in allowed_docs_bm25):
            break

    # 3) 候选池：dense + bm25_chunk 原始候选 → 页内 Top-1 → 邻页预取
    pool, seen = [], set()
    def add(doc: Document):
        doc = _fix_doc_meta(doc)
        key = (doc.metadata.get("doc"), int(doc.metadata.get("page", -1)), (doc.page_content or "")[:96])
        if key in seen:
            return
        seen.add(key)
        pool.append(doc)

    # 3.1 hybrid 原始候选（不先过滤 allowed_docs，避免路由漏召回）
    try:
        dense_raw = dense.invoke(query)[:16]
    except Exception:
        dense_raw = []
    try:
        bm25c_raw = bm25_chunk.invoke(query)[:16]
    except Exception:
        bm25c_raw = []

    for d in dense_raw + bm25c_raw:
        add(d)

    # 3.2 合并“允许文档”：doc-BM25 ∪ hybrid 现有候选出现过的文档（保持顺序去重）
    route_docs_hyb = [d.metadata.get("doc", "NA") for d in pool]
    allowed_docs = list(dict.fromkeys((allowed_docs_bm25 or []) + route_docs_hyb))

    # 3.3 页内 Top-1（针对路由到的页）
    for docid, pages in per_doc_pages_map.items():
        for pg in pages:
            try:
                hits = vectorstore.similarity_search(query, k=1, filter={"doc": docid, "page": int(pg)})
                for d in hits:
                    add(d)
            except Exception:
                pass

    # 3.4 邻页预取（page±1 的 Top-1）
    for docid, pages in per_doc_pages_map.items():
        for pg in pages:
            for nb in (pg - 1, pg + 1):
                if nb < 0:
                    continue
                try:
                    hits = vectorstore.similarity_search(query, k=1, filter={"doc": docid, "page": int(nb)})
                    for d in hits:
                        add(d)
                except Exception:
                    pass

    # 3.5 池大小限制
    if len(pool) > pool_cap:
        pool = pool[:pool_cap]

    # 4) 交叉编码器重排
    if not pool:
        return []
    try:
        reranked = reranker.compress_documents(
            query=query,
            documents=[Document(page_content=clip(d.page_content, RERANK_CLIP), metadata=d.metadata) for d in pool]
        )
    except Exception:
        reranked = []
    if not reranked:
        return []

    # 4.1 先按“每页≤PER_PAGE_LIMIT，整体≤MAX_TOTAL_CTX”装入第一批
    final, per_page = [], defaultdict(int)
    for d in reranked:
        d = _fix_doc_meta(d)
        k = (d.metadata.get("doc", "NA"), int(d.metadata.get("page", -1)))
        if per_page[k] >= per_page_limit:
            continue
        final.append(d)
        per_page[k] += 1
        if len(final) >= max_total:
            break

    # 4.2 邻页补全：对已入选 (doc,page) 各补 (page-1, page+1) 的首条
    #     若 pool 中没有该邻页，则兜底向量检索 1 条；严格遵守限流
    if final:
        seen_keys = {(d.metadata.get("doc", "NA"), int(d.metadata.get("page", -1))) for d in final}

        # 预建 pool 索引，便于 O(1) 找邻页
        pool_index = defaultdict(list)
        for cand in pool:
            cand = _fix_doc_meta(cand)
            key = (cand.metadata.get("doc", "NA"), int(cand.metadata.get("page", -1)))
            pool_index[key].append(cand)

        expanded = []
        for d in list(final):
            docid = d.metadata.get("doc", "NA")
            try:
                page = int(d.metadata.get("page", -1))
            except Exception:
                continue

            for nb in (page - 1, page + 1):
                if nb < 0:
                    continue
                key = (docid, nb)

                # 优先用 pool 中的邻页；没有则兜底取 1 条
                cand = None
                if key in pool_index and pool_index[key]:
                    cand = pool_index[key][0]
                else:
                    try:
                        hits = vectorstore.similarity_search(query, k=1, filter={"doc": docid, "page": int(nb)})
                        if hits:
                            cand = hits[0]
                    except Exception:
                        cand = None

                if cand is None or key in seen_keys:
                    continue
                if len(final) + len(expanded) >= max_total:
                    break
                if per_page[key] >= per_page_limit:
                    continue

                cand = _fix_doc_meta(cand)
                expanded.append(cand)
                per_page[key] += 1
                seen_keys.add(key)

        if expanded:
            final.extend(expanded)

    return final
