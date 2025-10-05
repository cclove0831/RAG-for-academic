# rag_pipeline/retrieval.py
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import os
import re
import math

from langchain.schema import Document

# -------------------------
# 与您的 config.py 完全对齐导入
# -------------------------
from .config import (
    MAX_TOTAL_CTX,
    PER_PAGE_LIMIT,
    POOL_CAP,
    PER_DOC_PAGES,
    RERANK_CLIP,
    DENSE_FETCH_K,   # 用于 dense 召回切片
    BM25_FETCH_K,    # 用于 bm25_chunk 召回切片
    RERANK_TOPN,
    USE_NEIGHBOR_EXPAND,
    NEIGHBOR_EXPAND_TOP_M,
)
from .utils import clip

# -------------------------
# 一次性调试打印（首轮调用时输出）
# -------------------------
DBG_ONCE = {"printed": False}


def _dbg_once(msg: str):
    if not DBG_ONCE["printed"]:
        print("[DBG]", msg, flush=True)
        DBG_ONCE["printed"] = True


# -------------------------
# 轻量工具
# -------------------------
LIST_TOKENS = ("type", "types", "category", "categories", "类别", "种类", "类型")


def is_list_query(q: str) -> bool:
    ql = (q or "").lower()
    return any(tok in ql for tok in LIST_TOKENS)


def _fix_doc_meta(d: Document) -> Document:
    """若 metadata['doc'] 缺失/异常，用 source 文件名兜底为 paper_xxx。"""
    docid = d.metadata.get("doc", None)
    if docid in (None, "", "NA", "paper"):
        src = d.metadata.get("source", "")
        if src:
            stem = os.path.splitext(os.path.basename(src))[0]
            if stem:
                d.metadata["doc"] = stem
    return d


def fmt_docs(docs: Optional[List[Document]]) -> str:
    """调试用：把文档列表渲染成可读字符串。兼容 None/[]。"""
    if not docs:
        return ""
    rows = []
    for i, d in enumerate(docs, 1):
        pg = int(d.metadata.get("page", 0))
        docid = d.metadata.get("doc", "NA")
        src = d.metadata.get("source", "")
        rows.append(f"[{i}] ({docid}, p.{pg}) {(d.page_content or '').strip().replace(chr(10), ' ')}\n<source:{src}>")
    return "\n\n".join(rows)


# -------------------------
# 路由页 Top-1 的“宽松回退”：
# 先试 (doc,page) 精确匹配；失败则仅按 doc 取若干，再选与目标页最接近的 chunk。
# -------------------------
def _route_top1_by_doc_page(query: str, vectorstore, docid: str, pg: int) -> Optional[Document]:
    # 1) (doc,page) 精确匹配
    try:
        hits = vectorstore.similarity_search(query, k=1, filter={"doc": docid, "page": int(pg)})
        if hits:
            return hits[0]
    except Exception:
        pass

    # 2) 仅按 doc 放宽，取 3 条，挑与 pg 最近的页
    try:
        hits = vectorstore.similarity_search(query, k=3, filter={"doc": docid})
        if hits:
            def _gap(h: Document) -> int:
                try:
                    return abs(int(h.metadata.get("page", 10 ** 9)) - int(pg))
                except Exception:
                    return 10 ** 9

            hits.sort(key=_gap)
            return hits[0]
    except Exception:
        pass

    return None


# ------------------------------------------------------------
# 优化后的主流程：
# - 候选：并行混合检索（dense + bm25_chunk）
# - 页路由：对 dense 命中文档（+ query 显式提示的 doc）进行“中后段优先”的页路由
# - 重排：Cross-Encoder（chunk级，clip 后）
# - 选入：两阶段（先保 1 个深页路由，再常规配额）+ 文档约束提升 precision
# - 邻页补全：±2（可选）
# ------------------------------------------------------------
def two_stage_retrieve_multi(
        query: str,
        vectorstore,
        hybrid: Tuple,   # (dense_retriever, bm25_chunk)
        doc_bm25,       # 命名保留（未直接使用）
        page_bm25,      # 页路由用
        reranker,
        per_doc_pages: int = PER_DOC_PAGES,
        max_total: int = MAX_TOTAL_CTX,
        per_page_limit: int = PER_PAGE_LIMIT,
        pool_cap: int = POOL_CAP,
) -> List[Document]:
    dense, bm25_chunk = hybrid

    # （可选）列表型问句适度放宽每文档页数（仅当启用页路由时）
    P = per_doc_pages if per_doc_pages is not None else PER_DOC_PAGES
    if is_list_query(query) and P is not None and P > 0 and P < 5:
        P = 5

    # 统一解析 query 中显式点名的 doc，如 "paper_008"
    hinted_docs_lower = set(re.findall(r"(paper_\d+)", (query or ""), flags=re.IGNORECASE))
    hinted_docs_lower = {h.lower() for h in hinted_docs_lower}

    # ---- 调试计数 ----
    routed_added_to_pool = 0
    routed_in_rerank_topN = 0
    routed_in_final = 0
    routed_keys: set = set()  # {(doc, page)}

    # 1) 并行混合检索，构建初始候选池
    pool: List[Document] = []
    seen = set()

    def add(doc: Document):
        doc = _fix_doc_meta(doc)
        # 软过滤：过短页（多为图注/目录/噪声）；如需更宽松可调低阈值或移除
        if len((doc.page_content or "").strip()) < 200:
            return
        # 使用更鲁棒的去重键
        key = (str(doc.metadata.get("doc")).lower(), int(doc.metadata.get("page", -1)), (doc.page_content or "").strip()[:96])
        if key in seen:
            return
        seen.add(key)
        pool.append(doc)

    # 1.1) Dense (向量) 召回
    try:
        dense_raw = dense.invoke(query)[:DENSE_FETCH_K]
    except Exception:
        dense_raw = []

    # 1.2) BM25-Chunk (关键词) 召回
    try:
        bm25_raw = bm25_chunk.invoke(query)[:BM25_FETCH_K]
    except Exception:
        bm25_raw = []

    # 1.3) 合并与去重（先 dense 再 bm25，按内容摘要避免重复）
    seen_content_digest = set()
    for d in dense_raw:
        content_digest = (d.page_content or "").strip()[:256]
        seen_content_digest.add(content_digest)
        add(d)
    for d in bm25_raw:
        content_digest = (d.page_content or "").strip()[:256]
        if content_digest not in seen_content_digest:
            add(d)

    # 收集 dense 命中的文档集合（页路由靶向）
    dense_docs = {str(_fix_doc_meta(d).metadata.get("doc", "NA")).lower() for d in dense_raw}

    # 2) 页路由（page_bm25）：对 dense 命中文档 + query 明示 doc 的文档，各取前 P 个“中后段优先”的页
    if page_bm25 is not None and P is not None and P > 0:
        route_target_docs = set(dense_docs) | set(hinted_docs_lower)

        # --- 收集（doc, page, rank） ---
        per_doc_pg_rank: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        try:
            page_hits = page_bm25.invoke(query)  # 若支持 k，可改为 k=60~80
        except Exception:
            page_hits = []

        for idx, d in enumerate(page_hits):
            d = _fix_doc_meta(d)
            docid = str(d.metadata.get("doc", "NA")).lower()
            if docid not in route_target_docs:
                continue
            txt = (d.page_content or "")
            low = txt.lower()
            # 过滤噪声/非内容页
            if len(txt) < 200 or any(t in low for t in (
                "references", "bibliography", "acknowledg", "table of contents", "目录", "致谢", "参考文献"
            )):
                continue
            try:
                pg = int(d.metadata.get("page", 0))
            except Exception:
                continue
            per_doc_pg_rank[docid].append((pg, idx))  # idx 即 BM25 全局名次（越小越好）

        # --- 用“名次 + 页号偏好”的联合打分，选每个 doc 的前 P 页 ---
        per_doc_pages_map: Dict[str, List[int]] = defaultdict(list)
        ALPHA = 3.0         # 页号偏好强度：越大越偏好深页
        EARLY_PENALTY = 6.0 # 对 p.0/p.1 的额外惩罚（避免总是拿导言页）
        for docid, lst in per_doc_pg_rank.items():
            scored: List[Tuple[float, int]] = []
            for (pg, rank) in lst:
                # 分数越小越好：rank 越小越好；pg 越大越好（log 调整）；p.0/p.1 额外惩罚
                score = rank - ALPHA * math.log(pg + 2.0)
                if pg <= 1:
                    score += EARLY_PENALTY
                scored.append((score, pg))
            scored.sort(key=lambda x: x[0])
            chosen = [pg for _, pg in scored[:int(P)]]
            per_doc_pages_map[docid] = chosen

        # --- 对路由到的 (doc,page) 获取该页的 Top-1 chunk（带宽松回退） ---
        for docid, pages in per_doc_pages_map.items():
            for pg in pages:
                cand = _route_top1_by_doc_page(query, vectorstore, docid, pg)
                if not cand:
                    continue
                add(cand)
                key = (str(_fix_doc_meta(cand).metadata.get("doc", "NA")).lower(), int(cand.metadata.get("page", -1)))
                if key not in routed_keys:
                    routed_keys.add(key)
                    routed_added_to_pool += 1

    # 3) 候选池上限
    if len(pool) > pool_cap:
        pool = pool[:pool_cap]

    if not pool:
        _dbg_once(f"per_doc_pages={per_doc_pages}  pool=0  topN=0  final=0 | routed: pool+=0, topN=0, final=0")
        return []

    # 4) CrossEncoder 重排（chunk级，clip 后）
    try:
        reranked = reranker.compress_documents(
            query=query,
            documents=[Document(page_content=clip(d.page_content, RERANK_CLIP), metadata=d.metadata) for d in pool]
        )[:RERANK_TOPN]
    except Exception:
        reranked = pool[:RERANK_TOPN]

    # 若 query 显式点名了 doc，则只保留这些 doc 的候选，精准提升 precision
    if hinted_docs_lower:
        reranked_only_hinted = []
        for dd in reranked:
            docid = str(_fix_doc_meta(dd).metadata.get("doc", "NA")).lower()
            if docid in hinted_docs_lower:
                reranked_only_hinted.append(dd)
        if reranked_only_hinted:
            reranked = reranked_only_hinted  # 有保留项才替换

    # 统计：路由页进入 TopN 的数量
    routed_in_rerank_topN = sum(
        ( (str(_fix_doc_meta(dd).metadata.get("doc", "NA")).lower(), int(dd.metadata.get("page", -1))) in routed_keys )
        for dd in reranked
    )

    # 5) 两阶段选入：
    #    Phase-1: 先保 1 个“深页路由”（页>=2）坑位；
    #    Phase-2: 常规配额（含路由配额 + 文档约束）。
    final: List[Document] = []
    per_page_count: Dict[Tuple[str, int], int] = defaultdict(int)

    # === Phase-1: 先放 1 个“深页路由”候选（页>=2） ===
    deep_routed_added = 0
    for d in reranked:
        d = _fix_doc_meta(d)
        key = (str(d.metadata.get("doc", "NA")).lower(), int(d.metadata.get("page", -1)))
        is_routed = key in routed_keys
        pg = key[1]
        if not is_routed or pg < 2:
            continue
        if per_page_count[key] >= per_page_limit:
            continue
        final.append(d)
        per_page_count[key] += 1
        deep_routed_added = 1
        break  # 只保 1 个坑位

    # === Phase-2: 常规配额（含路由配额 + 文档约束） ===
    ROUTED_FINAL_CAP = 2  # 给路由页多一点生存空间
    routed_final_cnt = 1 if deep_routed_added else 0

    # 定义“主文档”：dense 或 bm25 召回过，或被路由过（信号更强，优先保真）
    main_docs = set()
    main_docs |= {str(_fix_doc_meta(d).metadata.get("doc", "NA")).lower() for d in dense_raw}
    main_docs |= {str(_fix_doc_meta(d).metadata.get("doc", "NA")).lower() for d in bm25_raw}
    main_docs |= {doc for (doc, _pg) in routed_keys}

    other_doc_final_cnt = 0
    MAX_OTHER_DOCS_IN_FINAL = 1  # 未点名 doc 时，其它文档最多 1 条

    for d in reranked:
        d = _fix_doc_meta(d)
        doc_page_key = (str(d.metadata.get("doc", "NA")).lower(), int(d.metadata.get("page", -1)))
        is_routed = doc_page_key in routed_keys

        # 显式点名：强约束只允许点名文档进入
        if hinted_docs_lower and (doc_page_key[0] not in hinted_docs_lower):
            continue

        if is_routed and routed_final_cnt >= ROUTED_FINAL_CAP:
            continue

        # 未点名：限制“非主文档”的名额，提升整体 precision
        if not hinted_docs_lower:
            current_doc = doc_page_key[0]
            if current_doc not in main_docs and other_doc_final_cnt >= MAX_OTHER_DOCS_IN_FINAL:
                continue

        if per_page_count[doc_page_key] >= per_page_limit:
            continue

        # 去重：若 Phase-1 已放入相同 (doc,page)，跳过
        if any(
            (str(x.metadata.get("doc", "NA")).lower() == doc_page_key[0]) and (int(x.metadata.get("page", -1)) == doc_page_key[1])
            for x in final
        ):
            continue

        final.append(d)
        per_page_count[doc_page_key] += 1
        if is_routed:
            routed_final_cnt += 1
        if not hinted_docs_lower and (doc_page_key[0] not in main_docs):
            other_doc_final_cnt += 1

        if len(final) >= max_total:
            break

    # 6) 邻页补全（可选；±2；种子≥3）
    if final and USE_NEIGHBOR_EXPAND and NEIGHBOR_EXPAND_TOP_M is not None and NEIGHBOR_EXPAND_TOP_M >= 0:
        seen_keys = {(str(d.metadata.get("doc", "NA")).lower(), int(d.metadata.get("page", -1))) for d in final}
        expanded: List[Document] = []
        seeds = list(final[:min(len(final), max(NEIGHBOR_EXPAND_TOP_M, 3))])

        for d in seeds:
            docid = str(d.metadata.get("doc", "NA")).lower()
            try:
                page = int(d.metadata.get("page", -1))
            except Exception:
                continue

            for nb in (page - 2, page - 1, page + 1, page + 2):
                if nb < 0:
                    continue
                key = (docid, nb)
                if key in seen_keys:
                    continue
                try:
                    hits = vectorstore.similarity_search(query, k=1, filter={"doc": docid, "page": int(nb)})
                    cand = hits[0] if hits else None
                except Exception:
                    cand = None

                if not cand:
                    continue

                current_page_count = per_page_count[key]
                if current_page_count >= per_page_limit:
                    continue
                if len(final) + len(expanded) >= max_total:
                    break

                expanded.append(_fix_doc_meta(cand))
                per_page_count[key] += 1
                seen_keys.add(key)

        if expanded:
            final.extend(expanded)

    # 统计：路由页进入 final 的数量（含邻页扩展后的最终列表）
    routed_in_final = sum(
        ( (str(_fix_doc_meta(dd).metadata.get("doc", "NA")).lower(), int(dd.metadata.get("page", -1))) in routed_keys )
        for dd in final
    )

    # 一次性打印关键统计
    _dbg_once(
        f"per_doc_pages={per_doc_pages}  pool={len(pool)}  topN={len(reranked)}  final={len(final)} | "
        f"routed: pool+={routed_added_to_pool}, topN={routed_in_rerank_topN}, final={routed_in_final}"
    )

    return final if final else []
