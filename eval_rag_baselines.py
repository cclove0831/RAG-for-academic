import argparse, time, statistics
from pathlib import Path
from collections import Counter

from rag_pipeline.config import (
    SEED, MAX_TOTAL_CTX, DENSE_K, BM25_K
)
from rag_pipeline.utils import set_seed
from rag_pipeline.io_data import load_pdfs
from rag_pipeline.cleaning import section_aware_clean, split_docs
from rag_pipeline.indexing import build_indexes
from rag_pipeline.eval_utils import (
    load_eval, pred_pairs_from_docs, has_citation
)
from rag_pipeline.llm_chain import build_chain
from dotenv import load_dotenv

# ---------- 基础检索器（朴素） ----------

def dense_retriever(vs):
    """最朴素 Dense：相似度检索，直接取前 k 个。"""
    ret = vs.as_retriever(search_type="similarity",
                          search_kwargs={"k": max(MAX_TOTAL_CTX, DENSE_K)})
    def retrieve(q, k=MAX_TOTAL_CTX):
        return ret.invoke(q)[:k]
    return retrieve

def bm25_retriever(bm25_chunk):
    """最朴素 BM25：直接取前 k 个。"""
    def retrieve(q, k=MAX_TOTAL_CTX):
        return bm25_chunk.invoke(q)[:k]
    return retrieve

# ---------- 构造两段式函数 + 单题缓存，避免生成阶段重复检索 ----------

def make_two_stage_fn(retrieve_fn):
    """
    返回 (two_stage_fn, cache)：
    - two_stage_fn(q) 在没有缓存时调用 retrieve_fn(q)，有缓存直接返回；
    - 你在外面先检索计时后，把 (cache["q"], cache["docs"]) 设好，
      chain.invoke(q) 时就不会再打检索。
    """
    cache = {"q": None, "docs": None}
    def two_stage_fn(q: str):
        if cache["q"] == q and cache["docs"] is not None:
            return cache["docs"]
        docs = retrieve_fn(q, k=MAX_TOTAL_CTX)
        cache["q"], cache["docs"] = q, docs
        return docs
    return two_stage_fn, cache

# ---------- 单个 baseline 的端到端评测 ----------

def run_naive_rag_baseline(name, retrieve_fn, questions):
    # two-stage + chain
    two_stage_fn, cache = make_two_stage_fn(retrieve_fn)
    chain = build_chain(two_stage_fn)  # 关键：把检索函数传给链

    rows = []
    for item in questions:
        q = item["question"]
        must = [t.lower() for t in item.get("must_include_any", [])]
        gold = item.get("gold", [])

        q_aug = (q + " " + " ".join(must)).strip()

        # ---- 检索计时（并写入缓存，供链复用）----
        t0 = time.time()
        docs = retrieve_fn(q_aug, k=MAX_TOTAL_CTX)
        cache["q"], cache["docs"] = q_aug, docs  # 让链里的 two_stage_fn 直接用这批 docs
        t1 = time.time()

        # ---- 生成计时（不再包含检索耗时）----
        t2 = time.time()
        ans = chain.invoke(q_aug)     # build_chain 内部会调用 two_stage_fn(q_aug)，但命中缓存
        t3 = time.time()

        # ---- 检索指标 ----
        pred_pairs = pred_pairs_from_docs(docs)          # [(doc,page), ...]
        pred_set = set(pred_pairs); pred_docs = {d for d,_ in pred_set}

        gold_pairs, doc_wild = set(), set()
        for g in gold:
            doc = g["doc"]; pages = g.get("pages", [])
            if pages: gold_pairs.update({(doc,int(p)) for p in pages})
            else: doc_wild.add(doc)

        hits = len(gold_pairs & pred_set) + len(doc_wild & pred_docs)
        denom = len(gold_pairs) + len(doc_wild)
        precision = hits / max(len(set(pred_pairs)), 1)
        recall = hits / denom if denom>0 else 1.0

        # ---- 生成侧小指标 ----
        cite_flag = has_citation(str(ans))

        rows.append({
            "retrieval_ms": (t1 - t0) * 1000.0,
            "gen_ms":       (t3 - t2) * 1000.0,
            "precision": precision,
            "recall":    recall,
            "has_cite":  cite_flag
        })

        if denom>0 and hits==0:
            print(f"[DEBUG miss] {q[:60]}... | pred={sorted(set(pred_pairs))} | gold_pairs={sorted(gold_pairs)} | gold_docs={sorted(doc_wild)}")

    if not rows:
        return {}

    def col(k): return [r[k] for r in rows]
    p50 = round(statistics.median(col("retrieval_ms")), 1)
    p95 = round(sorted(col("retrieval_ms"))[max(int(0.95*len(rows))-1, 0)], 1)
    gp50 = round(statistics.median(col("gen_ms")), 1)
    gp95 = round(sorted(col("gen_ms"))[max(int(0.95*len(rows))-1, 0)], 1)

    return {
        "n": len(rows),
        "retrieval_p50_ms": p50,
        "retrieval_p95_ms": p95,
        "generation_p50_ms": gp50,
        "generation_p95_ms": gp95,
        "avg_precision_at_k": round(sum(col("precision"))/len(rows), 3),
        "avg_recall_at_k":    round(sum(col("recall"))/len(rows), 3),
        "citation_rate":      round(sum(1 if x else 0 for x in col("has_cite"))/len(rows), 3),
    }

# ---------- 主入口 ----------

def main(data_dir: str=None, pdf: str=None, eval_path: str="eval/eval_questions.jsonl"):
    # 1) 数据 → 清洗 → 切分
    pages = load_pdfs(data_dir=Path(data_dir).resolve() if data_dir else None,
                      pdf_path=Path(pdf).resolve() if pdf else None)
    cleaned = section_aware_clean(pages)
    chunks = split_docs(cleaned)

    # 2) 建索引（与主流程一致）
    vs, hybrid, doc_bm25, page_bm25, reranker = build_indexes(chunks)

    # 3) 评测集
    questions = load_eval(Path(eval_path).resolve())

    # 4) 两条最朴素端到端 RAG 基线
    baselines = {
        "rag_dense_only": run_naive_rag_baseline(
            "rag_dense_only", retrieve_fn=dense_retriever(vs), questions=questions
        ),
        "rag_bm25_only": run_naive_rag_baseline(
            "rag_bm25_only", retrieve_fn=bm25_retriever(hybrid[1]), questions=questions
        ),
    }

    print("\n===== Naive End-to-End RAG Baselines =====")
    for name, metrics in baselines.items():
        print(f"\n-- {name} --")
        print(metrics)

if __name__ == "__main__":
    load_dotenv()
    set_seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--pdf", type=str, default=None)
    ap.add_argument("--eval", type=str, default="eval/eval_questions.jsonl")
    args = ap.parse_args()
    main(data_dir=args.data_dir, pdf=args.pdf, eval_path=args.eval)
