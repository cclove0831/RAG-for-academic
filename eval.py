import argparse, time, re
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline.utils import set_seed
from rag_pipeline.io_data import load_pdfs
from rag_pipeline.cleaning import section_aware_clean, split_docs
from rag_pipeline.indexing import build_indexes
from rag_pipeline.retrieval import two_stage_retrieve_multi
from rag_pipeline.llm_chain import build_chain
from rag_pipeline.eval_utils import load_eval, pred_pairs_from_docs, has_citation, summarize
from rag_pipeline.config import SEED, MAX_TOTAL_CTX

def main(data_dir: str=None, pdf: str=None, eval_path: str="eval/eval_questions.jsonl"):
    pages = load_pdfs(data_dir=Path(data_dir).resolve() if data_dir else None,
                      pdf_path=Path(pdf).resolve() if pdf else None)
    cleaned = section_aware_clean(pages)
    chunks = split_docs(cleaned)
    vs, hybrid, doc_bm25, page_bm25, reranker = build_indexes(chunks)

    def retrieve(q):
        return two_stage_retrieve_multi(q, vs, hybrid, doc_bm25, page_bm25, reranker, max_total=MAX_TOTAL_CTX)
    chain = build_chain(retrieve)

    questions = load_eval(Path(eval_path).resolve())
    avail = {str(d.metadata.get("doc","NA")) for d in chunks}
    gold_ids = {str(g.get("doc","")) for q in questions for g in q.get("gold", [])}
    missing = sorted(gold_ids - avail)
    if missing:
        print("\n[WARN] Gold doc IDs NOT found:", missing, "\n")

    buckets = {"overall": []}
    for item in questions:
        q = item["question"]
        must = [t.lower() for t in item.get("must_include_any", [])]
        gold = item.get("gold", [])
        q_aug = (q + " " + " ".join(must)).strip()

        t0 = time.time()
        docs = retrieve(q_aug)
        ans = chain.invoke(q)
        lat = (time.time() - t0) * 1000.0

        pred_pairs = pred_pairs_from_docs(docs)
        pred_set = set(pred_pairs)
        pred_docs = {d for d, _ in pred_set}

        gold_pairs, gold_docs_only = set(), set()
        for g in gold:
            doc = g["doc"];
            pages = g.get("pages", [])
            if pages:
                gold_pairs.update({(doc, int(p)) for p in pages})
            else:
                gold_docs_only.add(doc)

        strict_hits = len(gold_pairs & pred_set) + len(gold_docs_only & pred_docs)
        denom = len(gold_pairs) + len(gold_docs_only)
        def tolerant_pair_hit(gpairs, ppairs, tol=1):
            phash = {}
            for d,p in ppairs:
                phash.setdefault(d, set()).add(p)
            c = 0
            for d,p in gpairs:
                if d in phash and any(abs(p - tp) <= tol for tp in phash[d]):
                    c += 1
            return c
        tol_hits_pairs = tolerant_pair_hit(gold_pairs, pred_set, tol=1)
        tol_hits = tol_hits_pairs + len(gold_docs_only & pred_docs)

        # —— 文献级命中（只看 doc 是否在 top-k）——
        doc_hit = 1 if any(d in pred_docs for d in [g["doc"] for g in gold]) else 0

        precision = strict_hits / max(len(set(pred_pairs)), 1)
        recall = strict_hits / denom if denom>0 else 1.0
        recall_tol = tol_hits / denom if denom>0 else 1.0

        cited = 1 if has_citation(ans) else 0
        key_hit = 1 if any(k in ans.lower() for k in must) else 0

        # 收集：lat, cited, key_hit, recall(strict), precision, recall_tol, doc_hit
        buckets["overall"].append((lat, cited, key_hit, recall, precision, recall_tol, doc_hit))

        if denom>0 and strict_hits==0:
            gold_pairs_list = [(g["doc"], int(p)) for g in gold for p in g.get("pages", [])]
            gold_docs_list = [g["doc"] for g in gold if not g.get("pages", [])]
            print(f"[DEBUG miss] {q[:60]}... | pred={sorted(set(pred_pairs))} | gold_pairs={sorted(set(gold_pairs_list))} | gold_docs={sorted(set(gold_docs_list))}")

    print("\n===== Eval Summary =====")
    print("overall =>", summarize(buckets["overall"]))

if __name__ == "__main__":
    load_dotenv(); set_seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--pdf", type=str, default=None)
    ap.add_argument("--eval", type=str, default="eval/eval_questions.jsonl")
    args = ap.parse_args()
    main(data_dir=args.data_dir, pdf=args.pdf, eval_path=args.eval)
