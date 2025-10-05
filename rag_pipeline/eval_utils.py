
import json, re, statistics
from typing import List, Dict, Any, Tuple
from langchain.schema import Document

def load_eval(path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def has_citation(ans: str) -> bool:
    return bool(re.search(r"\[\d+\]", ans))

def pred_pairs_from_docs(docs: List[Document]) -> List[Tuple[str, int]]:
    pairs = set()
    for d in docs:
        docid = str(d.metadata.get("doc", "NA"))
        try:
            page = int(d.metadata.get("page", -1))
        except Exception:
            page = -1
        pairs.add((docid, page))
    return list(pairs)

def summarize(rows):
    if not rows:
        return {}
    # rows: (lat, cited, key_hit, recall_strict, precision, recall_tol, doc_hit)
    lats = [r[0] for r in rows]
    cites = [r[1] for r in rows]
    keys = [r[2] for r in rows]
    recs = [r[3] for r in rows]        # strict recall
    precs = [r[4] for r in rows]
    recs_tol = [r[5] for r in rows]    # tolerant recall
    doc_hits = [r[6] for r in rows]    # doc-level hit flag

    p50 = round(statistics.median(lats), 1)
    p95 = round(sorted(lats)[max(int(0.95 * len(lats)) - 1, 0)], 1)
    return {
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "citation_rate": round(sum(cites) / len(cites), 3),
        "keyphrase_hit_rate": round(sum(keys) / len(keys), 3),
        "avg_recall_at_k_strict": round(sum(recs) / len(recs), 3),
        "avg_precision_at_k": round(sum(precs) / len(precs), 3),
        "avg_recall_at_k_tolerant(±1page)": round(sum(recs_tol) / len(recs_tol), 3),
        "doc_hit_rate@k": round(sum(doc_hits) / len(doc_hits), 3),
        "n": len(rows),
    }
