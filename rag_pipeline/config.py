import os

def _get(name, default, cast=str):
    v = os.getenv(name, default)
    try:
        return cast(v) if cast else v
    except Exception:
        return default

# 向量/检索
EMB_MODEL_NAME   = _get("EMB_MODEL_NAME", "BAAI/bge-m3")
DENSE_K          = _get("DENSE_K", 16, int)
BM25_K           = _get("BM25_K", 16, int)
MMR_FETCH_K      = _get("MMR_FETCH_K", 40, int)
MMR_LAMBDA       = _get("MMR_LAMBDA", 0.5, float)
RERANK_TOPN      = _get("RERANK_TOPN", 12, int)
MAX_TOTAL_CTX    = _get("MAX_TOTAL_CTX", 6, int)
PER_PAGE_LIMIT   = _get("PER_PAGE_LIMIT", 3, int)
POOL_CAP         = _get("POOL_CAP", 120, int)
PAGE_CLIP_CHARS  = _get("PAGE_CLIP_CHARS", 2000, int)
RERANK_CLIP      = _get("RERANK_CLIP", 1200, int)
DOC_ROUTE_K      = _get("DOC_ROUTE_K", 6, int)
PER_DOC_PAGES    = _get("PER_DOC_PAGES", 6, int)

# 其它
LLM_TIMEOUT      = _get("LLM_TIMEOUT", 120, int)
PERSIST_DIRNAME  = _get("PERSIST_DIRNAME", "chroma_db_eval")
KEEP_FIRST_PAGES = _get("KEEP_FIRST_PAGES", 3, int)
SEED             = _get("EVAL_SEED", 2025, int)
# 底部追加
DEVICE      = _get("DEVICE", "cuda")      # "cuda" / "cpu" / "mps"
CE_DEVICE   = _get("CE_DEVICE", "cuda")   # Cross-Encoder 的设备
EMB_BATCH   = _get("EMB_BATCH", 64, int)  # 嵌入批量：GPU 可 64~128，CPU 建议 16

