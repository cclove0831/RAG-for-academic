import os

def _get(name, default, cast=str):
    v = os.getenv(name, default)
    try:
        return cast(v) if cast else v
    except Exception:
        return default

# 向量/检索
EMB_MODEL_NAME   = _get("EMB_MODEL_NAME", "BAAI/bge-m3")
DENSE_K          = _get("DENSE_K", 20, int)
BM25_K           = _get("BM25_K", 16, int)
MMR_FETCH_K      = _get("MMR_FETCH_K", 40, int)
MMR_LAMBDA       = _get("MMR_LAMBDA", 0.5, float)
RERANK_TOPN      = _get("RERANK_TOPN", 20, int)
MAX_TOTAL_CTX    = _get("MAX_TOTAL_CTX", 4, int)
PER_PAGE_LIMIT   = _get("PER_PAGE_LIMIT", 3, int)
POOL_CAP         = _get("POOL_CAP", 40, int)
PAGE_CLIP_CHARS  = _get("PAGE_CLIP_CHARS", 2000, int)
RERANK_CLIP      = _get("RERANK_CLIP", 1200, int)
DOC_ROUTE_K      = _get("DOC_ROUTE_K", 6, int)
PER_DOC_PAGES    = _get("PER_DOC_PAGES", 5, int)

# 其它
LLM_TIMEOUT      = _get("LLM_TIMEOUT", 120, int)
PERSIST_DIRNAME  = _get("PERSIST_DIRNAME", "chroma_db_eval")
KEEP_FIRST_PAGES = _get("KEEP_FIRST_PAGES", 3, int)
SEED             = _get("EVAL_SEED", 2025, int)
# 底部追加
DEVICE      = _get("DEVICE", "cuda")
CE_DEVICE   = _get("CE_DEVICE", "cuda")
EMB_BATCH   = _get("EMB_BATCH", 64, int)
USE_PAGE_REPRESENTATION = False
USE_NEIGHBOR_EXPAND     = False
RERANK_SCOPE            = "chunk"
PAGE_REP_HEAD_CHARS = 450
PAGE_REP_TAIL_CHARS = 450
PER_PAGE_CHUNKS     = 3
DENSE_FETCH_K   = 80
BM25_FETCH_K    = 80
NEIGHBOR_EXPAND_TOP_M=0
