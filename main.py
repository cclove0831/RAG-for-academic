import argparse, time
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline.utils import set_seed
from rag_pipeline.io_data import load_pdfs
from rag_pipeline.cleaning import section_aware_clean, split_docs
from rag_pipeline.indexing import build_indexes
from rag_pipeline.retrieval import two_stage_retrieve_multi, fmt_docs
from rag_pipeline.llm_chain import build_chain
from rag_pipeline.config import SEED, MAX_TOTAL_CTX

def run_once(data_dir: str=None, pdf: str=None, question: str="What is the main contribution?"):
    pages = load_pdfs(data_dir=Path(data_dir).resolve() if data_dir else None,
                      pdf_path=Path(pdf).resolve() if pdf else None)
    cleaned = section_aware_clean(pages)
    chunks = split_docs(cleaned)
    vs, hybrid, doc_bm25, page_bm25, reranker = build_indexes(chunks)

    def retrieve(q):
        return two_stage_retrieve_multi(q, vs, hybrid, doc_bm25, page_bm25, reranker, max_total=MAX_TOTAL_CTX)

    chain = build_chain(retrieve)

    t0 = time.time()
    ctx = fmt_docs(retrieve(question))
    ans = chain.invoke(question)
    print("Context:\n", ctx)
    print("\nAnswer:\n", ans)
    print("\nLatency(ms):", round((time.time()-t0)*1000,1))

if __name__ == "__main__":
    load_dotenv(); set_seed(SEED)
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--pdf", type=str, default=None)
    ap.add_argument("--q", type=str, default="What is the main contribution?")
    args = ap.parse_args()
    run_once(data_dir=args.data_dir, pdf=args.pdf, question=args.q)
