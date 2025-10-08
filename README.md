# RAG-for-academic

面向学术论文的检索增强生成（RAG）小型工程：**页级 BM25 + 段级向量检索（BGE）+ 交叉编码器重排**，并内置**页路由与文档配额**，统一输出可证据的 `(doc/page/source)` 元数据，支持离线评测与**端到端基线对比**。检索与生成链路均为一键运行。 

---

## ✨ 特性

* **混合候选池（dense + BM25-chunk）**：并行召回、去重后进入重排，降低关键词漂移。 
* **页级路由（Page-BM25）**：在 *dense* 命中文档内做“中后段优先”的页选择，并过滤参考文献/致谢等噪声页。 
* **交叉编码器重排**：`cross-encoder/ms-marco-MiniLM-L-6-v2`，chunk 级重排并支持长度裁剪。 
* **文档/页配额与“深页位”**：先保 1 个深页路由位，再按配额筛入，平衡 precision 与上下文连续性。 
* **（可选）邻页扩展**：对最终 Top-M 的 `(doc,page)` 做 ±2 近邻补全（默认关闭，可在配置开启）。 
* **稳健元数据与可视前缀**：若缺失 `doc`，自动用 `source` 文件名兜底；打印上下文时带 `(doc, p.page)` 前缀，方便定位与评测。 

---

## 目录结构

```
.
├─ rag_pipeline/
│  ├─ cleaning.py        # 清洗与切块（含元数据兜底 & 内容前缀）
│  ├─ indexing.py        # BGE 向量库 + BM25（doc/page/chunk）+ 交叉编码器
│  ├─ retrieval.py       # 混合检索 + 页路由 + 重排 +（可选）邻页扩展
│  ├─ llm_chain.py       # 生成链路（仅用上下文作答 + [编号] 引用）
│  ├─ eval_utils.py, utils.py, config.py
├─ main.py               # 单次问答入口（打印上下文与答案）
├─ eval.py               # 评测（严格页命中、±1 容错、文献命中、引用率等）
├─ eval_rag_baselines.py # 一键跑“dense-only / bm25-only”端到端基线
└─ eval/eval_questions.jsonl
```

（索引与重排配置见 `indexing.py`；评测与基线脚本见 `eval.py / eval_utils.py / eval_rag_baselines.py`。）   

---

## 快速开始

### 1) 环境

```bash
# Python 3.10+
pip install -r requirements.txt
```

### 2) 数据准备

```
data/
└─ pdfs/
   ├─ paper_001.pdf
   ├─ paper_002.pdf
   └─ ...
```

> `doc_id` 会从文件名自动提取，如 `paper_001`。

### 3) 单次问答

```bash
python main.py --data_dir data/ --q "What are the two major modes of Photoacoustic Tomography?"
```

程序会：加载→清洗→切块→建索引→两阶段检索→打印上下文与答案。

### 4) 离线评测（带容错）

```bash
python eval.py --data_dir data/ --eval eval/eval_questions.jsonl
```

输出包括：延迟 p50/p95、严格页命中@k、±1 页容错召回、文献级命中率@k、引用率、关键短语命中，并打印 `DEBUG miss` 以定位问题。 

> 运行时控制台会打印如：`per_doc_pages=5  pool=36  topN=20  final=4 | routed: pool+=13, topN=13, final=2` 的调试行，便于理解路由与配额的实际生效情况。

### 5) 端到端基线对比（推荐）

```bash
python eval_rag_baselines.py --data_dir data/ --eval eval/eval_questions.jsonl
```

一次性跑出 `rag_dense_only` 与 `rag_bm25_only` 两条端到端基线（召回 + 生成 + 引用检测），用于与主系统对比。 

---

## 关键配置（`.env` 或环境变量覆盖）

* `MAX_TOTAL_CTX`：最终进入上下文的片段数（默认 4）
* `PER_PAGE_LIMIT`：同一页最多保留的片段数（默认 3）
* `PER_DOC_PAGES`：页路由每文档最多探查的页数（默认 5）
* `DOC_ROUTE_K`：文档级 BM25 的路由数量（默认 6）
* `RERANK_CLIP`：交叉编码器重排的文本截断长度（默认 1200）
* `USE_NEIGHBOR_EXPAND` / `NEIGHBOR_EXPAND_TOP_M`：是否启用邻页扩展以及对 Top-M 的扩展阈值（默认关闭，`0`）
* 其它：`EMB_MODEL_NAME`、`DENSE_K/BM25_K`、`MMR_*`、`RERANK_TOPN`、`DEVICE/CE_DEVICE` 等
  （完整默认值见 `config.py`。） 

---

## 原理简述

* **两阶段检索**：并行混合召回 → 页级路由（优先中后段、过滤噪声）→ 交叉编码器重排（clip 后）→ **文档/页配额筛入 + 深页位** →（可选）邻页扩展 → 生成链路仅使用所给上下文作答并以 `[1]..[k]` 编号引用。    

---

## 评测指标（本次实测，n=30）

使用命令：

```bash
python eval.py --data_dir data/ --eval eval/eval_questions.jsonl
```

**整体：**

* **latency**：p50 = **4152.3 ms**，p95 = **5408.1 ms**
* **citation_rate**：**1.000**
* **keyphrase_hit_rate**：**1.000**
* **strict recall@k**（严格页命中@k）：**0.737**
* **precision@k**：**0.350**
* **tolerant recall@k (±1 page)**：**0.822**
* **doc hit rate@k**（文献级命中@k）：**0.967**

> 指标统计口径见 `eval_utils.summarize()` 与 `eval.py`（其中严格页命中@k、±1 容错召回、文献级命中率@k、引用率与关键短语命中率的计算逻辑有明确定义）。 

---


## 复现清单（Checklist）

* [ ] 创建并激活虚拟环境，安装依赖
* [ ] 将 PDF 放入 `data/pdfs/`
* [ ] 首次运行 `python main.py --data_dir data/ --q "..."` 构建索引
* [ ] 运行 `python eval.py --data_dir data/ --eval eval/eval_questions.jsonl` 查看指标与 `DEBUG miss`
* [ ] （可选）运行 `python eval_rag_baselines.py ...` 获取端到端基线

---

## Roadmap

* [ ] **切块升级**：标题/句子感知 + 图/表/公式合并 + token 级长度控制
* [ ] **别名合并**：将 `paper_004_iterative_...` 等映射回主 `paper_004`，避免打分稀释
* [ ] **多查询扩展**：同义词/中英混排的轻量 query 扩展
* [ ] **判定优化**：评测增加 doc-level 容错与“多页等价答案”规则


