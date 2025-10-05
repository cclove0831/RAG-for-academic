# RAG-for-acdemic
基于检索增强生成的论文问答助手
下面是一份可直接粘贴到你仓库的 `README.md`（已贴合你项目实际：双索引混合检索、邻页预取/补全、评测指标、常见报错与复现步骤等）。你只需要把仓库名、许可证等位置替换为你的即可。

---


> 面向学术论文的检索增强生成（RAG）小型工程：**页级 BM25 + 段级向量检索 + 交叉编码器重排**，内置**邻页预取 / 邻页补全**，返回带编号的证据片段，支持离线评测与日志定位。

<p align="center">
  <b>文档命中@k</b>：0.967　|　<b>严格页命中@k</b>：0.696　|　<b>±1页容错召回@k</b>：0.855  
</p>

---

## ✨ 核心特性

* **双索引混合检索**：

  * 文档级路由：`doc-BM25`
  * 页级路由：`page-BM25`（按每文档前 P 页；列表型问句自动放宽）
  * 段级候选：向量检索（BGE 系列）与 `bm25-chunk` 混合
* **交叉编码器重排**：对候选进行语义精排，避免关键词漂移
* **邻页策略**：

  * 预取：对入选页自动将 `page±1` 的 Top-1 预加入候选池
  * 补全：重排后若仍缺邻页，兜底现取 1 条并加入最终上下文（遵守上下文与每页上限）
* **可证据返回**：统一元数据 `doc/page/source`，回答带**编号引用**
* **一键评测**：本地 `eval.py` 输出命中率、引用率、关键短语命中、延迟分布，并打印 `DEBUG miss` 方便定位

---

## 目录结构

```
.
├─ rag_pipeline/
│  ├─ cleaning.py        # 清洗与切块
│  ├─ indexing.py        # 构建向量与BM25索引
│  ├─ retrieval.py       # 双索引检索 + 重排 + 邻页预取/补全（已启用）
│  ├─ llm_chain.py       # 生成链路（带证据）
│  ├─ utils.py, config.py
├─ main.py               # 单次问答入口
├─ eval.py               # 批量评测
├─ eval_utils.py
├─ data/                 # 论文PDF与中间产物（见下）
└─ eval/
   └─ eval_questions.jsonl  # 评测集（问法覆盖“贡献、分类、定义、对比、设计”等）
```

---

## 快速开始

### 1) 环境

```bash
# Python 3.10+ 推荐
pip install -r requirements.txt

# 可选：HuggingFace 国内缓存（建议）：
# 设置 HUGGINGFACE_HUB_CACHE 指向本地盘，断网情况下也能复用
```

### 2) 数据准备

```
data/
├─ pdfs/
│   ├─ paper_001.pdf
│   ├─ paper_002.pdf
│   └─ ...
└─ (运行后自动生成的索引与缓存)
```

> 说明：`doc_id` 默认取 `source` 文件名（不含后缀），如 `paper_001`。若有补充稿/别名文件，文件名中带后缀即可（例如 `paper_004_iterative_optimization_structural_prior.pdf`）。

### 3) 运行

单次问答：

```bash
python main.py --data_dir data/ --q "What are the two major modes of Photoacoustic Tomography?"
```

离线评测：

```bash
python eval.py --data_dir data/ --eval eval/eval_questions.jsonl
```

---

## 关键配置（无需改代码即可使用）

`config.py` 已给出默认值。常用项：

* `MAX_TOTAL_CTX`：最终进入上下文的片段总数
* `PER_PAGE_LIMIT`：同一页最多保留的片段数
* `POOL_CAP`：候选池上限
* `PER_DOC_PAGES`：每个文档通过页级路由选取的页数上限
* `RERANK_CLIP`：交叉编码器重排的文本截断长度

> 注：当前仓库实现的**邻页预取/补全**为代码内置策略（不依赖新配置项），即使不调整 `config.py` 也会生效。

---

## 评测指标（实测）

本地跑出的典型结果（30问）：

* **文档命中@k**：0.967
* **严格页命中@k**：0.696
* **±1页容错召回@k**：0.855
* **引用率**：1.0
* **关键短语命中**：1.0
* **延迟**：p50 ≈ 5–9s，p95 ≈ 9–10s（取决于模型缓存/网络）

> 解释：最初严格页命中 ~0.587。通过**邻页预取 + 邻页补全 + 放宽重排截断**，严格页命中提升到 ~0.696，±1 容错 0.855，文档命中 0.967。剩余 miss 多因“页级路由漏掉含图注/定义句的关键页”，下一步通过**结构化切块**继续优化。

---

## 实现要点

* **双索引混合检索**

  1. `doc_bm25` 路由出候选文档；
  2. `page_bm25` 在候选文档内挑出前 P 页；
  3. dense/bm25-chunk 作为段级补充；
  4. 页内向量 Top-1 + 邻页预取，保证跨页上下文完整性。

* **邻页策略**

  * *预取*：在建池阶段对每个入选页预取 `page±1` 的 Top-1；
  * *补全*：重排后若该邻页仍不在最终集合中，则兜底向量检索 1 条加入（遵守 `MAX_TOTAL_CTX` 与 `PER_PAGE_LIMIT`）。

* **元数据稳定**

  * 若 `doc` 缺失（出现 `NA`/`paper`），用 `source` 的文件名 `stem` 兜底修复，确保 `(doc, page)` 正确统计与引用。

---


## 路线图（Roadmap）

* [ ] **切块升级**：标题/句子感知 + 图/表/公式合并 + token 级长度控制
* [ ] **别名合并**：将 `paper_004_iterative_...` 等映射回主 `paper_004`，避免打分稀释
* [ ] **多查询扩展**：同义词/中英混排的轻量 query 扩展
* [ ] **判定优化**：评测增加 doc-level 容错与 “多页等价答案” 规则

---

## 复现清单（Checklist）

* [ ] 创建并激活虚拟环境，安装依赖
* [ ] 将 PDF 放入 `data/pdfs/`
* [ ] 首次运行 `python main.py --data_dir data/ --q "..."` 以构建索引
* [ ] 运行 `python eval.py --data_dir data/ --eval eval/eval_questions.jsonl`
* [ ] 查看 `DEBUG miss` 与指标输出；若有网络下载错误，先完成模型缓存

