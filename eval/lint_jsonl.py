import json, sys, re

ALLOWED_CATS = {"contribution","method","result","limitation","data"}

def is_int_list(xs):
    return isinstance(xs, list) and all(isinstance(i, int) and i >= 0 for i in xs)

def lint_line(obj, i):
    errs = []
    # 必备字段
    for k in ["question","category","target_docs","gold","must_include_any"]:
        if k not in obj: errs.append(f"missing field: {k}")
    if errs: return errs

    # 基本类型
    if not isinstance(obj["question"], str): errs.append("question must be string")
    if obj["category"] not in ALLOWED_CATS: errs.append(f"category not in {ALLOWED_CATS}")
    if not isinstance(obj["target_docs"], list) or not all(isinstance(d, str) for d in obj["target_docs"]):
        errs.append("target_docs must be list[str]")
    if not isinstance(obj["gold"], list): errs.append("gold must be list[object]")
    if not isinstance(obj["must_include_any"], list) or not all(isinstance(t, str) for t in obj["must_include_any"]):
        errs.append("must_include_any must be list[str]")

    # gold 校验
    seen = set()
    for g in obj["gold"]:
        if not isinstance(g, dict) or "doc" not in g or "pages" not in g:
            errs.append("gold item must have {doc, pages}")
            continue
        if not isinstance(g["doc"], str): errs.append("gold.doc must be string")
        pages = g["pages"]
        if pages != [] and not is_int_list(pages):
            errs.append("gold.pages must be [] or list[int]>=0")
        # 轻量去重提示
        for p in pages:
            key = (g["doc"], p)
            if key in seen: errs.append(f"duplicate gold pair: {key}")
            seen.add(key)

        # 建议：每文档证据页不超过 5
        if isinstance(pages, list) and len(pages) > 5:
            errs.append(f"gold.pages too many ({len(pages)}); consider ≤3~5")

    return errs

def main(path):
    ok = 0; bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[L{i}] JSON parse error: {e}")
                bad += 1
                continue
            errs = lint_line(obj, i)
            if errs:
                print(f"[L{i}] " + " | ".join(errs))
                bad += 1
            else:
                ok += 1
    print(f"\nSummary: ok={ok}, bad={bad}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/eval_questions.jsonl"
    main(path)
