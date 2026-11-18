"""后处理：丢掉原文里不存在的实体，再评测一次。

抽取任务的答案必须出自原文，模型抄错字（“绿化贷”写成“绿意贷”）或凭空编出的实体都可以这样拦掉。
不需要重新推理，直接读 reports/preds/ 里已有的预测。

    python -m finlora.postfilter lora-0.5b-n800 ollama-3b-zeroshot
"""

from __future__ import annotations

import json
import sys

from finlora.data import OUT, canon, read_jsonl
from finlora.metrics import evaluate
from finlora.predict import REPORTS


def keep_in_text(text: str, pred: list[str]) -> list[str]:
    t = canon(text)
    return [e for e in pred if canon(e) in t]


def main(names: list[str]) -> None:
    records = read_jsonl(OUT / "test.jsonl")
    for name in names:
        rows = read_jsonl(REPORTS / "preds" / f"{name}.jsonl")
        before = evaluate(records, [r["pred"] for r in rows])
        after = evaluate(records, [keep_in_text(r["text"], r["pred"]) for r in rows])
        result = {"name": name, "before": before, "after": after}
        out = REPORTS / "metrics" / "postfilter"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        keys = ["entity_f1", "entity_precision", "entity_recall", "exact_match", "false_alarm_rate"]
        print(name, {k: (before[k], after[k]) for k in keys})


if __name__ == "__main__":
    main(sys.argv[1:])
