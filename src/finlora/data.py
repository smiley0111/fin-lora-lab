"""FINNSP（负面金融主体识别）数据整理。

原始数据来自 FinCUGE-Instruction，同一任务有十几种指令和回答模板。
这里统一成一种提示词和一种输出格式，所有方法（提示词基线、LoRA 微调）用同一份数据评测。

    python -m finlora.data          # 生成 data/train.jsonl、data/dev.jsonl、data/test.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

TASK = "FINNSP"
MAX_CHARS = 300
NONE = "无"
SEP = ";"

SYSTEM = "你是银行的金融舆情风控助手。"
INSTRUCTION = (
    "找出下面文本中被负面报道的金融主体，例如涉嫌诈骗、非法集资、逾期、跑路、被立案调查的公司或平台。"
    "只输出主体名称，多个主体用英文分号 ; 分隔；没有负面主体时只输出“无”。"
)


def build_messages(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{INSTRUCTION}\n\n文本：{text}"},
    ]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_CHARS]


def parse_gold(output: str) -> list[str]:
    """从原始回答模板里取出实体。带冒号的模板后面是实体列表，否则是“没有负面实体”的各种说法。"""
    if "：" not in output:
        return []
    tail = output.split("：", 1)[1]
    return [e.strip() for e in tail.split(";") if e.strip()]


def normalize_entity(e: str) -> str:
    return e.strip().strip("。.，,、;；\"'“”")


def canon(s: str) -> str:
    """比较用的规范形式：统一全半角括号和大小写。标注和原文里的括号写法不一致。"""
    return s.replace("（", "(").replace("）", ")").lower()


def parse_prediction(text: str) -> list[str]:
    """把模型输出解析成实体集合。模型可能用中文分号、顿号、换行分隔，或者带上“负面主体：”之类的前缀。"""
    text = text.strip()
    if not text:
        return []
    first = text.splitlines()[0] if "；" in text or ";" in text else text.replace("\n", ";")
    if "：" in first:
        first = first.split("：", 1)[1]
    parts = re.split(r"[;；、,，]", first)
    ents = []
    for p in parts:
        p = normalize_entity(p)
        if not p or p == NONE or p.startswith("没有") or "无负面" in p:
            continue
        if p not in ents:
            ents.append(p)
    return ents


def format_target(entities: list[str]) -> str:
    return SEP.join(entities) if entities else NONE


def make_record(raw_input: str, raw_output: str) -> dict:
    text = clean_text(raw_input)
    gold = [normalize_entity(e) for e in parse_gold(raw_output)]
    gold = list(dict.fromkeys(e for e in gold if e))
    return {
        "text": text,
        "entities": gold,
        "target": format_target(gold),
        "truncated": len(re.sub(r"\s+", " ", raw_input).strip()) > MAX_CHARS,
        "missing_in_text": [e for e in gold if canon(e) not in canon(text)],
    }


def load_raw(split: str) -> list[tuple[str, str]]:
    import pandas as pd

    df = pd.read_parquet(RAW / f"{split}.parquet")
    df = df[df.task == TASK]
    return list(zip(df.input, df.output, strict=True))


def build(seed: int = 42, dev_size: int = 200) -> dict[str, list[dict]]:
    test = [make_record(i, o) for i, o in load_raw("eval")]
    test = list({r["text"]: r for r in test}.values())
    test_texts = {r["text"] for r in test}

    train_all = [make_record(i, o) for i, o in load_raw("train")]
    seen: set[str] = set()
    train_pool = []
    leaked = 0
    for r in train_all:
        if r["text"] in test_texts:
            leaked += 1
            continue
        if r["text"] in seen:
            continue
        seen.add(r["text"])
        train_pool.append(r)

    rng = random.Random(seed)
    rng.shuffle(train_pool)
    splits = {"dev": train_pool[:dev_size], "train": train_pool[dev_size:], "test": test}
    splits["_stats"] = {"train_raw": len(train_all), "removed_overlap_with_test": leaked}
    return splits


def summarize(records: list[dict]) -> dict:
    n = len(records)
    return {
        "n": n,
        "no_entity": sum(not r["entities"] for r in records),
        "entities": sum(len(r["entities"]) for r in records),
        "truncated": sum(r["truncated"] for r in records),
        "entity_not_in_text": sum(bool(r["missing_in_text"]) for r in records),
        "entity_not_in_text_untruncated": sum(
            bool(r["missing_in_text"]) and not r["truncated"] for r in records
        ),
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    splits = build()
    stats = splits.pop("_stats")
    for name, recs in splits.items():
        write_jsonl(OUT / f"{name}.jsonl", recs)
        stats[name] = summarize(recs)
    (OUT / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
