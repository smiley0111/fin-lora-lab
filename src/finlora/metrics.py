"""评测指标。所有方法的预测都过同一套函数，结果才可比。

实体级：把每条样本的标注和预测都看成实体集合，比较前统一括号和大小写。
样本级：只看“有没有负面主体”这个判断，对应风控里的误报和漏报。
"""

from __future__ import annotations

from collections import Counter

from finlora.data import canon


def _sets(gold: list[str], pred: list[str]) -> tuple[set[str], set[str]]:
    return {canon(e) for e in gold}, {canon(e) for e in pred}


def classify_error(text: str, gold: list[str], pred: list[str]) -> str:
    """给一条样本打错误类型，方便看模型具体错在哪。"""
    g, p = _sets(gold, pred)
    if g == p:
        return "正确"
    if not g:
        return "误报"
    if not p:
        return "漏报"
    t = canon(text)
    if any(e not in t for e in p - g):
        return "输出了原文没有的实体"
    partial = any(a in b or b in a for a in p - g for b in g - p)
    if partial:
        return "实体边界不对"
    if p < g:
        return "少找了实体"
    if p > g:
        return "多找了实体"
    return "找错了实体"


def evaluate(records: list[dict], preds: list[list[str]]) -> dict:
    assert len(records) == len(preds)
    tp = fp = fn = 0
    exact = 0
    det = Counter()
    errors = Counter()
    hallucinated = 0
    pred_entities = 0
    for r, pred in zip(records, preds, strict=True):
        g, p = _sets(r["entities"], pred)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        exact += g == p
        det[(bool(g), bool(p))] += 1
        errors[classify_error(r["text"], r["entities"], pred)] += 1
        t = canon(r["text"])
        pred_entities += len(p)
        hallucinated += sum(e not in t for e in p)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = len(records)
    neg_total = det[(False, False)] + det[(False, True)]
    pos_total = det[(True, True)] + det[(True, False)]
    return {
        "n": n,
        "entity_precision": round(precision, 4),
        "entity_recall": round(recall, 4),
        "entity_f1": round(f1, 4),
        "exact_match": round(exact / n, 4),
        "detection_accuracy": round((det[(True, True)] + det[(False, False)]) / n, 4),
        "false_alarm_rate": round(det[(False, True)] / neg_total, 4) if neg_total else 0.0,
        "miss_rate": round(det[(True, False)] / pos_total, 4) if pos_total else 0.0,
        "hallucination_rate": round(hallucinated / pred_entities, 4) if pred_entities else 0.0,
        "errors": dict(errors.most_common()),
    }
