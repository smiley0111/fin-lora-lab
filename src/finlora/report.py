"""汇总 reports/metrics/*.json 和 runs/*/summary.json，生成 reports/results.md。

python -m finlora.report
"""

from __future__ import annotations

import json

from finlora.data import ROOT

REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"

ORDER = [
    "base-0.5b-zeroshot",
    "base-0.5b-fewshot3",
    "ollama-1.5b-fewshot3",
    "ollama-3b-zeroshot",
    "ollama-3b-fewshot3",
    "lora-0.5b-n200",
    "lora-0.5b-n800",
]

LABELS = {
    "base-0.5b-zeroshot": "0.5B 只用提示词",
    "base-0.5b-fewshot3": "0.5B 提示词 + 3 个示例",
    "ollama-1.5b-fewshot3": "1.5B 提示词 + 3 个示例（Ollama 4bit）",
    "ollama-3b-zeroshot": "3B 只用提示词（Ollama 4bit）",
    "ollama-3b-fewshot3": "3B 提示词 + 3 个示例（Ollama 4bit）",
    "lora-0.5b-n200": "0.5B + LoRA，200 条训练",
    "lora-0.5b-n800": "0.5B + LoRA，800 条训练",
}


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def main() -> None:
    metrics = {}
    for p in (REPORTS / "metrics").glob("*.json"):
        metrics[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    names = [n for n in ORDER if n in metrics] + sorted(set(metrics) - set(ORDER))

    lines = [
        "# 实验结果",
        "",
        "测试集：FinCUGE FINNSP 评测集 500 条（负面金融主体识别），所有方法用同一份提示词和同一套评测。",
        "",
        "| 方法 | 实体 F1 | 精确率 | 召回率 | 完全正确 | 误报率 | 漏报率 | 幻觉率 | 秒/条 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for n in names:
        m = metrics[n]
        lines.append(
            f"| {LABELS.get(n, n)} | **{pct(m['entity_f1'])}** | {pct(m['entity_precision'])} | "
            f"{pct(m['entity_recall'])} | {pct(m['exact_match'])} | {pct(m['false_alarm_rate'])} | "
            f"{pct(m['miss_rate'])} | {pct(m['hallucination_rate'])} | {m['seconds_per_sample']} |"
        )
    lines += [
        "",
        "- 误报率：没有负面主体的文本里，被报出负面主体的比例。漏报率：有负面主体的文本里，什么都没报的比例。",
        "- 幻觉率：预测出的实体中，原文里根本没有的比例。",
        "- 秒/条：本机 CPU（i5-1145G7）上的推理耗时，transformers 是 fp32、batch 8；Ollama 是 4bit 量化、逐条请求，两者不能直接比速度。",
        "",
        "## 错误类型分布",
        "",
    ]
    kinds = sorted({k for m in metrics.values() for k in m["errors"]} - {"正确"})
    lines.append("| 方法 | " + " | ".join(kinds) + " |")
    lines.append("| --- |" + " --- |" * len(kinds))
    for n in names:
        e = metrics[n]["errors"]
        lines.append(
            f"| {LABELS.get(n, n)} | " + " | ".join(str(e.get(k, 0)) for k in kinds) + " |"
        )

    summaries = []
    for p in sorted(RUNS.glob("*/summary.json")):
        summaries.append(json.loads(p.read_text(encoding="utf-8")))
    if summaries:
        lines += [
            "",
            "## 训练",
            "",
            "| 训练 | 样本数 | 步数 | 有效 batch | 学习率 | LoRA rank | 可训练参数 | 训练耗时 | dev loss 训练前 → 训练后 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for s in summaries:
            log = [
                json.loads(x)
                for x in (RUNS / s["name"] / "train_log.jsonl").read_text().splitlines()
            ]
            last = [x["dev_loss"] for x in log if "dev_loss" in x][-1]
            lines.append(
                f"| {s['name']} | {s['n_train']} | {s['steps']} | {s['effective_batch']} | {s['lr']} | "
                f"{s['rank']} | {s['trainable_params'] / 1e6:.1f}M（{s['trainable_params'] / s['total_params']:.2%}） | "
                f"{s['train_seconds'] / 60:.0f} 分钟 | {s['dev_loss_before']:.3f} → {last:.3f} |"
            )

    out = REPORTS / "results.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
