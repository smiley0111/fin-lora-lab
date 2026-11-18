"""在测试集上跑预测并评测。两种后端用同一份提示词：

- hf：transformers 加载本地模型，可选挂 LoRA 适配器。微调前后对比用这个，模型精度完全一致。
- ollama：本地 Ollama 服务，用来对比更大的模型只靠提示词能做到多少（注意 Ollama 的模型是 4bit 量化版）。

    python -m finlora.predict --backend hf --name base-0.5b-zeroshot
    python -m finlora.predict --backend hf --name base-0.5b-fewshot --shots 3
    python -m finlora.predict --backend hf --name lora-0.5b --adapter runs/lora-full/adapter
    python -m finlora.predict --backend ollama --model qwen2.5:3b --name ollama-3b-fewshot --shots 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from finlora.data import OUT, ROOT, build_messages, parse_prediction, read_jsonl
from finlora.metrics import evaluate

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
REPORTS = ROOT / "reports"


def few_shot_examples(k: int) -> list[dict]:
    """从训练集里固定挑 k 条短样本，有实体和无实体交替，保证每次实验用的示例相同。"""
    if k <= 0:
        return []
    train = read_jsonl(OUT / "train.jsonl")
    short = [r for r in train if len(r["text"]) <= 90 and not r["missing_in_text"]]
    pos = [r for r in short if r["entities"]]
    neg = [r for r in short if not r["entities"]]
    picked = []
    for i in range(k):
        pool = pos if i % 2 == 0 else neg
        picked.append(pool[i // 2])
    return picked


def make_messages(text: str, shots: list[dict]) -> list[dict]:
    msgs = build_messages(text)
    system, user = msgs[0], msgs[1]
    history = []
    for s in shots:
        ex = build_messages(s["text"])[1]
        history += [ex, {"role": "assistant", "content": s["target"]}]
    return [system, *history, user]


class HFBackend:
    def __init__(self, model_dir: Path, adapter: str | None, threads: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
            model = model.merge_and_unload()
        self.model = model.eval()

    def generate(self, batch: list[list[dict]], max_new_tokens: int) -> list[str]:
        prompts = [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch
        ]
        enc = self.tok(prompts, return_tensors="pt", padding=True)
        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        new = out[:, enc["input_ids"].shape[1] :]
        return [self.tok.decode(x, skip_special_tokens=True) for x in new]


class OllamaBackend:
    def __init__(self, model: str, url: str = "http://localhost:11434"):
        import httpx

        self.model = model
        self.client = httpx.Client(base_url=url, timeout=120)

    def generate(self, batch: list[list[dict]], max_new_tokens: int) -> list[str]:
        outs = []
        for msgs in batch:
            r = self.client.post(
                "/api/chat",
                json={
                    "model": self.model,
                    "messages": msgs,
                    "stream": False,
                    "options": {"temperature": 0, "seed": 42, "num_predict": max_new_tokens},
                },
            )
            r.raise_for_status()
            outs.append(r.json()["message"]["content"])
        return outs


def run(args: argparse.Namespace) -> dict:
    records = read_jsonl(OUT / f"{args.split}.jsonl")
    if args.limit:
        records = records[: args.limit]
    shots = few_shot_examples(args.shots)

    if args.backend == "hf":
        backend = HFBackend(Path(args.model_dir), args.adapter, args.threads)
    else:
        backend = OllamaBackend(args.model)

    pred_dir = REPORTS / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.perf_counter()
    for i in range(0, len(records), args.batch_size):
        chunk = records[i : i + args.batch_size]
        raws = backend.generate(
            [make_messages(r["text"], shots) for r in chunk], args.max_new_tokens
        )
        for r, raw in zip(chunk, raws, strict=True):
            rows.append(
                {
                    "text": r["text"],
                    "gold": r["entities"],
                    "raw": raw,
                    "pred": parse_prediction(raw),
                }
            )
        done = min(i + args.batch_size, len(records))
        print(
            f"\r{args.name}: {done}/{len(records)}  {time.perf_counter() - start:.0f}s",
            end="",
            flush=True,
        )
    elapsed = time.perf_counter() - start
    print()

    with (pred_dir / f"{args.name}.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = evaluate(records, [r["pred"] for r in rows])
    result.update(
        name=args.name,
        backend=args.backend,
        model=args.model if args.backend == "ollama" else Path(args.model_dir).name,
        adapter=args.adapter,
        shots=args.shots,
        split=args.split,
        seconds=round(elapsed, 1),
        seconds_per_sample=round(elapsed / len(records), 3),
    )
    metric_dir = REPORTS / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / f"{args.name}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "errors"}, ensure_ascii=False))
    print(result["errors"])
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "ollama"], default="hf")
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="qwen2.5:3b", help="Ollama 模型名")
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--adapter", default=None)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=48)
    p.add_argument("--threads", type=int, default=8)
    run(p.parse_args())


if __name__ == "__main__":
    main()
