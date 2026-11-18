"""LoRA 微调 Qwen2.5-0.5B-Instruct。CPU 可跑，没有用 Trainer，训练循环写在明处。

    python -m finlora.train --name lora-n1000 --n-train 1000
    python -m finlora.train --name smoke --n-train 16 --max-steps 2     # 冒烟

要点：
- 只对回答部分算 loss。提示词部分的 label 设为 -100，否则模型会花力气去“背”提示词。
- 按长度分桶组 batch，减少 padding；CPU 上 padding 就是白算。
- 只训练 LoRA 参数（注意力和 MLP 的投影层），基座权重冻结。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

from finlora.data import OUT, ROOT, build_messages, read_jsonl

MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B-Instruct"
RUNS = ROOT / "runs"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, record: dict, max_len: int) -> dict:
    """返回 input_ids 和 labels。提示词部分的 label 是 -100，只有回答和结束符参与 loss。"""
    msgs = build_messages(record["text"])
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": record["target"]}], tokenize=False
    )
    assert full.startswith(prompt), "chat 模板的提示词部分必须是完整对话的前缀"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"]
    assert f_ids[: len(p_ids)] == p_ids
    f_ids = f_ids[:max_len]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids) :]
    labels = labels[: len(f_ids)]
    return {"input_ids": f_ids, "labels": labels}


def make_batches(examples: list[dict], batch_size: int, seed: int) -> list[list[dict]]:
    """按长度排序后切 batch，再打乱 batch 的顺序：batch 内长度接近，batch 之间仍是随机的。"""
    rng = random.Random(seed)
    ordered = sorted(examples, key=lambda e: len(e["input_ids"]))
    batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
    rng.shuffle(batches)
    return batches


def collate(batch: list[dict], pad_id: int):
    import torch

    n = max(len(e["input_ids"]) for e in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, e in enumerate(batch):
        k = len(e["input_ids"])
        ids[i, :k] = torch.tensor(e["input_ids"])
        labels[i, :k] = torch.tensor(e["labels"])
        mask[i, :k] = 1
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def lr_at(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def dev_loss(model, examples: list[dict], pad_id: int, batch_size: int) -> float:
    import torch

    model.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for b in make_batches(examples, batch_size, seed=0):
            x = collate(b, pad_id)
            n = int((x["labels"] != -100).sum())
            total += float(model(**x).loss) * n
            tokens += n
    model.train()
    return total / tokens


def train(args: argparse.Namespace) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    train_recs = read_jsonl(OUT / "train.jsonl")
    random.Random(args.seed).shuffle(train_recs)
    train_recs = train_recs[: args.n_train]
    dev_recs = read_jsonl(OUT / "dev.jsonl")[: args.n_dev]
    train_ex = [encode(tok, r, args.max_len) for r in train_recs]
    dev_ex = [encode(tok, r, args.max_len) for r in dev_recs]

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size / args.grad_accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * 0.05))

    log_path = out_dir / "train_log.jsonl"
    log = log_path.open("w", encoding="utf-8")
    loss0 = dev_loss(model, dev_ex, tok.pad_token_id, args.batch_size)
    log.write(json.dumps({"step": 0, "dev_loss": loss0}) + "\n")
    print(
        f"可训练参数 {trainable:,} / {total_params:,}（{trainable / total_params:.2%}），"
        f"{len(train_ex)} 条样本，共 {total_steps} 步；初始 dev loss {loss0:.4f}"
    )

    model.train()
    step, micro, running = 0, 0, 0.0
    start = time.perf_counter()
    epoch = 0
    while step < total_steps:
        for batch in make_batches(train_ex, args.batch_size, seed=args.seed + epoch):
            x = collate(batch, tok.pad_token_id)
            loss = model(**x).loss / args.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, warmup)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            entry = {
                "step": step,
                "loss": round(running, 4),
                "lr": opt.param_groups[0]["lr"],
                "elapsed": round(time.perf_counter() - start, 1),
            }
            running = 0.0
            if step % args.eval_every == 0 or step == total_steps:
                entry["dev_loss"] = dev_loss(model, dev_ex, tok.pad_token_id, args.batch_size)
            log.write(json.dumps(entry) + "\n")
            log.flush()
            eta = entry["elapsed"] / step * (total_steps - step)
            msg = f"step {step}/{total_steps} loss {entry['loss']:.4f}"
            if "dev_loss" in entry:
                msg += f" dev {entry['dev_loss']:.4f}"
            print(f"{msg}  已用 {entry['elapsed']:.0f}s  预计剩余 {eta:.0f}s", flush=True)
            if step >= total_steps:
                break
        epoch += 1
    log.close()

    elapsed = time.perf_counter() - start
    model.save_pretrained(out_dir / "adapter")
    summary = {
        "name": args.name,
        "n_train": len(train_ex),
        "epochs": args.epochs,
        "steps": total_steps,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "rank": args.rank,
        "max_len": args.max_len,
        "trainable_params": trainable,
        "total_params": total_params,
        "train_seconds": round(elapsed, 1),
        "dev_loss_before": round(loss0, 4),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--n-train", type=int, default=1000)
    p.add_argument("--n-dev", type=int, default=100)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="省内存但慢约 15%%，0.5B 在 16GB 内存上用不着",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=320)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())


if __name__ == "__main__":
    main()
