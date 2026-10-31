"""Evaluate a base model or base + LoRA adapter on the held-out test set.

    python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --tag base
    python src/evaluate.py --model Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/qwen05-lora --tag finetuned

Writes results/<tag>_metrics.json and results/<tag>_predictions.jsonl (for error analysis).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import compute_metrics, prompt_messages  # noqa: E402


def load_rows(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return rows[:limit] if limit else rows


@torch.no_grad()
def generate(model, tokenizer, texts, batch_size, max_new_tokens=48):
    tokenizer.padding_side = "left"   # left-pad for batched generation in decoder-only models
    outputs = []
    for i in range(0, len(texts), batch_size):
        prompts = [
            tokenizer.apply_chat_template(prompt_messages(t), tokenize=False, add_generation_prompt=True)
            for t in texts[i:i + batch_size]
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        outputs += tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--test", default="data/test.jsonl")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if args.adapter and not Path(args.adapter, "adapter_config.json").exists():
        sys.exit(f"No trained adapter found at {args.adapter}. Run src/train.py first and check it finished.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)   # LoRA doesn't change the tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()   # merge LoRA into base weights: no adapter overhead at inference
    model.eval()

    rows = load_rows(args.test, args.limit)
    start = time.perf_counter()
    preds = generate(model, tokenizer, [r["text"] for r in rows], args.batch)
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(rows, preds)
    metrics["ms_per_example"] = round(1000 * elapsed / len(rows), 1)
    metrics.update({"model": args.model, "adapter": args.adapter, "device": device})

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / f"{args.tag}_predictions.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({**r, "raw_output": p}, ensure_ascii=False) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
