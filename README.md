# Hinglish Support Intent — LoRA Fine-tune of Qwen2.5-0.5B

Fine-tunes a small open LLM to read customer-support messages written in **Hinglish, Romanized Hindi or English** and return structured JSON:

```
"mra order OD48213377 kahan pahuncha jaldi"
→ {"intent": "order_status", "order_id": "OD48213377"}
```

Ten intents from e-commerce and banking support (order status, refund, cancellation, failed payment, card block, address change, delivery delay, return, login issue, talk to an agent).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abhisheklpu1731/hinglish-intent-qlora/blob/main/train_colab.ipynb)

## Demo

A Gradio app (`space/`) runs the fine-tuned model next to the base model on the same message, so the effect of fine-tuning is visible side by side, with CPU latency for each. It runs on a free Hugging Face CPU Space.

**Live demo:** _link added after deployment_

## Results

Held-out test set: 240 messages built from sentence patterns **never seen in training**.

| Model | JSON valid | Intent accuracy | Macro-F1 | Order-ID exact | ms / example |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (zero-shot) | _run notebook_ | | | | |
| + LoRA SFT (this repo) | _run notebook_ | | | | |

Loss curve: `results/loss.png` · per-example predictions for error analysis: `results/*_predictions.jsonl`

## How it works

**1. Synthetic data** (`data/generate.py`)
- Hand-written templates per intent, mixing Hinglish, Romanized Hindi and English, filled with random order IDs, products, amounts and day counts.
- Noise that real chat text has: spelling variants (`nahi/nhi`, `kya/kia`, `please/pls`), lowercase, dropped punctuation, "urgent/jaldi" suffixes.
- **Leakage control:** ~25% of templates per intent are held out *before* slot filling, so the test set checks new phrasings, not memorised ones. 1,917 train / 213 val / 240 test.

**2. LoRA SFT** (`src/train.py`, TRL `SFTTrainer`)
- **LoRA r=16, alpha=32** on all linear layers (8.8M trainable parameters, 1.7% of the model).
- On the free Colab T4 it trains in full fp32 (`--fp32`). On Ampere GPUs it runs as **QLoRA**: base weights frozen in 4-bit NF4 with double quantisation, bf16 compute.
- Precision handling for older GPUs: the T4 has no real bf16 support, and TRL casts QLoRA adapters to bf16, which breaks fp16 gradient scaling. `train.py` detects the GPU generation and picks a working setup instead of crashing.
- Conversational prompt-completion format with **loss on the completion only**, so the model learns the answer rather than the system prompt.
- 2 epochs, lr 2e-4, cosine schedule. Optional Weights & Biases logging (`--report-to wandb`).

**3. Evaluation** (`src/evaluate.py`)
- Greedy decoding, batched with left padding; LoRA merged into the base weights for inference.
- Metrics: JSON validity, intent accuracy, macro-F1, order-ID exact match, latency per example.
- Same script and test set for base and fine-tuned, so the comparison is like for like.

## Run it

Colab (free T4, ~25 min): open `train_colab.ipynb` and *Run all*.

Locally with a GPU:
```bash
pip install -r requirements.txt
python data/generate.py --out data
python src/evaluate.py --tag base
python src/train.py --out outputs/qwen05-lora
python src/evaluate.py --adapter outputs/qwen05-lora --tag finetuned
pytest tests
```

## Limitations and next steps

- Data is synthetic and template-based; real support transcripts are messier (longer, multi-intent, Devanagari mixed in).
- Next: a DPO pass on preference pairs (correct JSON vs common mistakes), compare LoRA ranks, and try Qwen2.5-1.5B to measure the accuracy/latency trade-off.

## Deploy the demo (free)

1. Run the Colab notebook to the end. Its last cell pushes the LoRA adapter to the Hugging Face Hub (this repo's is [`abhikrs41/qwen05-hinglish-intent-lora`](https://huggingface.co/abhikrs41/qwen05-hinglish-intent-lora)).
2. On huggingface.co: **New Space** → SDK **Gradio** → hardware **CPU basic (free)**.
3. Upload the three files from `space/` (`app.py`, `requirements.txt`, `README.md`).
4. If your adapter repo name differs, set `ADAPTER_ID` under Space → Settings → Variables.

The first start downloads the 0.5B model (~1 GB) and takes a few minutes.

## Structure
```
data/generate.py      synthetic dataset generator
src/common.py         prompt format, output parsing, metrics
src/train.py          QLoRA SFT with TRL
src/evaluate.py       base vs fine-tuned evaluation
train_colab.ipynb     end-to-end Colab run
tests/                unit tests for parsing and metrics
space/                Gradio demo for Hugging Face Spaces
```
