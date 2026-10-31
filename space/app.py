"""Gradio demo for the Hinglish support-intent model (runs on a free Hugging Face CPU Space).

Shows the fine-tuned model's JSON output next to the base model's, so the effect of
fine-tuning is visible on the same message.

Env vars (Space → Settings → Variables):
    BASE_MODEL  default Qwen/Qwen2.5-0.5B-Instruct
    ADAPTER_ID  your pushed LoRA adapter, e.g. abhikrs41/qwen05-hinglish-intent-lora
"""

import json
import os
import time

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER = os.getenv("ADAPTER_ID", "abhikrs41/qwen05-hinglish-intent-lora")

INTENTS = ["order_status", "refund_request", "cancel_order", "payment_failed", "card_block",
           "address_change", "delivery_delay", "product_return", "account_login_issue", "talk_to_agent"]
SYSTEM_PROMPT = (
    "You are a customer-support intent classifier for an Indian e-commerce and banking "
    "helpdesk. Customers write in Hinglish, Romanized Hindi or English. "
    "Reply with ONLY a JSON object: {\"intent\": <one of: " + ", ".join(INTENTS) + ">, "
    "\"order_id\": <the order ID exactly as written, or null>}."
)

torch.set_num_threads(os.cpu_count() or 2)
tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
tuned = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
tuned = PeftModel.from_pretrained(tuned, ADAPTER).merge_and_unload().eval()


@torch.no_grad()
def run(model, text):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    ms = (time.perf_counter() - t0) * 1000
    raw = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        parsed = {"error": "not valid JSON", "raw_output": raw}
    return parsed, f"{ms:.0f} ms"


def classify(text, compare):
    if not text.strip():
        return {}, "", {}, ""
    tuned_out, tuned_ms = run(tuned, text)
    if compare:
        base_out, base_ms = run(base_model, text)
    else:
        base_out, base_ms = {}, ""
    return tuned_out, tuned_ms, base_out, base_ms


EXAMPLES = [
    ["mra order OD48213377 kahan pahuncha jaldi", True],
    ["paise kat gaye par order place nahi hua", True],
    ["card chori ho gaya, jaldi block karo", True],
    ["size chhota hai, KX99812345 return karna hai", True],
    ["bot se baat nahi karni, kisi insaan se connect karo", True],
]

with gr.Blocks(title="Hinglish Support Intent") as demo:
    gr.Markdown(
        "# Hinglish Support Intent\n"
        "Qwen2.5-0.5B fine-tuned with **QLoRA** to read customer messages in Hinglish, Romanized Hindi or "
        "English and return the intent and order ID as JSON. "
        "[Code & results](https://github.com/Abhisheklpu1731/hinglish-intent-qlora)"
    )
    with gr.Row():
        text = gr.Textbox(label="Customer message", placeholder="mera order OD12345678 abhi tak nahi aaya", lines=2, scale=4)
        compare = gr.Checkbox(label="Compare with base model", value=True, scale=1)
    btn = gr.Button("Classify", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Fine-tuned (QLoRA)")
            tuned_json = gr.JSON(label="Output")
            tuned_ms = gr.Textbox(label="Latency (CPU)", interactive=False)
        with gr.Column():
            gr.Markdown("### Base model (zero-shot)")
            base_json = gr.JSON(label="Output")
            base_ms = gr.Textbox(label="Latency (CPU)", interactive=False)
    gr.Examples(EXAMPLES, inputs=[text, compare])
    btn.click(classify, [text, compare], [tuned_json, tuned_ms, base_json, base_ms])
    text.submit(classify, [text, compare], [tuned_json, tuned_ms, base_json, base_ms])

if __name__ == "__main__":
    demo.launch()
