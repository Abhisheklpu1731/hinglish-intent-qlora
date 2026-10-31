"""Live demo: Hinglish support-intent model, fine-tuned vs base, side by side.

Runs on Streamlit Community Cloud's free CPU. One copy of Qwen2.5-0.5B is loaded in bf16
(~1 GB); the base-model answer comes from the same weights with the LoRA adapter switched off,
so the comparison costs no extra memory.

    streamlit run demo/streamlit_app.py
"""

import json
import os
import time

import streamlit as st
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
EXAMPLES = [
    "mra order OD48213377 kahan pahuncha jaldi",
    "paise kat gaye par order place nahi hua",
    "UPI se pay kiya, transaction failed dikha raha hai",
    "kisi insaan se baat karni hai",
    "size chhota hai, KX99812345 return karna hai",
    "call back karwa do kisi executive se",
]

st.set_page_config(page_title="Hinglish Support Intent", page_icon="💬", layout="wide")
try:
    ADAPTER = st.secrets.get("ADAPTER_ID", ADAPTER)
except Exception:   # no secrets file
    pass


@st.cache_resource(show_spinner="Loading Qwen2.5-0.5B and the LoRA adapter (first load takes about a minute)…")
def load():
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    tok = AutoTokenizer.from_pretrained(BASE)
    dtype = torch.bfloat16 if os.getenv("DEMO_DTYPE", "bf16") == "bf16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=dtype, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, ADAPTER).eval()
    return tok, model


@torch.no_grad()
def classify(tok, model, text, use_adapter):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt")
    t0 = time.perf_counter()
    if use_adapter:
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    else:
        with model.disable_adapter():      # same weights, LoRA switched off = the base model
            out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    secs = time.perf_counter() - t0
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        parsed = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        parsed = None
    return parsed, raw, secs


def show(col, title, parsed, raw, secs):
    with col:
        st.subheader(title)
        if parsed is None:
            st.error("Not valid JSON")
            st.code(raw)
            return
        intent = parsed.get("intent")
        valid = intent in INTENTS
        st.metric("Intent", intent or "—", delta=None if valid else "not an allowed label",
                  delta_color="off" if valid else "inverse")
        st.metric("Order ID", parsed.get("order_id") or "none")
        st.code(json.dumps(parsed, ensure_ascii=False), language="json")
        st.caption(f"{secs:.1f} s on CPU")


st.title("💬 Hinglish Support Intent")
st.markdown(
    "Qwen2.5-0.5B fine-tuned with **LoRA** to read customer messages in Hinglish, Romanized Hindi or "
    "English and return the intent and order ID as JSON. "
    "On 240 held-out messages it scores **95.0%** intent accuracy (base model: 55.4%) and **100%** "
    "order-ID extraction (base: 76.7%). "
    "[Code, data and results](https://github.com/Abhisheklpu1731/hinglish-intent-qlora)"
)

with st.expander("Allowed intents"):
    st.write(", ".join(f"`{i}`" for i in INTENTS))

example = st.pills("Try an example", EXAMPLES, selection_mode="single")
text = st.text_input("Customer message", value=example or "", placeholder="mera order OD12345678 abhi tak nahi aaya")
compare = st.toggle("Compare with the base model", value=True)

if st.button("Classify", type="primary", disabled=not text.strip()):
    tok, model = load()
    left, right = st.columns(2)
    with st.spinner("Fine-tuned model thinking…"):
        show(left, "Fine-tuned (LoRA)", *classify(tok, model, text, use_adapter=True))
    if compare:
        with st.spinner("Base model thinking…"):
            show(right, "Base model (zero-shot)", *classify(tok, model, text, use_adapter=False))
else:
    st.info("Pick an example or type a message, then click **Classify**. The first run loads the model (about a minute).")
