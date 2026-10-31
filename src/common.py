"""Shared prompt format, output parsing and metrics for training and evaluation."""

import json
import re
from collections import Counter

INTENTS = [
    "order_status", "refund_request", "cancel_order", "payment_failed", "card_block",
    "address_change", "delivery_delay", "product_return", "account_login_issue",
    "talk_to_agent",
]

SYSTEM_PROMPT = (
    "You are a customer-support intent classifier for an Indian e-commerce and banking "
    "helpdesk. Customers write in Hinglish, Romanized Hindi or English. "
    "Reply with ONLY a JSON object: {\"intent\": <one of: " + ", ".join(INTENTS) + ">, "
    "\"order_id\": <the order ID exactly as written, or null>}."
)


def prompt_messages(text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def target_json(intent, order_id):
    return json.dumps({"intent": intent, "order_id": order_id}, ensure_ascii=False)


def to_prompt_completion(example):
    """Convert a raw row into TRL's conversational prompt-completion format.
    Loss is computed on the completion only (the JSON answer)."""
    return {
        "prompt": prompt_messages(example["text"]),
        "completion": [{"role": "assistant",
                        "content": target_json(example["intent"], example["order_id"])}],
    }


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_output(raw):
    """Return (intent, order_id, valid_json). Tolerates text around the JSON."""
    m = _JSON_RE.search(raw or "")
    if not m:
        return None, None, False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, None, False
    intent = obj.get("intent")
    oid = obj.get("order_id")
    if isinstance(oid, str) and oid.strip().lower() in ("", "null", "none"):
        oid = None
    return intent, oid, True


def compute_metrics(rows, predictions):
    """rows: list of gold dicts; predictions: list of raw model strings."""
    n = len(rows)
    valid = correct = oid_correct = 0
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold, raw in zip(rows, predictions):
        intent, oid, ok = parse_output(raw)
        valid += ok
        if intent == gold["intent"]:
            correct += 1
            tp[gold["intent"]] += 1
        else:
            fn[gold["intent"]] += 1
            if intent:
                fp[intent] += 1
        oid_correct += (oid == gold["order_id"])

    f1s = []
    for label in INTENTS:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)

    return {
        "n": n,
        "json_valid": round(valid / n, 4),
        "intent_accuracy": round(correct / n, 4),
        "intent_macro_f1": round(sum(f1s) / len(f1s), 4),
        "order_id_exact": round(oid_correct / n, 4),
    }
