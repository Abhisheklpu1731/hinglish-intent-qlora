"""Generate a synthetic Hinglish customer-support dataset.

Each example is a customer message (Hinglish / Romanized Hindi / English) labelled with
an intent and, when present, an order ID. The model is trained to return JSON:
    {"intent": "<label>", "order_id": "<id or null>"}

Leakage control: templates are split per intent BEFORE filling slots, so the test set
uses sentence patterns the model never saw in training (not just new slot values).

Usage:
    python data/generate.py --out data --seed 42
"""

import argparse
import json
import random
from pathlib import Path

# {oid} = order ID, {item} = product, {days} = number of days, {amt} = amount
TEMPLATES = {
    "order_status": [
        "mera order {oid} kahan pahuncha?",
        "order {oid} ka status kya hai bhai",
        "can you check where my order {oid} is",
        "{oid} wala order abhi tak ship hua ya nahi?",
        "mera {item} ka order kab tak aayega",
        "order track karna hai, id {oid}",
        "kya aap bata sakte ho mera parcel kahan hai",
        "status update chahiye order {oid} ka please",
        "maine {item} order kiya tha, uska kya hua",
        "my order {oid} - any update?",
    ],
    "refund_request": [
        "mujhe order {oid} ka refund chahiye",
        "refund kab milega? {days} din ho gaye",
        "paisa wapas karo, order {oid} cancel ho gaya tha",
        "I want my money back for {oid}",
        "{amt} rupaye ka refund abhi tak nahi aaya",
        "refund process start karo please",
        "mera refund pending dikha raha hai {days} din se",
        "return ho gaya {item}, refund kab aayega",
        "please refund order {oid}, product kharab tha",
    ],
    "cancel_order": [
        "order {oid} cancel kar do",
        "mujhe ye order nahi chahiye, cancel karo",
        "please cancel my order {oid}",
        "galti se {item} order ho gaya, cancel karna hai",
        "{oid} cancel karna hai abhi ke abhi",
        "order cancel kaise karu?",
        "cancel kar do bhai, ab zarurat nahi hai",
        "I placed order {oid} by mistake, cancel it",
        "mera {item} wala order cancel kardo",
    ],
    "payment_failed": [
        "payment fail ho gaya par paisa kat gaya",
        "{amt} debit ho gaye but order place nahi hua",
        "UPI se pay kiya, transaction failed dikha raha hai",
        "my payment failed but money was deducted",
        "card se payment nahi ho raha",
        "paise kat gaye, order {oid} pending hai",
        "transaction fail hua {amt} ka, ab kya karu",
        "payment declined aa raha hai baar baar",
        "amount deduct ho gaya par confirmation nahi aaya",
    ],
    "card_block": [
        "mera card block kar do turant",
        "card kho gaya hai, block karna hai",
        "please block my debit card, it got stolen",
        "card chori ho gaya, jaldi block karo",
        "credit card band karna hai",
        "kisi ne mera card use kiya hai, block karo",
        "block my card immediately",
        "debit card lost ho gaya kal",
        "card temporarily block kar sakte ho?",
    ],
    "address_change": [
        "delivery address change karna hai order {oid} ka",
        "address galat daal diya, update karo",
        "can I change the shipping address for {oid}",
        "naya address daalna hai mere order mein",
        "pincode galat hai, address badalna hai",
        "mera address update kar do please",
        "ghar shift ho gaya, address change karna hai",
        "order {oid} dusre address pe bhej sakte ho?",
        "flat number galat likha hai address mein",
    ],
    "delivery_delay": [
        "order {oid} late kyun hai?",
        "{days} din ho gaye abhi tak delivery nahi hui",
        "delivery date nikal gayi, {item} nahi aaya",
        "why is my delivery delayed",
        "expected delivery kal thi, aaj tak nahi aaya",
        "kitna time aur lagega delivery mein, {oid}",
        "bahut late ho raha hai order",
        "delivery boy aaya hi nahi aaj",
        "my {item} was supposed to arrive {days} days ago",
    ],
    "product_return": [
        "mujhe {item} return karna hai",
        "product damaged aaya hai, return karo",
        "size chhota hai, return kaise karu",
        "I want to return the {item} from order {oid}",
        "return pickup schedule karna hai {oid} ka",
        "galat item bhej diya, wapas lo",
        "{item} pasand nahi aaya, return karna hai",
        "return window abhi open hai kya order {oid} ke liye",
        "defective {item} mila hai, replace ya return chahiye",
    ],
    "account_login_issue": [
        "login nahi ho raha mera account mein",
        "OTP nahi aa raha login ke liye",
        "password bhool gaya, reset kaise karu",
        "I can't log in to my account",
        "account locked dikha raha hai",
        "app mein sign in fail ho raha hai",
        "mera account access nahi ho raha",
        "forgot password link kaam nahi kar raha",
        "OTP galat bata raha hai har baar",
    ],
    "talk_to_agent": [
        "kisi insaan se baat karni hai",
        "customer care agent se connect karo",
        "I want to talk to a human",
        "bot se baat nahi karni, agent do",
        "real person se baat karwao please",
        "call back karwa do kisi executive se",
        "manager se baat karni hai",
        "connect me to customer support agent",
        "koi senior se baat karwao",
    ],
}

ITEMS = ["phone", "shoes", "t-shirt", "headphones", "laptop bag", "kurta",
         "mixer grinder", "watch", "saree", "charger", "jeans", "water bottle"]

# Spelling variants common in typed Hinglish
VARIANTS = {
    "nahi": ["nahi", "nhi", "nahin"],
    "kya": ["kya", "kia"],
    "mera": ["mera", "mra", "mera"],
    "hai": ["hai", "h", "hai"],
    "karo": ["karo", "kro", "karo"],
    "please": ["please", "pls", "plz"],
}


def make_oid(rng):
    return rng.choice(["OD", "ORD", "KX", ""]) + str(rng.randint(10**7, 10**9))


def noisify(text, rng):
    words = text.split(" ")
    out = []
    for w in words:
        key = w.lower().strip("?,.!")
        if key in VARIANTS and rng.random() < 0.4:
            w = w.lower().replace(key, rng.choice(VARIANTS[key]))
        out.append(w)
    text = " ".join(out)
    if rng.random() < 0.3:
        text = text.lower()
    if rng.random() < 0.3:
        text = text.rstrip("?.!")
    if rng.random() < 0.15:
        text = text + rng.choice([" jaldi", " urgent", " asap", " !!"])
    return text


def fill(template, rng):
    oid = make_oid(rng) if "{oid}" in template else None
    text = template.format(
        oid=oid or "",
        item=rng.choice(ITEMS),
        days=rng.randint(2, 15),
        amt=rng.choice([299, 499, 799, 1299, 2499, 5999]),
    )
    return noisify(text, rng), oid


def build(per_template_train, per_template_test, seed):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for intent, temps in TEMPLATES.items():
        temps = temps[:]
        rng.shuffle(temps)
        n_test = max(2, len(temps) // 4)          # ~25% of patterns held out
        test_t, train_t = temps[:n_test], temps[n_test:]
        for t in train_t:
            for i in range(per_template_train):
                text, oid = fill(t, rng)
                ex = {"text": text, "intent": intent, "order_id": oid}
                (val if i % 10 == 0 else train).append(ex)
        for t in test_t:
            for _ in range(per_template_test):
                text, oid = fill(t, rng)
                test.append({"text": text, "intent": intent, "order_id": oid})
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--per-template-train", type=int, default=30)
    ap.add_argument("--per-template-test", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, val, test = build(args.per_template_train, args.per_template_test, args.seed)
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(val, out / "val.jsonl")
    write_jsonl(test, out / "test.jsonl")
    print(f"train={len(train)} val={len(val)} test={len(test)} intents={len(TEMPLATES)}")
