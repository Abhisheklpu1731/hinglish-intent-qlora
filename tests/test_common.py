import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from common import compute_metrics, parse_output, to_prompt_completion  # noqa: E402


def test_parse_clean_json():
    assert parse_output('{"intent": "card_block", "order_id": null}') == ("card_block", None, True)


def test_parse_json_with_surrounding_text():
    raw = 'Sure! {"intent": "refund_request", "order_id": "OD12345678"} hope this helps'
    assert parse_output(raw) == ("refund_request", "OD12345678", True)


def test_parse_string_null_order_id():
    assert parse_output('{"intent": "talk_to_agent", "order_id": "null"}')[1] is None


def test_parse_invalid():
    assert parse_output("I think it's a refund") == (None, None, False)


def test_metrics():
    rows = [
        {"text": "a", "intent": "card_block", "order_id": None},
        {"text": "b", "intent": "refund_request", "order_id": "OD1"},
    ]
    preds = ['{"intent": "card_block", "order_id": null}', "garbage"]
    m = compute_metrics(rows, preds)
    assert m["json_valid"] == 0.5
    assert m["intent_accuracy"] == 0.5
    assert m["order_id_exact"] == 0.5


def test_prompt_completion_format():
    ex = to_prompt_completion({"text": "card block karo", "intent": "card_block", "order_id": None})
    assert ex["prompt"][-1] == {"role": "user", "content": "card block karo"}
    assert ex["completion"][0]["role"] == "assistant"
    assert '"intent": "card_block"' in ex["completion"][0]["content"]
