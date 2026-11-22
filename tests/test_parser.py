import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from voicearm.nlp.parser import OllamaParser, RuleParser, extract_color  # noqa: E402

ROWS = [json.loads(line) for line in open(ROOT / "data/commands.jsonl", encoding="utf-8")]


@pytest.mark.parametrize("row", [r for r in ROWS if r["category"] != "zor"], ids=lambda r: r["text"])
def test_rule_parser_dataset(row):
    assert [t.to_dict() for t in RuleParser().parse(row["text"])] == row["expected"]


def test_extract_color_from_answer():
    assert extract_color("Kırmızı olanı") == "kirmizi"
    assert extract_color("bilmiyorum") is None


def test_llm_json_is_normalized():
    tasks = OllamaParser.from_json(
        '{"steps": [{"action": "pick", "object_color": "Kırmızı", "object_kind": "kup",'
        ' "target_color": null, "target_kind": null}, {"action": "uç"}]}')
    assert len(tasks) == 1 and tasks[0].object_color == "kirmizi"
