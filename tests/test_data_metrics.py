import pytest

from finlora.data import canon, format_target, make_record, parse_gold, parse_prediction
from finlora.metrics import classify_error, evaluate


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "负面金融实体有以下几个：小资钱包;北京正聚源通鼎信用管理有限公司",
            ["小资钱包", "北京正聚源通鼎信用管理有限公司"],
        ),
        ("文中包含的负面主体：北银创投", ["北银创投"]),
        ("以上文本不存在负面主体。", []),
        ("分析上述文本，发现没有负面金融实体。", []),
    ],
)
def test_parse_gold(raw, expected):
    assert parse_gold(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("小资钱包;北银创投", ["小资钱包", "北银创投"]),
        ("小资钱包；北银创投。", ["小资钱包", "北银创投"]),
        ("小资钱包、北银创投", ["小资钱包", "北银创投"]),
        ("负面主体：小资钱包", ["小资钱包"]),
        ("无", []),
        ("无。", []),
        ("没有负面主体", []),
        ("", []),
        ("小资钱包;小资钱包", ["小资钱包"]),
        ("小资钱包\n北银创投", ["小资钱包", "北银创投"]),
    ],
)
def test_parse_prediction(raw, expected):
    assert parse_prediction(raw) == expected


def test_target_round_trip():
    for ents in ([], ["甲"], ["甲", "乙公司"]):
        assert parse_prediction(format_target(ents)) == ents


def test_make_record_brackets_and_truncation():
    r = make_record(
        "资易贷（北京）金融信息服务有限公司涉嫌诈骗", "负面主体：资易贷(北京)金融信息服务有限公司"
    )
    assert r["entities"] == ["资易贷(北京)金融信息服务有限公司"]
    assert r["missing_in_text"] == []
    long = make_record("甲" * 400 + "乙公司跑路", "负面主体：乙公司")
    assert long["truncated"] and long["missing_in_text"] == ["乙公司"]


def test_canon():
    assert canon("A（B）") == canon("a(b)")


def test_evaluate_counts():
    recs = [
        {"text": "甲公司跑路，乙公司正常", "entities": ["甲公司"]},
        {"text": "今天天气不错", "entities": []},
        {"text": "丙平台逾期", "entities": ["丙平台"]},
        {"text": "丁公司被立案", "entities": []},
    ]
    preds = [["甲公司"], [], [], ["丁公司"]]
    m = evaluate(recs, preds)
    assert m["entity_precision"] == 0.5
    assert m["entity_recall"] == 0.5
    assert m["exact_match"] == 0.5
    assert m["false_alarm_rate"] == 0.5
    assert m["miss_rate"] == 0.5
    assert m["errors"] == {"正确": 2, "漏报": 1, "误报": 1}


def test_hallucination_counted():
    recs = [{"text": "甲公司跑路", "entities": ["甲公司"]}]
    m = evaluate(recs, [["甲公司", "戊集团"]])
    assert m["hallucination_rate"] == 0.5
    assert m["errors"] == {"输出了原文没有的实体": 1}


def test_postfilter_drops_entities_not_in_text():
    from finlora.postfilter import keep_in_text

    text = "绿化贷、资易贷（北京）金融信息服务有限公司涉嫌非法吸收公众存款"
    assert keep_in_text(text, ["绿意贷", "资易贷(北京)金融信息服务有限公司"]) == [
        "资易贷(北京)金融信息服务有限公司"
    ]


@pytest.mark.parametrize(
    ("gold", "pred", "kind"),
    [
        (["甲公司"], ["甲公司旗下"], "实体边界不对"),
        (["甲公司", "乙公司"], ["甲公司"], "少找了实体"),
        (["甲公司"], ["甲公司", "乙公司"], "多找了实体"),
        (["甲公司"], ["乙公司"], "找错了实体"),
    ],
)
def test_classify_error(gold, pred, kind):
    text = "甲公司旗下平台与乙公司均涉嫌诈骗"
    assert classify_error(text, gold, pred) == kind
