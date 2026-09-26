"""逐条钉住 merge_transcription_with_roles 的口径。

实现要点（被以下断言钉死）：
- 窗口：abs(trans.start - role_seg.start) < 2.0，严格小于，abs 对称；
- 窗口内多条候选取 role_segments 里的第一条，不是最近的一条；
- 匹配不上填「未知」（与 classify 的「其他参会方」不是同一个词）；
- 转写段自带的 role 会被覆盖（包括被覆盖成「未知」）；
- 输出顺序跟随输入转写段顺序。
"""
import pytest


def _merge(diarizer, trans, role_segments):
    return diarizer.merge_transcription_with_roles(trans, role_segments)


def test_window_threshold_is_two_seconds_exclusive(diarizer):
    """阈值 2.0 秒，严格小于：差值正好 2.0 不匹配。"""
    trans = [{"start": 10.0, "end": 10.5, "text": "x"}]
    inside = [{"start": 11.999, "end": 12.5, "speaker": "S", "role": "仓储方"}]
    on_edge = [{"start": 12.0, "end": 12.5, "speaker": "S", "role": "仓储方"}]
    assert _merge(diarizer, trans, inside)[0]["role"] == "仓储方"
    assert _merge(diarizer, trans, on_edge)[0]["role"] == "未知"


def test_window_is_symmetric_role_segment_may_start_earlier(diarizer):
    """说话人段比转写更早开始（差值在 2 秒内）也算命中。"""
    trans = [{"start": 10.0, "end": 10.5, "text": "x"}]
    role_segments = [{"start": 8.5, "end": 9.0, "speaker": "S", "role": "物流方"}]
    assert _merge(diarizer, trans, role_segments)[0]["role"] == "物流方"


def test_first_candidate_wins_not_nearest(diarizer):
    """窗口内多条候选取第一条，不是最近的一条。

    用背景里那场布局优化会的形状构造：转写在 10.0 秒，
    role_segments 里先出现 11.9 秒的「物流方」（差 1.9），
    后出现 10.2 秒的「仓储方」（差 0.2）。取最近应是仓储方，
    实际取第一条 → 物流方。这正是「物流方的话被标成仓储方」
    一类错标在 merge 环节的镜像机理。"""
    trans = [{"start": 10.0, "end": 10.5, "text": "A区货架号得重排"}]
    role_segments = [
        {"start": 11.9, "end": 12.4, "speaker": "S1", "role": "物流方"},
        {"start": 10.2, "end": 10.7, "speaker": "S0", "role": "仓储方"},
    ]
    assert _merge(diarizer, trans, role_segments)[0]["role"] == "物流方"


def test_unmatched_transcription_gets_unknown(diarizer):
    """匹配不上时 role 填「未知」——不是三个判定标签里的任何一个。"""
    trans = [{"start": 100.0, "end": 100.5, "text": "x"}]
    role_segments = [{"start": 0.0, "end": 0.5, "speaker": "S", "role": "仓储方"}]
    assert _merge(diarizer, trans, role_segments)[0]["role"] == "未知"


def test_existing_role_is_overwritten_on_match(diarizer):
    """转写段本来带 role 时会被匹配结果覆盖。"""
    trans = [{"start": 0.0, "end": 0.5, "text": "x", "role": "仓储方"}]
    role_segments = [{"start": 0.1, "end": 0.6, "speaker": "S", "role": "物流方"}]
    assert _merge(diarizer, trans, role_segments)[0]["role"] == "物流方"


def test_existing_role_is_overwritten_to_unknown_on_miss(diarizer):
    """转写段本来带 role、又匹配不上时，会被覆盖成「未知」。"""
    trans = [{"start": 0.0, "end": 0.5, "text": "x", "role": "仓储方"}]
    role_segments = [{"start": 100.0, "end": 100.5, "speaker": "S", "role": "物流方"}]
    assert _merge(diarizer, trans, role_segments)[0]["role"] == "未知"


def test_output_order_follows_transcription_order(diarizer):
    """输出顺序跟随输入转写段顺序，不按时间重排。"""
    trans = [{"start": 20.0, "end": 20.5, "text": "后"},
             {"start": 0.0, "end": 0.5, "text": "先"}]
    role_segments = [{"start": 0.1, "end": 0.6, "speaker": "S", "role": "仓储方"},
                     {"start": 20.1, "end": 20.6, "speaker": "T", "role": "物流方"}]
    merged = _merge(diarizer, trans, role_segments)
    assert [(m["text"], m["role"]) for m in merged] == [("后", "物流方"), ("先", "仓储方")]


def test_output_item_keys(diarizer):
    """输出每项 = 转写段原有的键 + role；不新增其他键。"""
    trans = [{"start": 0.0, "end": 0.5, "text": "x"}]
    role_segments = [{"start": 0.1, "end": 0.6, "speaker": "S", "role": "仓储方"}]
    merged = _merge(diarizer, trans, role_segments)
    assert list(merged[0].keys()) == ["start", "end", "text", "role"]


def test_each_transcription_matched_independently(diarizer):
    """多条转写各自独立匹配，可命中同一条 role_segment。"""
    trans = [{"start": 0.0, "end": 0.4, "text": "a"},
             {"start": 0.5, "end": 0.9, "text": "b"}]
    role_segments = [{"start": 0.2, "end": 0.6, "speaker": "S", "role": "仓储方"}]
    merged = _merge(diarizer, trans, role_segments)
    assert [m["role"] for m in merged] == ["仓储方", "仓储方"]


def test_label_vocabularies_diverge_between_classify_and_merge(diarizer):
    """【已钉住的口径不一致，见 README 缺陷清单第 4 条】

    同一语义「判不出来 / 贴不上」，classify 给「其他参会方」，
    merge 给「未知」。下游按 role 字符串分组，会被拆成两组。
    本测试按现状钉死这两个值；按做题口径这里应当统一，
    统一前此断言即缺陷的量化证据。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    classified = diarizer.classify_speaker_roles(spk, [])
    classify_label = classified["role_segments"][0]["role"]

    trans = [{"start": 100.0, "end": 100.5, "text": "x"}]
    merged = diarizer.merge_transcription_with_roles(trans, classified["role_segments"])
    merge_label = merged[0]["role"]

    assert classify_label == "其他参会方"
    assert merge_label == "未知"
    assert classify_label != merge_label  # 不一致本身是现状，列为待修缺陷
