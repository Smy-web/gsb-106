"""逐条钉住 classify_speaker_roles 的口径。

实现要点（被以下断言钉死）：
- 时间窗只比 start 字段：abs(spk.start - trans.start) < 1.0，严格小于；
- 一个说话人段附近有多条转写时取第一条（命中即 break）；
- 同一说话人多段发言的文本会累加；
- 计票是「命中了多少个不同关键词」，不是出现次数；
- 互为子串的关键词（货架/货架号 等）会让同一句话重复计票；
- 平票（含 0:0）判「其他参会方」。
"""
import pytest


def _classify(diarizer, speaker_segments, transcription_segments):
    return diarizer.classify_speaker_roles(speaker_segments, transcription_segments)


# ---------- 时间窗口 ----------

def test_window_compares_start_fields_not_overlap(diarizer):
    """转写段完全落在说话人段内部也不算命中：比的是 start 对 start。"""
    spk = [{"start": 0.0, "end": 10.0, "speaker": "S"}]
    trans = [{"start": 5.0, "end": 5.5, "text": "货架要重排"}]
    result = _classify(diarizer, spk, trans)
    assert result["speaker_roles"]["S"]["role"] == "其他参会方"
    assert result["speaker_roles"]["S"]["storage_keyword_count"] == 0


def test_window_threshold_is_one_second_exclusive(diarizer):
    """阈值 1.0 秒，严格小于：差值正好 1.0 不算在窗口内。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    inside = [{"start": 0.999, "end": 1.5, "text": "货架"}]
    on_edge = [{"start": 1.0, "end": 1.5, "text": "货架"}]
    assert _classify(diarizer, spk, inside)["speaker_roles"]["S"]["storage_keyword_count"] == 1
    assert _classify(diarizer, spk, on_edge)["speaker_roles"]["S"]["storage_keyword_count"] == 0


def test_window_is_symmetric_around_speaker_start(diarizer):
    """转写比说话人段更早开始、差值在 1 秒内也算命中（abs 对称）。"""
    spk = [{"start": 5.0, "end": 6.0, "speaker": "S"}]
    trans = [{"start": 4.5, "end": 4.9, "text": "库存"}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["storage_keyword_count"] == 1


def test_first_matching_transcription_wins(diarizer):
    """一个说话人段附近有多条转写时取第一条，后面的不参与计票。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.2, "end": 0.6, "text": "货架"},
             {"start": 0.4, "end": 0.8, "text": "运输"}]
    roles = _classify(diarizer, spk, trans)["speaker_roles"]["S"]
    assert roles["storage_keyword_count"] == 1
    assert roles["logistics_keyword_count"] == 0
    assert roles["role"] == "仓储方"


# ---------- 文本累加与计票 ----------

def test_text_accumulates_across_segments_of_same_speaker(diarizer):
    """同一说话人分几段发言时文本累加：两段各命中一个词，合计 1:1 平票。
    若不累加（只取某一段），结果会是仓储方或物流方而非其他参会方。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"},
           {"start": 10.0, "end": 11.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架"},
             {"start": 10.1, "end": 10.5, "text": "运输"}]
    roles = _classify(diarizer, spk, trans)["speaker_roles"]["S"]
    assert roles["storage_keyword_count"] == 1
    assert roles["logistics_keyword_count"] == 1
    assert roles["role"] == "其他参会方"


def test_keyword_count_counts_distinct_keywords_not_occurrences(diarizer):
    """计票按「命中多少个不同关键词」：货架出现 3 次也只计 1 票。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架货架货架"}]
    roles = _classify(diarizer, spk, trans)["speaker_roles"]["S"]
    assert roles["storage_keyword_count"] == 1


def test_substring_keywords_double_count_storage(diarizer):
    """「货架」与「货架号」互为子串且同收表中：一句「货架号」计 2 票。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架号得重排"}]
    roles = _classify(diarizer, spk, trans)["speaker_roles"]["S"]
    assert roles["storage_keyword_count"] == 2
    assert roles["role"] == "仓储方"


def test_substring_keywords_double_count_logistics(diarizer):
    """物流侧同样：「配送中心」同时命中「配送」与「配送中心」，计 2 票。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "配送中心的车"}]
    roles = _classify(diarizer, spk, trans)["speaker_roles"]["S"]
    assert roles["logistics_keyword_count"] == 2
    assert roles["role"] == "物流方"


# ---------- 判定结果 ----------

def test_storage_majority_yields_warehouse(diarizer):
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架和库存都要管"}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["role"] == "仓储方"


def test_logistics_majority_yields_logistics(diarizer):
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "运输和配送要抓紧"}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["role"] == "物流方"


def test_tie_yields_other_participant(diarizer):
    """平票（1:1）判「其他参会方」。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架和运输"}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["role"] == "其他参会方"


def test_zero_keyword_hits_yields_other_participant(diarizer):
    """一个关键词都没命中（0:0 也是平票）判「其他参会方」。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "今天主要同步一下进度"}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["role"] == "其他参会方"


def test_empty_transcription_text_yields_other_participant(diarizer):
    """转写文本是空串：0:0，判「其他参会方」。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": ""}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["role"] == "其他参会方"


def test_no_matching_transcription_yields_other_participant(diarizer):
    """说话人段没命中任何转写（文本为空字符串）：判「其他参会方」。"""
    spk = [{"start": 50.0, "end": 51.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架"}]
    assert _classify(diarizer, spk, trans)["speaker_roles"]["S"]["role"] == "其他参会方"


# ---------- 输出形状 ----------

def test_speaker_roles_entry_shape(diarizer):
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架号"}]
    entry = _classify(diarizer, spk, trans)["speaker_roles"]["S"]
    assert set(entry.keys()) == {"role", "storage_keyword_count", "logistics_keyword_count"}
    assert entry == {"role": "仓储方", "storage_keyword_count": 2,
                     "logistics_keyword_count": 0}


def test_role_segments_keys_and_order(diarizer):
    """role_segments 每项 = 原说话人段的键（保持原顺序）+ 末尾追加 role。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"}]
    result = _classify(diarizer, spk, [])
    assert list(result["role_segments"][0].keys()) == ["start", "end", "speaker", "role"]


def test_role_segments_preserve_extra_input_keys(diarizer):
    """说话人段上的额外键会被 **spk_seg 原样带进 role_segments。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S", "confidence": 0.9}]
    result = _classify(diarizer, spk, [])
    assert result["role_segments"][0]["confidence"] == 0.9


def test_default_unknown_role_branch_is_unreachable(diarizer):
    """classify 里 speaker_roles.get(spk, {"role": "未知"}) 的默认分支不可达：
    speaker_roles 由同一份 speaker_segments 构建，每个 speaker 必有条目。
    因此 classify 的输出里永远不会出现「未知」——没词可判时给的是「其他参会方」。
    （「未知」只会出现在 merge_transcription_with_roles 里。）"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"},
           {"start": 100.0, "end": 101.0, "speaker": "T"}]
    result = _classify(diarizer, spk, [])
    roles = [seg["role"] for seg in result["role_segments"]]
    assert roles == ["其他参会方", "其他参会方"]
    assert "未知" not in roles
