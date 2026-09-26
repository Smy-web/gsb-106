"""缺键与脏输入：哪些缺键抛 KeyError、哪些不抛，以及影响范围。

特别注意「列表为空时不抛、非空时才抛」的差异：字段是在内层循环里
才被访问的，外层列表为空时循环体根本不执行。
"""
import pytest


# ---------- classify_speaker_roles ----------

def test_classify_missing_speaker_key_raises(diarizer):
    """speaker 键在循环开头就被访问，必抛 KeyError。"""
    with pytest.raises(KeyError):
        diarizer.classify_speaker_roles([{"start": 0.0, "end": 1.0}], [])


def test_classify_missing_start_ok_when_transcription_empty(diarizer):
    """说话人段缺 start、转写列表为空：内层循环不执行，不抛。"""
    result = diarizer.classify_speaker_roles([{"speaker": "S", "end": 1.0}], [])
    assert result["speaker_roles"]["S"]["role"] == "其他参会方"
    assert "start" not in result["role_segments"][0]


def test_classify_missing_start_raises_when_transcription_nonempty(diarizer):
    """同一个缺键，转写列表非空时就抛 KeyError —— 与上一条成对钉死。"""
    with pytest.raises(KeyError):
        diarizer.classify_speaker_roles(
            [{"speaker": "S", "end": 1.0}],
            [{"start": 0.0, "end": 0.5, "text": "货架"}])


def test_classify_transcription_missing_start_raises(diarizer):
    with pytest.raises(KeyError):
        diarizer.classify_speaker_roles(
            [{"start": 0.0, "end": 1.0, "speaker": "S"}],
            [{"end": 0.5, "text": "货架"}])


def test_classify_transcription_missing_text_raises_when_window_hits(diarizer):
    """转写缺 text：窗口命中时才会访问 text，此时抛 KeyError。"""
    with pytest.raises(KeyError):
        diarizer.classify_speaker_roles(
            [{"start": 0.0, "end": 1.0, "speaker": "S"}],
            [{"start": 0.1, "end": 0.5}])


def test_classify_transcription_missing_text_ok_when_window_misses(diarizer):
    """转写缺 text 但窗口没命中：text 不被访问，不抛。"""
    result = diarizer.classify_speaker_roles(
        [{"start": 0.0, "end": 1.0, "speaker": "S"}],
        [{"start": 50.0, "end": 50.5}])
    assert result["speaker_roles"]["S"]["role"] == "其他参会方"


# ---------- merge_transcription_with_roles ----------

def test_merge_missing_start_ok_when_role_segments_empty(diarizer):
    """转写缺 start、role_segments 为空：内层循环不执行，不抛，role 填「未知」。"""
    merged = diarizer.merge_transcription_with_roles([{"end": 0.5, "text": "x"}], [])
    assert merged[0]["role"] == "未知"


def test_merge_missing_start_raises_when_role_segments_nonempty(diarizer):
    """同一个缺键，role_segments 非空时就抛 KeyError。"""
    with pytest.raises(KeyError):
        diarizer.merge_transcription_with_roles(
            [{"end": 0.5, "text": "x"}],
            [{"start": 0.0, "end": 0.4, "speaker": "S", "role": "仓储方"}])


def test_merge_role_segment_missing_start_raises(diarizer):
    with pytest.raises(KeyError):
        diarizer.merge_transcription_with_roles(
            [{"start": 0.0, "end": 0.5, "text": "x"}],
            [{"end": 0.4, "speaker": "S", "role": "仓储方"}])


def test_merge_role_segment_missing_role_raises_when_matched(diarizer):
    """role_segment 缺 role：窗口命中时才会访问 role，此时抛 KeyError。"""
    with pytest.raises(KeyError):
        diarizer.merge_transcription_with_roles(
            [{"start": 0.0, "end": 0.5, "text": "x"}],
            [{"start": 0.1, "end": 0.4, "speaker": "S"}])


# ---------- 影响范围 ----------

def test_one_bad_segment_fails_entire_classify_call(diarizer):
    """一处缺键不是只坏一行：整个 classify 调用抛异常，整场会拿不到任何结果。"""
    spk = [{"start": 0.0, "end": 1.0, "speaker": "S"},
           {"start": 10.0, "end": 11.0, "speaker": "T"}]
    trans = [{"start": 0.1, "end": 0.5, "text": "货架"},
             {"start": 10.1, "end": 10.5}]  # 缺 text，且窗口命中第二个说话人段
    with pytest.raises(KeyError):
        diarizer.classify_speaker_roles(spk, trans)


def test_one_bad_segment_fails_entire_merge_call(diarizer):
    """merge 同样：任何一条脏数据让整场会的合并结果整体丢失。"""
    trans = [{"start": 0.0, "end": 0.5, "text": "好段"},
             {"end": 1.5, "text": "坏段"}]  # 缺 start
    role_segments = [{"start": 0.1, "end": 0.4, "speaker": "S", "role": "仓储方"}]
    with pytest.raises(KeyError):
        diarizer.merge_transcription_with_roles(trans, role_segments)
