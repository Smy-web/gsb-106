"""钉住 diarize 在说话人分离不可用（pyannote 缺失）时走的兜底路径。

本环境里 pyannote.audio 未安装，_load_pipeline 的 import 立即失败，
diarize 的 except 分支把它吞掉并转去 _mock_diarize —— 以下全部断言
针对这条兜底路径的实际行为。
"""
import pytest

from backend.core.speaker_diarizer import SpeakerDiarizer


def _segments(n):
    return [{"start": float(i * 10), "end": float(i * 10) + 5.0, "text": f"第{i}段"}
            for i in range(n)]


def test_fallback_labels_alternate_by_segment_index(diarizer, audio_path):
    """兜底按段下标奇偶发标签：偶数段 SPEAKER_00，奇数段 SPEAKER_01。"""
    result = diarizer.diarize(audio_path, _segments(4))
    speakers = [s["speaker"] for s in result["speaker_segments"]]
    assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]


def test_fallback_copies_start_end_from_input(diarizer, audio_path):
    """兜底段的 start/end 直接抄自输入转写段。"""
    result = diarizer.diarize(audio_path, _segments(3))
    for src, out in zip(_segments(3), result["speaker_segments"]):
        assert out["start"] == src["start"]
        assert out["end"] == src["end"]
        assert set(out.keys()) == {"start", "end", "speaker"}


def test_fallback_num_speakers_hardcoded_two_with_many_segments(diarizer, audio_path):
    """5 段发言也只报 2 个说话人（写死的常量，不是数出来的）。"""
    result = diarizer.diarize(audio_path, _segments(5))
    assert result["num_speakers"] == 2


def test_fallback_num_speakers_two_even_with_one_segment(diarizer, audio_path):
    """只有一段发言（标签只有 SPEAKER_00）也报 num_speakers == 2。"""
    result = diarizer.diarize(audio_path, _segments(1))
    assert [s["speaker"] for s in result["speaker_segments"]] == ["SPEAKER_00"]
    assert result["num_speakers"] == 2


def test_fallback_num_speakers_two_even_with_zero_segments(diarizer, audio_path):
    """零段发言：speaker_segments 为空，num_speakers 仍然报 2。"""
    result = diarizer.diarize(audio_path, [])
    assert result["speaker_segments"] == []
    assert result["num_speakers"] == 2


def test_fallback_segments_none_raises_typeerror(diarizer, audio_path):
    """segments 传 None：except 只兜加载/推理的异常，兜不住 enumerate(None)。"""
    with pytest.raises(TypeError):
        diarizer.diarize(audio_path, None)


def test_fallback_segments_omitted_raises_typeerror(diarizer, audio_path):
    """不传 segments（默认 None）同样抛 TypeError。"""
    with pytest.raises(TypeError):
        diarizer.diarize(audio_path)


def test_fallback_segment_missing_start_raises_keyerror(diarizer, audio_path):
    with pytest.raises(KeyError):
        diarizer.diarize(audio_path, [{"end": 1.0}])


def test_fallback_segment_missing_end_raises_keyerror(diarizer, audio_path):
    with pytest.raises(KeyError):
        diarizer.diarize(audio_path, [{"start": 0.0}])


def test_fallback_result_has_no_path_marker(diarizer, audio_path):
    """调用方无法从返回值分辨真分离与兜底：顶层只有这两个键，无任何标记。"""
    result = diarizer.diarize(audio_path, _segments(2))
    assert set(result.keys()) == {"speaker_segments", "num_speakers"}


def test_failed_load_is_retried_on_every_call(diarizer, audio_path, monkeypatch):
    """加载失败后 pipeline 仍是 None，下一次 diarize 会再试一次加载：
    不是「一个实例只试一次」，而是「每次调用都试一遍」。"""
    attempts = []

    def failing_load():
        attempts.append(1)
        raise ImportError("No module named 'pyannote'")

    monkeypatch.setattr(diarizer, "_load_pipeline", failing_load)
    for _ in range(3):
        diarizer.diarize(audio_path, _segments(1))
    assert len(attempts) == 3
    assert diarizer.pipeline is None
