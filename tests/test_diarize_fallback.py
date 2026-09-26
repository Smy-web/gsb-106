"""diarize 的兜底路径（pyannote 缺失时走 _mock_diarize）口径。

本环境没有 pyannote.audio，import 立即失败，diarize 每次都会落进兜底分支，
这些用例钉死这条路径的全部可观察行为。
"""
import pytest

from conftest import make_transcription


class TestFallbackLabels:
    def test_labels_alternate_by_segment_index(self, diarizer, audio_path):
        segments = [make_transcription(i * 5.0, i * 5.0 + 4.0, f"第{i}句") for i in range(4)]
        result = diarizer.diarize(audio_path, segments)
        speakers = [s["speaker"] for s in result["speaker_segments"]]
        assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", "SPEAKER_01"]

    def test_segment_fields_copied_from_input(self, diarizer, audio_path):
        segments = [make_transcription(3.5, 7.25, "任意")]
        result = diarizer.diarize(audio_path, segments)
        seg = result["speaker_segments"][0]
        assert seg == {"start": 3.5, "end": 7.25, "speaker": "SPEAKER_00"}


class TestFallbackNumSpeakers:
    def test_num_speakers_hardcoded_two_with_many_segments(self, diarizer, audio_path):
        segments = [make_transcription(float(i), float(i) + 1.0, "x") for i in range(7)]
        assert diarizer.diarize(audio_path, segments)["num_speakers"] == 2

    def test_num_speakers_two_even_with_single_segment(self, diarizer, audio_path):
        # 只有一段发言也报 2 个说话人
        result = diarizer.diarize(audio_path, [make_transcription(0.0, 1.0, "x")])
        assert len(result["speaker_segments"]) == 1
        assert result["num_speakers"] == 2

    def test_num_speakers_two_even_with_zero_segments(self, diarizer, audio_path):
        # 零段：没有任何说话人段，仍报 2 个说话人
        result = diarizer.diarize(audio_path, [])
        assert result["speaker_segments"] == []
        assert result["num_speakers"] == 2


class TestFallbackDirtyInput:
    def test_segments_none_raises_type_error(self, diarizer, audio_path):
        with pytest.raises(TypeError):
            diarizer.diarize(audio_path, None)

    def test_missing_start_raises_key_error(self, diarizer, audio_path):
        with pytest.raises(KeyError):
            diarizer.diarize(audio_path, [{"end": 1.0, "text": "x"}])

    def test_missing_end_raises_key_error(self, diarizer, audio_path):
        with pytest.raises(KeyError):
            diarizer.diarize(audio_path, [{"start": 0.0, "text": "x"}])


class TestCallerCannotDistinguish:
    def test_return_schema_has_no_fallback_marker(self, diarizer, audio_path):
        """钉死结论：调用方无法从返回值分辨「真分离」与「兜底」。

        兜底路径返回的键集合与真分离路径完全相同
        （{"speaker_segments", "num_speakers"}），没有任何标记字段；
        num_speakers 在兜底路径恒为 2，但真分离也可能恰好是 2，
        因此没有任何一个字段能可靠区分两条路——这正是运维日志里
        看不出走了哪条路的根因（见 README 缺陷清单）。
        """
        result = diarizer.diarize(audio_path, [make_transcription(0.0, 1.0, "x")])
        assert set(result.keys()) == {"speaker_segments", "num_speakers"}
