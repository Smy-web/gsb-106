"""缺键与脏输入：哪些缺键抛 KeyError、哪些不抛，以及影响范围。

关键结论：所有 KeyError 都直接向上抛出，没有任何局部容错——
一处缺键不是「只坏一行」，而是整场会拿不到任何结果。
"""
import pytest

from conftest import make_speaker_segment, make_transcription


class TestClassifyMissingKeys:
    def test_speaker_segment_missing_speaker_raises(self, diarizer):
        with pytest.raises(KeyError):
            diarizer.classify_speaker_roles([{"start": 0.0, "end": 1.0}], [])

    def test_transcription_missing_start_raises_when_speaker_segments_nonempty(self, diarizer):
        spk = [make_speaker_segment(0.0, 1.0, "SPEAKER_00")]
        with pytest.raises(KeyError):
            diarizer.classify_speaker_roles(spk, [{"end": 1.0, "text": "x"}])

    def test_transcription_missing_start_ok_when_speaker_segments_empty(self, diarizer):
        """钉死「列表为空时不抛、非空时才抛」的差异：
        speaker_segments 为空时内层循环不执行，缺 start 的转写段不会被触碰。"""
        result = diarizer.classify_speaker_roles([], [{"end": 1.0, "text": "x"}])
        assert result == {"speaker_roles": {}, "role_segments": []}

    def test_transcription_missing_text_raises_only_when_window_matches(self, diarizer):
        spk = [make_speaker_segment(0.0, 1.0, "SPEAKER_00")]
        with pytest.raises(KeyError):
            diarizer.classify_speaker_roles(spk, [{"start": 0.5, "end": 1.0}])

    def test_transcription_missing_text_ok_when_window_misses(self, diarizer):
        # 窗口外：text 字段根本不会被读取，缺了也不抛
        spk = [make_speaker_segment(0.0, 1.0, "SPEAKER_00")]
        result = diarizer.classify_speaker_roles(spk, [{"start": 50.0, "end": 51.0}])
        assert result["speaker_roles"]["SPEAKER_00"]["role"] == "其他参会方"

    def test_one_bad_segment_fails_whole_meeting(self, diarizer):
        """影响范围钉死：5 段里只有 1 段缺 start，整个调用抛 KeyError，
        其余 4 段完好的结果也拿不到——不是只坏一行，是整场失败。"""
        spk = [make_speaker_segment(float(i * 10), float(i * 10) + 5.0, f"SPEAKER_0{i % 2}") for i in range(5)]
        trans = [make_transcription(float(i * 10), float(i * 10) + 1.0, "货架") for i in range(4)]
        trans.append({"end": 46.0, "text": "缺 start 的一段"})
        with pytest.raises(KeyError):
            diarizer.classify_speaker_roles(spk, trans)


class TestMergeMissingKeys:
    def test_transcription_missing_start_raises_when_role_segments_nonempty(self, diarizer):
        roles = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "role": "仓储方"}]
        with pytest.raises(KeyError):
            diarizer.merge_transcription_with_roles([{"end": 1.0, "text": "x"}], roles)

    def test_transcription_missing_start_ok_when_role_segments_empty(self, diarizer):
        """钉死「列表为空时不抛、非空时才抛」的差异：
        role_segments 为空时内层循环不执行，缺 start 的转写段不会被触碰。"""
        merged = diarizer.merge_transcription_with_roles([{"end": 1.0, "text": "x"}], [])
        assert merged[0]["role"] == "未知"

    def test_role_segment_missing_start_raises(self, diarizer):
        trans = [make_transcription(0.0, 1.0, "x")]
        with pytest.raises(KeyError):
            diarizer.merge_transcription_with_roles(trans, [{"end": 1.0, "speaker": "S", "role": "仓储方"}])

    def test_role_segment_missing_role_raises_when_window_matches(self, diarizer):
        trans = [make_transcription(0.0, 1.0, "x")]
        bad = [{"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}]
        with pytest.raises(KeyError):
            diarizer.merge_transcription_with_roles(trans, bad)

    def test_role_segment_missing_role_ok_when_window_misses(self, diarizer):
        # 窗口外：role 字段不会被读取，缺了也不抛，转写段拿「未知」
        trans = [make_transcription(100.0, 101.0, "x")]
        bad = [{"start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"}]
        merged = diarizer.merge_transcription_with_roles(trans, bad)
        assert merged[0]["role"] == "未知"

    def test_one_bad_transcription_fails_whole_merge(self, diarizer):
        """影响范围钉死：5 段转写里 1 段缺 start，整个 merge 抛 KeyError，
        一行结果都拿不到。"""
        trans = [make_transcription(float(i * 10), float(i * 10) + 1.0, f"第{i}句") for i in range(4)]
        trans.append({"end": 46.0, "text": "缺 start 的一段"})
        roles = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "role": "仓储方"}]
        with pytest.raises(KeyError):
            diarizer.merge_transcription_with_roles(trans, roles)
