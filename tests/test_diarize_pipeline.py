"""diarize 的真分离路径：用标准库替身 pipeline 把该分支跑起来。

不装 pyannote、不联网、不下载模型；直接往实例上塞假 pipeline。
"""
from conftest import ExplodingPipeline, FakePipeline, make_transcription


class TestRealPipelinePath:
    def test_segments_come_from_itertracks(self, diarizer, audio_path, monkeypatch):
        fake = FakePipeline([(0.0, 4.2, "SPEAKER_00"), (4.2, 9.0, "SPEAKER_01")])
        monkeypatch.setattr(diarizer, "pipeline", fake)
        result = diarizer.diarize(audio_path, [])
        assert fake.calls == [audio_path]
        assert result["speaker_segments"] == [
            {"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"},
            {"start": 4.2, "end": 9.0, "speaker": "SPEAKER_01"},
        ]

    def test_num_speakers_counts_distinct_speakers(self, diarizer, audio_path, monkeypatch):
        fake = FakePipeline([
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.0, "SPEAKER_01"),
            (2.0, 3.0, "SPEAKER_00"),
        ])
        monkeypatch.setattr(diarizer, "pipeline", fake)
        assert diarizer.diarize(audio_path, [])["num_speakers"] == 2

    def test_num_speakers_one_when_single_speaker(self, diarizer, audio_path, monkeypatch):
        fake = FakePipeline([(0.0, 5.0, "SPEAKER_00")])
        monkeypatch.setattr(diarizer, "pipeline", fake)
        assert diarizer.diarize(audio_path, [])["num_speakers"] == 1

    def test_num_speakers_zero_when_no_tracks(self, diarizer, audio_path, monkeypatch):
        fake = FakePipeline([])
        monkeypatch.setattr(diarizer, "pipeline", fake)
        result = diarizer.diarize(audio_path, [])
        assert result["speaker_segments"] == []
        assert result["num_speakers"] == 0

    def test_num_speakers_semantics_differ_between_paths(self, diarizer, audio_path, monkeypatch):
        """钉死两条路的 num_speakers 口径差：真分离按实际去重人数，
        兜底恒为 2。同一场「只有一人发言」的会，两条路分别报 1 和 2。"""
        fake = FakePipeline([(0.0, 5.0, "SPEAKER_00")])
        monkeypatch.setattr(diarizer, "pipeline", fake)
        real_result = diarizer.diarize(audio_path, [])

        monkeypatch.setattr(diarizer, "pipeline", None)  # 强制走兜底
        fallback_result = diarizer.diarize(audio_path, [make_transcription(0.0, 5.0, "x")])

        assert real_result["num_speakers"] == 1
        assert fallback_result["num_speakers"] == 2


class TestFallbackFromPipelineFailure:
    def test_pipeline_exception_falls_back_to_mock(self, diarizer, audio_path, monkeypatch):
        """替身自己抛异常时，diarize 静默落到兜底路径（异常被吞）。"""
        monkeypatch.setattr(diarizer, "pipeline", ExplodingPipeline())
        segments = [make_transcription(0.0, 1.0, "x"), make_transcription(1.0, 2.0, "y")]
        result = diarizer.diarize(audio_path, segments)
        assert [s["speaker"] for s in result["speaker_segments"]] == ["SPEAKER_00", "SPEAKER_01"]
        assert result["num_speakers"] == 2


class TestLoadRetryBehavior:
    def test_load_retried_on_every_call_after_failure(self, diarizer, audio_path, monkeypatch):
        """钉死：加载失败后 pipeline 仍是 None，每次 diarize 调用都会重试加载，
        不是「一个实例只试一次」。"""
        attempts = []

        def failing_load():
            attempts.append(1)
            raise ImportError("No module named 'pyannote'")

        monkeypatch.setattr(diarizer, "_load_pipeline", failing_load)
        segment = [make_transcription(0.0, 1.0, "x")]
        diarizer.diarize(audio_path, segment)
        diarizer.diarize(audio_path, segment)
        diarizer.diarize(audio_path, segment)
        assert len(attempts) == 3

    def test_pipeline_not_reloaded_once_set(self, diarizer, audio_path, monkeypatch):
        """pipeline 已就位时 _load_pipeline 是空操作，不会重复加载。"""
        fake = FakePipeline([(0.0, 1.0, "SPEAKER_00")])
        monkeypatch.setattr(diarizer, "pipeline", fake)
        diarizer.diarize(audio_path, [])
        diarizer.diarize(audio_path, [])
        assert diarizer.pipeline is fake
        assert fake.calls == [audio_path, audio_path]
