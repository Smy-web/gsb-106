"""用标准库替身（monkeypatch 往实例上塞假 pipeline）跑真分离分支，
并对比两条路径的 num_speakers 口径差异。

不安装 pyannote、不联网、不下载模型；假对象只需可调用、并给出
itertracks(yield_label=True) 的迭代结果。
"""
import pytest

from tests.conftest import FakePipeline, ExplodingPipeline, make_tracks


def test_real_path_uses_pipeline_output(diarizer, audio_path):
    """真分离路径：speaker_segments 来自 pipeline 的 itertracks。"""
    fake = FakePipeline(make_tracks([(0.0, 4.0, "SPEAKER_00"),
                                     (4.0, 9.0, "SPEAKER_01"),
                                     (9.0, 12.0, "SPEAKER_00")]))
    diarizer.pipeline = fake
    result = diarizer.diarize(audio_path, [{"start": 0.0, "end": 1.0}])
    assert result["speaker_segments"] == [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 9.0, "speaker": "SPEAKER_01"},
        {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_00"},
    ]
    assert result["num_speakers"] == 2


def test_real_path_num_speakers_counts_distinct_labels(diarizer, audio_path):
    """真分离路径的 num_speakers 是不同标签的数量，不是写死的 2。"""
    fake = FakePipeline(make_tracks([(0.0, 1.0, "SPEAKER_00"),
                                     (1.0, 2.0, "SPEAKER_01"),
                                     (2.0, 3.0, "SPEAKER_02")]))
    diarizer.pipeline = fake
    result = diarizer.diarize(audio_path, [])
    assert result["num_speakers"] == 3


def test_real_path_single_speaker_reports_one(diarizer, audio_path):
    """真分离路径下一个说话人报 1 —— 与兜底「恒为 2」形成口径差。"""
    fake = FakePipeline(make_tracks([(0.0, 3.0, "SPEAKER_00"),
                                     (3.0, 6.0, "SPEAKER_00")]))
    diarizer.pipeline = fake
    result = diarizer.diarize(audio_path, [])
    assert result["num_speakers"] == 1


def test_num_speakers_diverges_between_paths(diarizer, audio_path):
    """同一场会（一个说话人、两段发言）：真分离报 1，兜底报 2。"""
    fake = FakePipeline(make_tracks([(0.0, 3.0, "SPEAKER_00"),
                                     (3.0, 6.0, "SPEAKER_00")]))
    diarizer.pipeline = fake
    real = diarizer.diarize(audio_path, [])

    diarizer.pipeline = None  # 强制走兜底
    segments = [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 6.0}]
    fallback = diarizer.diarize(audio_path, segments)

    assert real["num_speakers"] == 1
    assert fallback["num_speakers"] == 2


def test_real_path_result_keys_identical_to_fallback(diarizer, audio_path):
    """两条路径返回值键集合完全一致，调用方无法据此分辨走的哪条路。"""
    fake = FakePipeline(make_tracks([(0.0, 1.0, "SPEAKER_00")]))
    diarizer.pipeline = fake
    real = diarizer.diarize(audio_path, [])

    diarizer.pipeline = None
    fallback = diarizer.diarize(audio_path, [{"start": 0.0, "end": 1.0}])

    assert set(real.keys()) == set(fallback.keys()) == {"speaker_segments", "num_speakers"}


def test_pipeline_called_with_audio_path_and_itertracks_yield_label(diarizer, audio_path):
    """真分离路径把 audio_path 原样传给 pipeline，并以 yield_label=True 取轨道。"""
    fake = FakePipeline(make_tracks([(0.0, 1.0, "SPEAKER_00")]))
    diarizer.pipeline = fake
    diarizer.diarize(audio_path, [])
    assert fake.calls == [audio_path]
    assert fake.diarization.itertracks_kwargs == [{"yield_label": True}]


def test_pipeline_exception_falls_back_to_mock(diarizer, audio_path):
    """替身自己抛异常时，diarize 落到兜底路径（异常被 except 吞掉）。"""
    diarizer.pipeline = ExplodingPipeline()
    result = diarizer.diarize(audio_path, [{"start": 0.0, "end": 1.0},
                                           {"start": 1.0, "end": 2.0}])
    assert [s["speaker"] for s in result["speaker_segments"]] == ["SPEAKER_00", "SPEAKER_01"]
    assert result["num_speakers"] == 2
