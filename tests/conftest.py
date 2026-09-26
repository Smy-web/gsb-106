"""共享夹具与标准库替身。只允许内存合成数据；音频路径用 tmp_path 占位空文件。"""
from types import SimpleNamespace

import pytest

from backend.core.speaker_diarizer import SpeakerDiarizer


@pytest.fixture
def diarizer():
    return SpeakerDiarizer()


@pytest.fixture
def audio_path(tmp_path):
    """占位空文件，仅作为路径实参；两条路径都不会真正读它。"""
    placeholder = tmp_path / "placeholder.wav"
    placeholder.touch()
    return str(placeholder)


class FakePipeline:
    """标准库替身：可被调用，返回带 itertracks(yield_label=True) 的对象。"""

    def __init__(self, tracks):
        # tracks: [(start, end, speaker), ...]
        self._tracks = tracks
        self.calls = []

    def __call__(self, audio_path):
        self.calls.append(audio_path)
        return FakeDiarization(self._tracks)


class FakeDiarization:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=False):
        assert yield_label is True
        for start, end, speaker in self._tracks:
            turn = SimpleNamespace(start=start, end=end)
            yield turn, None, speaker


class ExplodingPipeline:
    """调用即抛异常的替身，用于钉住 diarize 的兜底分支。"""

    def __call__(self, audio_path):
        raise RuntimeError("fake pipeline exploded")


def make_transcription(start, end, text, **extra):
    return {"start": start, "end": end, "text": text, **extra}


def make_speaker_segment(start, end, speaker, **extra):
    return {"start": start, "end": end, "speaker": speaker, **extra}
