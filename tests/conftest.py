"""共享夹具与标准库替身。

只使用内存中的合成数据；音频路径用 tmp_path 下的占位空文件，
任何测试都不读写真实音频、不联网、不安装任何包。
"""
import pytest

from backend.core.speaker_diarizer import SpeakerDiarizer


@pytest.fixture
def diarizer():
    return SpeakerDiarizer()


@pytest.fixture
def audio_path(tmp_path):
    """占位空文件，仅作为音频路径参数；pyannote 不存在，绝不会被真正读取。"""
    placeholder = tmp_path / "placeholder.wav"
    placeholder.touch()
    return str(placeholder)


class FakeTurn:
    """模拟 pyannote 的时间区间对象，只带 start/end。"""

    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    """模拟 pyannote 的分离结果，记录 itertracks 的调用参数。"""

    def __init__(self, tracks):
        self._tracks = tracks
        self.itertracks_kwargs = []

    def itertracks(self, yield_label=False):
        self.itertracks_kwargs.append({"yield_label": yield_label})
        yield from self._tracks


class FakePipeline:
    """可调用替身：__call__(audio_path) 返回 FakeDiarization。

    tracks 形如 [(FakeTurn, None, "SPEAKER_00"), ...]。
    """

    def __init__(self, tracks):
        self._tracks = tracks
        self.calls = []
        self.diarization = FakeDiarization(tracks)

    def __call__(self, audio_path):
        self.calls.append(audio_path)
        return self.diarization


class ExplodingPipeline:
    """调用即抛异常的替身，用来钉住「真分离中途失败会落到兜底」。"""

    def __call__(self, audio_path):
        raise RuntimeError("fake pipeline exploded")


def make_tracks(spec):
    """spec: [(start, end, speaker), ...] -> pyannote 风格的 tracks。"""
    return [(FakeTurn(s, e), None, spk) for s, e, spk in spec]
