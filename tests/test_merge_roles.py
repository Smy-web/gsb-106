"""merge_transcription_with_roles 的贴角色口径。

实现口径（本文件逐条钉死）：
- 时间窗口：abs(转写段.start - 角色段.start) < 2.0，对称，严格小于；
- 窗口内多条候选时取 role_segments 列表中第一条，不是时间上最近的一条；
- 匹配不上填「未知」；
- 转写段自带的 role 会被覆盖；
- 输出顺序跟随输入转写段顺序。
"""
from conftest import make_speaker_segment, make_transcription


def role_seg(start, role, speaker="SPEAKER_00", end=None):
    return {"start": start, "end": end if end is not None else start + 1.0,
            "speaker": speaker, "role": role}


class TestTimeWindow:
    def test_window_boundary_just_inside(self, diarizer):
        trans = [make_transcription(0.0, 1.0, "x")]
        roles = [role_seg(1.999, "仓储方")]
        assert diarizer.merge_transcription_with_roles(trans, roles)[0]["role"] == "仓储方"

    def test_window_boundary_exactly_two_seconds_is_outside(self, diarizer):
        # 阈值 2.0 严格小于：正好差 2.0 秒不算命中
        trans = [make_transcription(0.0, 1.0, "x")]
        roles = [role_seg(2.0, "仓储方")]
        assert diarizer.merge_transcription_with_roles(trans, roles)[0]["role"] == "未知"

    def test_window_is_symmetric_transcription_may_start_earlier(self, diarizer):
        # 转写比说话人段更早开始也算命中（abs 对称窗口）
        trans = [make_transcription(8.5, 9.5, "x")]
        roles = [role_seg(10.0, "物流方")]
        assert diarizer.merge_transcription_with_roles(trans, roles)[0]["role"] == "物流方"


class TestCandidateSelection:
    def test_first_candidate_wins_not_nearest(self, diarizer):
        """布局优化会事故复现：窗口内多条候选时取列表第一条，而非时间上最近的一条。

        转写「车辆和时效我来盯」（物流内容，start=10.0）同时落在两个角色段
        的 2 秒窗口内：仓储方段（start=11.8，差 1.8s）在列表里靠前，
        物流方段（start=10.1，差 0.1s）才是真正最近的说话人。
        实现取第一条 → 物流方的话被贴上「仓储方」。
        若实现改为「取最近的一条」，本断言会变红——这正是要钉住的现状。
        """
        trans = [make_transcription(10.0, 12.0, "车辆和时效我来盯")]
        roles = [
            role_seg(11.8, "仓储方", speaker="SPEAKER_00"),  # 差 1.8s，列表靠前
            role_seg(10.1, "物流方", speaker="SPEAKER_01"),  # 差 0.1s，时间最近
        ]
        merged = diarizer.merge_transcription_with_roles(trans, roles)
        assert merged[0]["role"] == "仓储方"

class TestFallbackLabel:
    def test_unmatched_gets_unknown(self, diarizer):
        trans = [make_transcription(100.0, 101.0, "x")]
        roles = [role_seg(0.0, "仓储方")]
        assert diarizer.merge_transcription_with_roles(trans, roles)[0]["role"] == "未知"

    def test_empty_role_segments_all_unknown(self, diarizer):
        trans = [make_transcription(0.0, 1.0, "x"), make_transcription(5.0, 6.0, "y")]
        merged = diarizer.merge_transcription_with_roles(trans, [])
        assert [m["role"] for m in merged] == ["未知", "未知"]

    def test_label_vocabularies_differ_pinned_as_current_behavior(self, diarizer):
        """第 6 条的选择：把「两处口径不一致」当现状钉死，并列作待修缺陷。

        classify 只产出「仓储方/物流方/其他参会方」三个标签，
        merge 匹配不上时填的是第四个词「未知」；下游按 role 字符串分组，
        同一场会的责任方会被劈成两套词汇。本断言锁死现状，
        修复时应让 merge 的兜底标签与 classify 的标签体系统一，
        届时同步修改本用例（见 README 缺陷清单）。
        """
        classify_labels = {"仓储方", "物流方", "其他参会方"}
        trans = [make_transcription(100.0, 101.0, "x")]
        merged = diarizer.merge_transcription_with_roles(trans, [])
        assert merged[0]["role"] == "未知"
        assert merged[0]["role"] not in classify_labels


class TestExistingRoleAndOutputShape:
    def test_existing_role_is_overwritten(self, diarizer):
        trans = [make_transcription(0.0, 1.0, "x", role="物流方")]
        roles = [role_seg(0.5, "仓储方")]
        assert diarizer.merge_transcription_with_roles(trans, roles)[0]["role"] == "仓储方"

    def test_existing_role_overwritten_by_unknown_when_unmatched(self, diarizer):
        # 外部系统导入的转写自带 role，匹配不上也会被「未知」覆盖
        trans = [make_transcription(100.0, 101.0, "x", role="物流方")]
        assert diarizer.merge_transcription_with_roles(trans, [])[0]["role"] == "未知"

    def test_output_order_follows_transcription_order(self, diarizer):
        trans = [
            make_transcription(30.0, 31.0, "第三句"),
            make_transcription(0.0, 1.0, "第一句"),
            make_transcription(15.0, 16.0, "第二句"),
        ]
        roles = [role_seg(0.0, "仓储方"), role_seg(15.0, "物流方"), role_seg(30.0, "其他参会方")]
        merged = diarizer.merge_transcription_with_roles(trans, roles)
        assert [m["text"] for m in merged] == ["第三句", "第一句", "第二句"]
        assert [m["role"] for m in merged] == ["其他参会方", "仓储方", "物流方"]

    def test_output_keys_are_transcription_keys_plus_role(self, diarizer):
        trans = [make_transcription(0.0, 1.0, "x")]
        merged = diarizer.merge_transcription_with_roles(trans, [])
        assert list(merged[0].keys()) == ["start", "end", "text", "role"]

