"""classify_speaker_roles 的角色判定口径。

实现口径（本文件逐条钉死）：
- 时间窗口：abs(说话人段.start - 转写段.start) < 1.0，只比 start，严格小于；
- 一个说话人段附近有多条转写时取列表中第一条命中的；
- 同一说话人多段发言的文本会累加；
- 计票按「命中了多少个不同关键词」，不按出现次数；
- 互为子串的关键词会重复计票；
- 平票与零命中都判「其他参会方」。
"""
from conftest import make_speaker_segment, make_transcription


def classify(diarizer, speaker_segments, transcription_segments):
    return diarizer.classify_speaker_roles(speaker_segments, transcription_segments)


class TestTimeWindow:
    def test_window_compares_start_fields_only(self, diarizer):
        # end 差得再远也不影响，只看 start
        spk = [make_speaker_segment(0.0, 0.5, "SPEAKER_00")]
        trans = [make_transcription(0.9, 999.0, "货架要重排")]
        result = classify(diarizer, spk, trans)
        assert result["speaker_roles"]["SPEAKER_00"]["role"] == "仓储方"

    def test_window_boundary_just_inside(self, diarizer):
        spk = [make_speaker_segment(0.0, 1.0, "SPEAKER_00")]
        trans = [make_transcription(0.999, 2.0, "货架要重排")]
        result = classify(diarizer, spk, trans)
        assert result["speaker_roles"]["SPEAKER_00"]["role"] == "仓储方"

    def test_window_boundary_exactly_one_second_is_outside(self, diarizer):
        # 阈值 1.0 是严格小于：正好差 1.0 秒不算在窗口内
        spk = [make_speaker_segment(0.0, 1.0, "SPEAKER_00")]
        trans = [make_transcription(1.0, 2.0, "货架要重排")]
        result = classify(diarizer, spk, trans)
        assert result["speaker_roles"]["SPEAKER_00"]["role"] == "其他参会方"

class TestCandidateSelection:
    def test_first_matching_transcription_wins(self, diarizer):
        # 窗口内多条转写时取列表中第一条，其余被 break 丢弃
        spk = [make_speaker_segment(0.0, 3.0, "SPEAKER_00")]
        trans = [
            make_transcription(0.2, 1.0, "货架"),
            make_transcription(0.4, 1.5, "运输 车辆 配送"),
        ]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 1
        assert roles["SPEAKER_00"]["logistics_keyword_count"] == 0
        assert roles["SPEAKER_00"]["role"] == "仓储方"

    def test_text_accumulates_across_segments_of_same_speaker(self, diarizer):
        spk = [
            make_speaker_segment(0.0, 1.0, "SPEAKER_00"),
            make_speaker_segment(10.0, 11.0, "SPEAKER_00"),
        ]
        trans = [
            make_transcription(0.2, 1.0, "货架"),
            make_transcription(10.2, 11.0, "库存"),
        ]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 2


class TestKeywordCounting:
    def test_counts_distinct_keywords_not_occurrences(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "库存 库存 库存 库存")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        # 「库存」出现 4 次，只计 1 票
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 1

    def test_substring_keywords_double_counted_storage(self, diarizer):
        # 「货架」与「货架号」同在词表且互为子串，一句话同时命中两票
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "A区货架号得重排")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 2
        assert roles["SPEAKER_00"]["logistics_keyword_count"] == 0

    def test_substring_keywords_triple_counted_logistics(self, diarizer):
        # 「运输」「路线」「运输路线」三者互为子串且同在词表，
        # 一句「运输路线」同时命中三票
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "运输路线要优化")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["logistics_keyword_count"] == 3
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 0

    def test_substring_double_count_can_flip_role(self, diarizer):
        # 一句话「货架号」（实际 1 个概念 2 票）压过两个不同的物流词
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "货架号旁边的车辆和时效")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 2
        assert roles["SPEAKER_00"]["logistics_keyword_count"] == 2
        # 2:2 平票
        assert roles["SPEAKER_00"]["role"] == "其他参会方"


class TestRoleDecision:
    def test_storage_majority(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "货架和库存，车辆")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["role"] == "仓储方"

    def test_logistics_majority(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "运输和配送，货架")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["role"] == "物流方"

    def test_tie_goes_to_other(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "货架和车辆")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["storage_keyword_count"] == 1
        assert roles["SPEAKER_00"]["logistics_keyword_count"] == 1
        assert roles["SPEAKER_00"]["role"] == "其他参会方"

    def test_zero_hits_goes_to_other(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "今天主要同步一下进度")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["role"] == "其他参会方"

    def test_empty_text_goes_to_other(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(0.1, 5.0, "")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["role"] == "其他参会方"

    def test_no_matching_transcription_goes_to_other(self, diarizer):
        # 转写来自另一套系统、时间戳对不上（窗口外）时，文本为空，0:0 判「其他参会方」
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        trans = [make_transcription(100.0, 105.0, "货架 库存 盘点")]
        roles = classify(diarizer, spk, trans)["speaker_roles"]
        assert roles["SPEAKER_00"]["role"] == "其他参会方"


class TestOutputShape:
    def test_speaker_roles_entry_keys(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        result = classify(diarizer, spk, [])
        assert set(result.keys()) == {"speaker_roles", "role_segments"}
        assert set(result["speaker_roles"]["SPEAKER_00"].keys()) == {
            "role", "storage_keyword_count", "logistics_keyword_count",
        }

    def test_role_segments_keys_and_order(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00")]
        result = classify(diarizer, spk, [])
        role_seg = result["role_segments"][0]
        assert list(role_seg.keys()) == ["start", "end", "speaker", "role"]

    def test_role_segments_preserve_extra_keys(self, diarizer):
        spk = [make_speaker_segment(0.0, 5.0, "SPEAKER_00", confidence=0.9)]
        result = classify(diarizer, spk, [])
        role_seg = result["role_segments"][0]
        assert role_seg["confidence"] == 0.9
        assert list(role_seg.keys()) == ["start", "end", "speaker", "confidence", "role"]

class TestUnknownDefaultBranchUnreachable:
    def test_role_segments_never_produce_unknown_label(self, diarizer):
        """钉死结论：classify 里 `speaker_roles.get(spk, {"role": "未知"})`
        的默认分支不可达。speaker_roles 由同一批 speaker_segments 构建，
        每个 speaker 必有条目，因此 role_segments 只会出现三个固定标签，
        永远不会出现「未知」。（「未知」只来自 merge 的匹配失败分支。）
        """
        spk = [
            make_speaker_segment(0.0, 1.0, "SPEAKER_00"),
            make_speaker_segment(50.0, 51.0, "SPEAKER_01"),
            make_speaker_segment(99.0, 100.0, "SPEAKER_02"),
        ]
        trans = [make_transcription(0.1, 1.0, "货架")]
        result = classify(diarizer, spk, trans)
        roles = [s["role"] for s in result["role_segments"]]
        assert "未知" not in roles
        assert set(roles) <= {"仓储方", "物流方", "其他参会方"}
        # 三个说话人：一个仓储方，两个零命中
        assert roles == ["仓储方", "其他参会方", "其他参会方"]
