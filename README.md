# 应急物资储备库会议纪要系统 · 说话人角色判定子系统

## 这个仓库是什么

应急物资储备库开布局优化会，录音先转写成带时间戳的分段文本，再由说话人分离给出「哪一段时间是谁在说」，然后交给这里的模块：把两者对齐，并按发言内容判定每个说话人是仓储方还是物流方。纪要里每一句发言后面挂的责任方、行动计划里的负责方，都是这么来的。这里裁剪出来的就是「对齐与角色判定」这一段，转写、抽取、渲染、邮件都不在范围内。

## 目录

- `backend/config.py` — 配置项，pydantic-settings，从环境变量和 `.env` 读。
- `backend/models.py` — 上下游交换的数据形状（pydantic 模型），本模块不 import 它，留着是让你看清转写段的字段。
- `backend/core/speaker_diarizer.py` — 本次要处理的模块。

`backend` 和 `backend/core` 都没有 `__init__.py`，按命名空间包用。验证命令必须从仓库根目录、用 `-m` 的形式跑，这样仓库根才会进 `sys.path`。

## 运行环境

离线。`~/venvs/gsb-warehouse` 里只有 `requirements-task.txt` 列出来的包。**pyannote.audio、torch、torchaudio 都没有装，也不允许装、不允许联网下载模型**，没有任何音频文件可用。测试只能用内存里的合成数据和标准库替身（需要假 pipeline 就用 monkeypatch 往实例上塞）。

## 数据形状

转写段（`transcription_segments` / `segments`）：

```json
{"start": 0.0, "end": 4.2, "text": "A区货架号得重排"}
```

说话人段（`speaker_segments`）：

```json
{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00"}
```

`diarize(audio_path, segments)` 返回 `{"speaker_segments": [...], "num_speakers": int}`。

`classify_speaker_roles(speaker_segments, transcription_segments)` 返回：

```json
{
  "speaker_roles": {"SPEAKER_00": {"role": "仓储方", "storage_keyword_count": 2, "logistics_keyword_count": 0}},
  "role_segments": [{"start": 0.0, "end": 4.2, "speaker": "SPEAKER_00", "role": "仓储方"}]
}
```

`merge_transcription_with_roles(transcription_segments, role_segments)` 返回转写段的副本，每项多一个 `role` 键，顺序跟随输入的转写段顺序。

## 角色口径

判定结果只有三个标签：`仓储方`、`物流方`、`其他参会方`。关键词表在实现里，仓储一侧是货架、库存、盘点这一类，物流一侧是运输、配送、车辆、时效这一类。下游（纪要渲染与邮件）按 `role` 的字符串值分组，`merge_transcription_with_roles` 匹配不上时填的是另一个词，两者的关系需要你用测试确认清楚。

## 已定口径与待确认口径

上面几节是系统已经定死的口径。其余行为（时间窗口按哪个字段比、阈值多少、多条候选取哪一条、关键词怎么计票、平票判什么、说话人数怎么统计、缺字段会怎样、说话人分离不可用时走哪条路）在实现里都有具体做法，但没有任何文档和测试。这次的任务就是把它们用测试钉住，并把你认为不对的地方列成缺陷清单。

### 本次做题人填写

**验证结果**：`~/venvs/gsb-warehouse/bin/python -m pytest tests/ -q` → **59 passed**（用例总数 59，≤60）。全部使用内存合成数据与标准库替身，`tmp_path` 仅用于占位空音频文件，未联网、未安装任何包、未设置 `HUGGINGFACE_TOKEN`。

**测试文件与各自钉住的口径**

- `tests/conftest.py` — 夹具与标准库替身（`FakePipeline`/`ExplodingPipeline`），占位音频路径。
- `tests/test_diarize_fallback.py` — 兜底路径：标签按下标奇偶交替发 `SPEAKER_00/01`；`num_speakers` 恒为 2（1 段、0 段也报 2）；`segments=None` 抛 `TypeError`；段缺 `start`/`end` 抛 `KeyError`；返回值无任何「走了兜底」的标记。
- `tests/test_diarize_pipeline.py` — 真分离路径（假 pipeline）：`num_speakers` 按实际去重人数（1 人报 1、0 段报 0），与兜底恒 2 的口径差已钉死；pipeline 抛异常时静默落回兜底；加载失败后**每次** `diarize` 调用都重试加载（不是一个实例只试一次）；pipeline 就位后不再重复加载。
- `tests/test_classify_roles.py` — 角色判定：窗口比的是 `start` 字段、阈值 1.0 秒、严格小于（正好 1.0 不算）；窗口内多条转写取列表第一条；同一说话人多段发言文本累加；计票按「命中不同关键词数」而非出现次数；互为子串的关键词重复计票（「货架号」计 2 票、「运输路线」计 3 票，两个计数字段已钉进断言）；平票、零命中、空文本均判「其他参会方」；`role_segments` 键集合与顺序为输入键 + 末尾 `role`。
- `tests/test_merge_roles.py` — 贴角色：窗口比 `start`、阈值 2.0 秒、对称（转写更早开始也算命中）、严格小于；窗口内多条候选取**列表第一条**而非时间最近一条（有专门用例区分两种实现，复现布局优化会事故形状）；匹配不上填「未知」；转写段自带 `role` 会被覆盖；输出顺序跟随转写段顺序；输出键为转写段键 + `role`。
- `tests/test_missing_keys.py` — 缺键与脏输入：哪些缺键抛 `KeyError`、哪些在列表为空时不抛（非空才抛）的差异用例；一处缺键导致**整场调用**抛异常，不是只坏一行。

**第 2 条结论：调用方无法从返回值分辨「真分离」与「兜底」**。两条路返回完全同构的 `{"speaker_segments", "num_speakers"}`，没有任何标记字段；兜底 `num_speakers` 恒为 2，但真分离也可能恰好是 2，不构成可靠判别。这就是运维日志里看不出走了哪条路的根因（缺陷 D5）。

**第 4 条默认分支结论：不可达**。`classify_speaker_roles` 里 `speaker_roles.get(spk, {"role": "未知"})` 的默认值永远用不上——`speaker_roles` 由同一批 `speaker_segments` 构建，每个 speaker 必有条目。已由 `test_role_segments_never_produce_unknown_label` 用断言+注释钉死：classify 产出的 role 只会是三个固定标签，「未知」只可能来自 merge 的匹配失败分支。

**第 6 条的选择：把「两处口径不一致」当现状钉死，同时列为待修缺陷（D6）**。理由：本次任务是钉死现状而非改代码，测试锁死现状（classify 三标签 vs merge 兜底「未知」）能防止行为无意漂移；缺陷清单指明修复方向是统一标签体系，修复时同步更新 `test_label_vocabularies_differ_pinned_as_current_behavior` 即可，结论从测试里即可读出。

**负例验证（第 8 条）**：临时把 `backend/core/speaker_diarizer.py` 平票分支的 `role = "其他参会方"` 改为 `role = "仓储方"`，8 条测试变红：`test_window_boundary_exactly_one_second_is_outside`、`test_substring_double_count_can_flip_role`、`test_tie_goes_to_other`、`test_zero_hits_goes_to_other`、`test_empty_text_goes_to_other`、`test_no_matching_transcription_goes_to_other`、`test_role_segments_never_produce_unknown_label`、`test_transcription_missing_text_ok_when_window_misses`。验完已用 `git checkout` 还原，`sha256sum` 校验 `backend/` 三个文件与 baseline 逐字节一致，复跑 59 passed。

**既有缺陷清单**

- **D1 merge 候选取「列表第一条」而非「时间最近的一条」**。触发输入：一条转写段同时落在多个角色段的 2 秒窗口内（如转写 start=10.0，仓储方段 start=11.8 在列表靠前，物流方段 start=10.1 才是最近的说话人）。实际行为：贴上列表靠前的「仓储方」。业务期望：贴时间上最近的说话人段的角色。影响：**对应线上症状「物流方的话被标成仓储方」**——物流方讲运输/车辆的发言被挂到仓储方名下，纪要责任方与邮件收件人分组全部错位。
- **D2 classify 时间窗口只比 `start` 且阈值仅 1.0 秒，对不上外部系统导入的转写**。触发输入：转写来自另一套系统，时间戳整体偏移或基准不同，所有转写段与说话人段的 `start` 差都 ≥1.0 秒。实际行为：每个说话人累加到的文本都是空串，计票 0:0，全员判「其他参会方」。业务期望：窗口匹配失败应可识别、可告警，或用更稳健的对齐方式。影响：**对应线上症状「整场所有人都成了其他参会方」**的主要原因。
- **D3 零命中与真平票不加区分，都判「其他参会方」**。触发输入：一个关键词都没命中（包括 D2 的文本为空场景）与两侧票数真正相等的场景。实际行为：两种语义完全不同的情形输出同一个标签，且无任何标记。业务期望：「没拿到可判定的文本」应与「证据打平」区分开。影响：与 D2 合谋放大了「整场其他参会方」——行动计划负责方全空，纪要发出去等于没发，且事后无法从输出分辨是数据问题还是真实判定。
- **D4 兜底路径 `num_speakers` 恒为 2**。触发输入：兜底路径下 0 段或 1 段发言。实际行为：零段（没有任何说话人段）也报 2 个说话人。业务期望：与真分离口径一致，按实际去重人数（0 段应报 0）。影响：下游按说话人数做的任何统计/校验在兜底路径下全是假数据。
- **D5 diarize 兜底完全静默，且加载失败每次调用都重试**。触发输入：pyannote 不可用的任何环境（即线上现状）。实际行为：异常被吞，返回值与真分离同构、无标记、无日志；`pipeline` 保持 `None`，每次调用都重新尝试 import。业务期望：走了兜底应有明确标记/日志，加载失败应缓存结论或按策略重试。影响：**对应运维「日志里看不出走的是真分离还是兜底」**；且线上可能长期在跑兜底结果而无人知晓。
- **D6 角色标签词汇不一致**。触发输入：任何 merge 匹配不上的转写段。实际行为：classify 只产出「仓储方/物流方/其他参会方」，merge 匹配不上填第四个词「未知」。业务期望：同一套标签体系。影响：下游按 `role` 字符串分组，同一场会的责任方被劈成两套词汇，「未知」组静默漏进邮件分组。
- **D7 互为子串的关键词重复计票**。触发输入：一句「A区货架号得重排」（「货架」「货架号」各计 1 票共 2 票）；一句「运输路线要优化」（「运输」「路线」「运输路线」共 3 票）。实际行为：一个概念算多票，可翻转角色判定。业务期望：一个概念一票，或词表去掉子串重叠。影响：计票虚高的一侧被放大，角色判定被词表结构而非发言内容左右。
- **D8 merge 无条件覆盖转写段自带的 `role`**。触发输入：从另一套系统导入、已带 `role` 的转写段。实际行为：匹配上被覆盖成新角色，匹配不上被覆盖成「未知」。业务期望：已有角色应保留或至少冲突时告警。影响：外部系统已有的责任方信息被静默冲掉，与 D2 叠加时整场会的既有标注全部丢失。
- **D9 一处缺键整场失败**。触发输入：任一说话人段缺 `speaker`、任一转写段缺 `start`（非空列表场景）等。实际行为：`KeyError` 直接抛出，其余完好段的结果一行都拿不到。业务期望：单行脏数据不应让整场会拿不到纪要。影响：一条脏数据即整场纪要生成失败，且无部分结果可降级使用。
