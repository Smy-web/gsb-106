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

#### 测试结论（共 58 条用例，全部通过）

**diarize 的兜底路径（pyannote 不可用时的实际走向）**

- 标签规律：按段下标奇偶交替发 `SPEAKER_00` / `SPEAKER_01`，与真实说话人无关。
- `num_speakers` 是写死的常量 2：1 段发言报 2，0 段发言（空列表）也报 2。
- `segments` 传 `None`（或不传）抛 `TypeError`；段里缺 `start` 或 `end` 抛 `KeyError`——except 只兜加载/推理的异常，兜不住兜底路径自身的异常。
- **调用方无法从返回值分辨真分离与兜底**：两条路径返回值的键集合完全相同（`speaker_segments` / `num_speakers`），没有任何标记字段；唯一的间接迹象是兜底路径 `num_speakers` 恒为 2 且标签严格交替。
- 加载失败后 `pipeline` 仍是 `None`，**每次调用都会重试加载**，不是「一个实例只试一次」。

**diarize 的真分离路径（用标准库假 pipeline 钉住）**

- `num_speakers` 是不同标签的实际数量（1 个说话人报 1，3 个报 3），与兜底的「恒为 2」是两条路径最显著的口径差。
- 假 pipeline 抛异常时落到兜底路径（异常被吞，返回交替标签 + `num_speakers == 2`）。

**classify_speaker_roles 的口径**

- 时间窗只比 `start` 字段：`abs(spk.start - trans.start) < 1.0`，严格小于，正好 1.0 不算；转写段完全落在说话人段内部也不算命中。
- 多条转写命中同一说话人段时取第一条；同一说话人多段发言的文本会累加。
- 计票是「命中了多少个不同关键词」，不是出现次数；互为子串的关键词（货架/货架号、库存/库存管理、配送/配送中心、运输/运输路线）会让同一句话重复计票（一句「货架号」计 2 票）。
- 平票（含 0:0）、零命中、空文本都判「其他参会方」。
- `role_segments` 每项 = 原说话人段的键（保持原顺序）+ 末尾追加 `role`。
- 「取不到角色信息就填 `未知`」的默认分支**不可达**：`speaker_roles` 由同一份 `speaker_segments` 构建，每个 speaker 必有条目，classify 的输出里永远不会出现 `未知`（已用断言钉死）。

**merge_transcription_with_roles 的口径**

- 窗口：`abs(trans.start - role_seg.start) < 2.0`，严格小于，abs 对称（说话人段更早开始也算命中）。
- 窗口内多条候选取 `role_segments` 里的**第一条**，不是最近的一条（有专门用例区分这两种实现）。
- 匹配不上填 `未知`；转写段自带的 `role` 会被覆盖（包括被覆盖成 `未知`）。
- 输出顺序跟随输入转写段顺序；输出每项 = 转写段原有的键 + `role`。

**缺键与脏输入**

- `classify`：说话人段缺 `speaker` 必抛 `KeyError`；缺 `start` 时转写列表为空不抛、非空才抛；转写缺 `text` 时窗口命中才抛。
- `merge`：转写缺 `start` 时 `role_segments` 为空不抛、非空才抛；`role_segment` 缺 `role` 时窗口命中才抛。
- 影响范围：一处缺键不是只坏一行，整个调用抛异常，整场会拿不到任何结果。

#### 第 6 条的取舍

选「**测试按现状钉死两个标签的实际值，同时在测试与清单中明确标注这是应当统一的待修缺陷**」。理由：下游按 `role` 字符串分组，`其他参会方` 与 `未知` 会被拆成两组，这是已经造成实际后果（行动计划负责方一栏全空）的口径不一致；只当现状钉死会把缺陷洗成「特性」。对应测试 `test_label_vocabularies_diverge_between_classify_and_merge` 断言当前两值不等，并在注释中写明应当统一——缺陷修复后该测试需要随新口径更新。

#### 负例验证（第 8 条，源码已还原）

- 改动：`classify_speaker_roles` 平票分支 `else: role = "其他参会方"` 临时改为 `role = "仓储方"`。
- 结果：10 条测试变红——`test_window_compares_start_fields_not_overlap`、`test_text_accumulates_across_segments_of_same_speaker`、`test_tie_yields_other_participant`、`test_zero_keyword_hits_yields_other_participant`、`test_empty_transcription_text_yields_other_participant`、`test_no_matching_transcription_yields_other_participant`、`test_default_unknown_role_branch_is_unreachable`、`test_label_vocabularies_diverge_between_classify_and_merge`、`test_classify_missing_start_ok_when_transcription_empty`、`test_classify_transcription_missing_text_ok_when_window_misses`。
- 验证后已用 `git checkout -- backend/` 还原，`git status` 确认 `backend/` 与 baseline 逐字节一致，58 条用例恢复全绿。

#### 既有缺陷清单

1. **兜底路径按段下标奇偶伪造说话人标签，且无任何降级标记**（对应线上症状一「物流方的话被标成仓储方」）。
   - 触发输入：pyannote 不可用时（线上常态）的任何 `diarize` 调用。
   - 实际行为：标签按段序号奇偶交替发放，与真实说话人无关；`num_speakers` 恒为 2；返回值无任何字段表明走了兜底。相邻两段真实发言若分属不同方，会被奇偶规则随意归并：物流方的运输/车辆发言一旦被并入以仓储关键词为主的 `SPEAKER_xx`，分类与合并环节就会把这段话贴成「仓储方」，邮件也就发给了仓储方。
   - 业务期望：分离不可用时应显式报错或在返回值中标记降级，而不是静默伪造分离结果。
   - 影响：纪要责任方错挂、邮件发错对象，且运维无法从输出或日志分辨走的是哪条路。

2. **classify 的时间窗只比 `start`、阈值 1 秒、命中即取第一条**（对应线上症状二「整场所有人都成了其他参会方」）。
   - 触发输入：转写来自另一套系统、时间戳与说话人段存在超过 1 秒的系统性偏移（或只对齐了 `end`）。
   - 实际行为：所有说话人段都匹配不到转写，文本为空，0:0 平票，全部判「其他参会方」；行动计划负责方一栏随之全空。
   - 业务期望：按时间重叠匹配，或阈值可配置并对「全场零命中」给出告警。
   - 影响：整场会纪要责任方全部失效，纪要发出去等于没发。

3. **互为子串的关键词重复计票**。
   - 触发输入：发言含「货架号」「库存管理」「配送中心」「运输路线」等词。
   - 实际行为：一句「货架号得重排」同时命中「货架」与「货架号」，计 2 票；物流侧同理。计票被子串词放大，分类结果向含有子串词的一侧倾斜。
   - 业务期望：一个词在一次发言中只计一票（按命中最长词或先去重）。
   - 影响：角色判定被措辞细节扭曲，与症状一的错标叠加放大。

4. **两处「判不出来」的标签口径不一致**（第 6 条取舍对应项，待修）。
   - 触发输入：classify 零命中 → `其他参会方`；merge 窗口匹配不上 → `未知`；且 merge 会把转写段原有的 `role` 覆盖（包括覆盖成 `未知`）。
   - 实际行为：下游按 `role` 字符串分组时，同一种「未知」被拆成两个组；症状二中「负责方一栏全空」正是分组键不一致的直接后果。
   - 业务期望：统一为一个标签（或统一的空值约定）。
   - 影响：下游分组统计与邮件分组错乱。

5. **merge 窗口内取第一条候选而非最近的一条**。
   - 触发输入：多人交替发言、2 秒窗口内有多条 `role_segments`（测试中用布局优化会的形状复现：10.0 秒的转写，候选 11.9 秒「物流方」在前、10.2 秒「仓储方」在后）。
   - 实际行为：取列表中第一条（差 1.9 秒的），而不是时间上最近的（差 0.2 秒的）。
   - 业务期望：取时间差最小的候选。
   - 影响：交替发言密集时责任方贴错，与症状一同类。

6. **diarize 失败后每次调用都重试加载，且重试不可见**。
   - 触发输入：pyannote 持续不可用。
   - 实际行为：`pipeline` 保持 `None`，每次 `diarize` 都重新 import 并失败一次，再静默走兜底；日志里看不到「已降级」。
   - 业务期望：加载失败应记录日志或缓存失败状态，并让降级可见。
   - 影响：运维手上「谁也说不清跑的是哪条路」的直接成因。

7. **兜底路径对 0 段、1 段发言也报 `num_speakers == 2`**。
   - 触发输入：空转写或单段转写。
   - 实际行为：无人/单人参会也报两个说话人。
   - 业务期望：`num_speakers` 应反映实际发出的不同标签数。
   - 影响：下游任何按说话人数做的校验/展示都被误导。

#### 验证

- `~/venvs/gsb-warehouse/bin/python -m pytest tests/ -q` → 58 passed。
- 测试只用内存合成数据与标准库替身；`tmp_path` 仅用于占位空音频文件；未联网、未安装任何包、未设置 `HUGGINGFACE_TOKEN`。

