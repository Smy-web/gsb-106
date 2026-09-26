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

（把测试结论与缺陷清单的摘要写在这里。）
