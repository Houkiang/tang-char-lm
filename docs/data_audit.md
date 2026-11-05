# 原始语料审计记录

记录日期：2026-11-05

来源：[`nndl/llm-beginner/poetryFromTang.txt`](https://github.com/nndl/llm-beginner/blob/master/poetryFromTang.txt)，通过已连接的 GitHub 文件接口读取原始文件字节，存于本地 `data/raw/poetryFromTang.txt`。本地文件与上游返回的 Git blob SHA-1 一致：`b0bcc7b97ae2b469af708e721acf363d7698a053`。

| 审计项 | 当前值 |
| --- | ---: |
| 原始文件大小 | 48,133 字节 |
| 原始文件 SHA-256 | `2798806f731be51e94323ba05326e4f59caecb8cb19a4403706f4b22d2c549db` |
| 按空行切分的诗篇数 | 164 |
| 完全重复诗篇数 | 0 |
| 诗文字符数（含诗内换行） | 16,298 |
| 不同字符数（含诗内换行） | 2,514 |

以上是基建阶段的原始数据审计值，不是训练/测试划分后的统计，也不是模型结果。源文件中存在疑似噪声 `屈rH猛`；当前只记录，不修改。后续数据管线完成后，应重新输出划分、词表及 OOV 审计值。

当前执行环境的直接 HTTP 下载被拒绝，因此 `data fetch` 在这里未能完成联网获取；现有本地文件已按 Git blob SHA 校验。其他环境可使用 `data fetch`，也可手动放入原始文件，再用 `data stats` 校验 SHA-256。

## 固定划分与词表

执行 `project.py data prepare`，种子 42；先按整篇 SHA-256 去重，再随机划分。产物为 `data/processed/poetryFromTang-seed42/` 下的 `manifest.json`、`splits.json` 和 `vocab.json`。三份划分的诗篇哈希没有交叉。

| 集合 | 篇数 | 字符数（含诗内换行） | 目标 token 数（含结束标记） | 未登录字符 | OOV 率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 131 | 13,166 | 13,297 | 0 | 0% |
| Validation | 16 | 1,681 | 1,697 | 108 | 6.42% |
| Test | 17 | 1,451 | 1,468 | 133 | 9.17% |

训练词表共 2,315 个 token，包含 `<pad>`、`<bos>`、`<eos>`、`<unk>` 四个特殊符号。验证/测试未见字符映射为 `<unk>`；该比例较高，后续解释困惑度时必须同时报告 OOV 率。`splits.json` SHA-256 为 `fe21435dac6d84e59c69eb30f36afbf324d5dba1307db683896644c3ad6b7d76`，`vocab.json` SHA-256 为 `e0ea6b0dce357b1b7b22bca82d3261e6ab9028d19cc272fde44510177d732883`。
