# 基于 LSTM 与 GRU 的唐诗字符级语言模型

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://github.com/Houkiang/tang-char-lm/actions/workflows/tests.yml/badge.svg)](https://github.com/Houkiang/tang-char-lm/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

作者：**Houkiang**

本项目使用 PyTorch 实现 LSTM 与 GRU 字符级自回归语言模型，在唐诗语料上比较字符交叉熵和困惑度，并实现 greedy、temperature、top-k、top-p 以及五言/七言表层约束生成。项目包含固定数据协议、三随机种子实验、统计语言模型基线、可复现运行记录和 XeLaTeX 论文框架。

## 主要特点

- 按完整诗篇切分 train/validation/test，避免同一首诗跨集合泄漏。
- 词表仅由训练集建立，显式报告验证集和测试集 OOV。
- LSTM 与 GRU 共用数据、网络宽度、训练预算和早停规则。
- 困惑度先累加全测试集 NLL 与有效 token 数，再统一取指数。
- 使用固定协议执行 seed 42、43、44 三次实验，报告均值和样本标准差。
- 统计 unigram/bigram 基线只在验证集选择平滑系数。
- 生成实验包含固定提示、固定随机种子、重复率分析与训练集完全匹配检查。
- 五言和七言约束解码保证四句字数与标点结构，并明确其文学质量边界。

## 实验结果

数据集共 164 首诗，按诗篇划分为 131/16/17；训练词表包含 2,315 个 token，测试集 OOV 率为 9.17%。

三随机种子结果如下，数值为均值 ± 样本标准差：

| 模型 | 参数量 | 测试交叉熵 | 测试困惑度 |
| --- | ---: | ---: | ---: |
| LSTM | 1,812,875 | 6.5850 ± 0.0615 | **725.05 ± 44.12** |
| GRU | 1,582,475 | 6.6420 ± 0.0410 | 767.03 ± 31.17 |

统计基线：unigram PPL 为 767.04，bigram PPL 为 1187.86。LSTM 在当前配置下平均 PPL 较低，但只在 3 个种子中的 2 个优于 GRU，且参数量更多，因此不作统计显著性或纯单元结构因果结论。完整数据见[多种子结果](docs/multiseed_results.md)和[实验协议](docs/experiment_protocol.md)。

生成实验中，greedy 解码出现明显循环；top-p 与约束解码显著降低重复率。LSTM 与 GRU 的五言、七言约束格式成功率均为 3/3，36 个固定样本均未与训练诗篇完全相同。约束只保证字数和标点，不保证平仄、押韵、语义连贯或文学质量。

## 项目结构

```text
.
├── configs/                 # 冻结的主实验协议
├── data/
│   ├── raw/                 # 原始语料，不提交到 Git
│   └── processed/           # 划分与词表，不提交到 Git
├── docs/                    # 数据审计、开发记录、实验结果和部署文档
├── report/                  # XeLaTeX 学术报告
├── results/                 # 可公开的轻量实验结果快照
├── runs/                    # 本地运行目录，整体不提交 Git
├── scripts/                 # 多种子实验入口
├── src/tang_char_lm/        # 数据、模型、训练、评估与生成实现
├── tests/                   # unittest 测试
├── project.py               # 无需安装包即可调用的源码入口
└── pyproject.toml           # Python 包配置
```

## 环境要求

- Python 3.11（项目实测版本 3.11.9）
- PyTorch 2.11.0
- NumPy 2.4.6
- NVIDIA GPU 可选；RTX 50 系需要支持 Blackwell 的 CUDA 构建
- XeLaTeX 可选，仅用于编译论文

提供三份依赖文件：

- `requirements.txt`：通用 PyTorch 2.11 环境
- `requirements-cu128.txt`：CUDA 12.8 固定环境
- `requirements-cpu.txt`：CPU 固定环境

完整的 Windows、Linux、GPU、CPU 和离线安装说明见[部署文档](docs/deployment.md)。

## 快速开始

### 1. 克隆并创建环境

```powershell
git clone https://github.com/Houkiang/tang-char-lm.git
cd tang-char-lm
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

RTX 50 系、CUDA 12.8：

```powershell
python -m pip install -r requirements-cu128.txt
python -m pip install -e . --no-deps
```

仅使用 CPU：

```powershell
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
```

### 2. 获取和审计数据

```powershell
python project.py data fetch
python project.py data stats
python project.py data prepare
python project.py data audit
```

数据来源为 [`nndl/llm-beginner/poetryFromTang.txt`](https://github.com/nndl/llm-beginner/blob/master/poetryFromTang.txt)。原始数据不随本仓库重新分发，下载后应得到 SHA-256：

```text
2798806f731be51e94323ba05326e4f59caecb8cb19a4403706f4b22d2c549db
```

如网络无法直接访问 GitHub，可手动下载文件并保存为 `data/raw/poetryFromTang.txt`。

### 3. 验证环境

```powershell
python project.py doctor
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

GPU 环境应显示 `cuda_available: true`。

## 训练

先运行小样本过拟合检查：

```powershell
python project.py overfit-check --model both --device cuda
```

单个模型：

```powershell
python project.py train --model lstm --run-dir runs/lstm-seed42 --device cuda --seed 42
python project.py train --model gru  --run-dir runs/gru-seed42  --device cuda --seed 42
```

中断后使用完全相同的命令并添加 `--resume`。程序会拒绝用不一致配置覆盖已有 run。

完整三种子实验：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_multiseed.ps1 -Device cuda
```

## 评估与生成

```powershell
python project.py baseline --output runs/baselines-seed42.json
python project.py evaluate --run-dir runs/lstm-seed42 --device cpu
python project.py generate-suite --run-dir runs/lstm-seed42 --device cpu
```

单条五言约束生成：

```powershell
python project.py generate `
  --run-dir runs/lstm-seed42 `
  --prompt 春 `
  --method top_p `
  --seed 99 `
  --line-length 5
```

汇总三种子结果：

```powershell
python project.py aggregate `
  --protocol configs/experiment_protocol_v1.json `
  --output runs/multiseed_summary.json `
  --run-dir runs/lstm-seed42 --run-dir runs/lstm-seed43 --run-dir runs/lstm-seed44 `
  --run-dir runs/gru-seed42  --run-dir runs/gru-seed43  --run-dir runs/gru-seed44
```

## 论文

报告使用 Abstract、Introduction、Related Work、Method、Experiment、Result 和 Conclusion 结构：

```powershell
xelatex -output-directory=report report/main.tex
xelatex -output-directory=report report/main.tex
```

## 可复现性说明

- 主实验协议存放在 `configs/experiment_protocol_v1.json`，包含数据、词表和划分哈希。
- `runs/` 是完全忽略的本地运行目录；`results/` 保存无 checkpoint、无本机路径的公开结果快照。
- `evaluate`、`baseline` 和 `generate-suite` 在结果存在时采用幂等读取，避免反复查看测试集后调参。
- seed 42 的测试结果早于多种子协议冻结，这一点已在协议中公开记录。
- 当前主比较保持隐藏维度一致，但 LSTM 与 GRU 参数量不同。

## 测试

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

测试覆盖数据划分与防泄漏、词表、批处理、PPL 聚合、LSTM/GRU 输出、训练恢复、统计基线、一次性测试评估和约束生成。

更新公开结果快照：

```powershell
python project.py export-results --run-root runs --output results
```

## License

代码以 [MIT License](LICENSE) 发布。唐诗数据集来自上游项目，不包含在本仓库许可证授权范围内。
