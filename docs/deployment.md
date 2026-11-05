# 环境安装与部署

本文档说明如何从全新环境运行数据处理、训练、评估、文本生成和论文编译。项目是本地研究程序，不需要数据库或常驻服务。

## 1. 前置条件

- 64 位 Windows 10/11 或现代 Linux。
- Python 3.11，建议使用 3.11.9。
- 至少 8 GB 内存和 8 GB 可用磁盘空间。
- GPU 可选。CPU 能运行全部功能，但训练较慢。
- NVIDIA GPU 环境只需正确的显卡驱动；PyTorch wheel 自带 CUDA 运行库，无需单独安装 CUDA Toolkit 或 cuDNN。
- 编译报告需要 TeX Live 与 XeLaTeX。

本项目实测环境：Python 3.11.9、PyTorch 2.11.0+cu128、NumPy 2.4.6、RTX 5060 Laptop GPU 8 GB。

## 2. 获取代码

```powershell
git clone https://github.com/Houkiang/tang-char-lm.git
cd tang-char-lm
```

Linux：

```bash
git clone https://github.com/Houkiang/tang-char-lm.git
cd tang-char-lm
```

## 3. 创建隔离环境

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4. 安装依赖

### CUDA 12.8

适用于支持 CUDA 12.8 的 NVIDIA 驱动和 RTX 50 系 GPU：

```powershell
python -m pip install -r requirements-cu128.txt
python -m pip install -e . --no-deps
```

如果官方 PyTorch 下载站速度较慢，可从阿里云镜像目录手动下载与 Python 匹配的 wheel。例如 Python 3.11 64 位 Windows 对应：

```text
torch-2.11.0+cu128-cp311-cp311-win_amd64.whl
```

安装本地 wheel 与其余依赖：

```powershell
python -m pip install "<wheel-path>" -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install numpy==2.4.6 -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install -e . --no-deps
```

### CPU

```powershell
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
```

### 其他 CUDA 版本

前往 PyTorch 官方安装选择器，选择操作系统、Pip 和与驱动匹配的 CUDA 构建。安装 PyTorch 后再执行：

```powershell
python -m pip install numpy==2.4.6
python -m pip install -e . --no-deps
```

## 5. 环境验收

```powershell
python project.py doctor
python -m pip check
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
```

CUDA 快速检查：

```powershell
python -c "import torch; x=torch.randn(32,32,device='cuda'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0),(x@x).sum().item())"
```

## 6. 数据部署

```powershell
python project.py data fetch
python project.py data stats
python project.py data prepare
python project.py data audit
```

如果自动下载失败，手动下载上游 `poetryFromTang.txt` 并放到：

```text
data/raw/poetryFromTang.txt
```

执行 `data stats` 后确认 SHA-256 为：

```text
2798806f731be51e94323ba05326e4f59caecb8cb19a4403706f4b22d2c549db
```

## 7. 运行实验

单模型训练：

```powershell
python project.py train --model lstm --run-dir runs/lstm-seed42 --device cuda --seed 42
```

全部三种子实验：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_multiseed.ps1 -Device cuda
```

CPU 部署时将 `-Device cuda` 改为 `-Device cpu`。脚本会恢复已有训练或创建新训练，并为最佳 checkpoint 生成测试指标，最后写入 `runs/multiseed_summary.json`。

## 8. 结果文件

| 文件 | 含义 | 是否提交 Git |
| --- | --- | --- |
| `config.json` | 完整运行配置 | 导出到 `results/` |
| `metrics.json` | 每轮训练与验证指标 | 导出到 `results/` |
| `test_metrics.json` | 固定测试结果与哈希 | 导出到 `results/` |
| `generation.json` | 固定生成样本 | seed 42 导出到 `results/` |
| `best.pt` | 最佳模型与优化器状态 | 否 |
| `last.pt` | 最后模型与恢复状态 | 否 |

checkpoint 可通过相同配置重新训练获得。

将本地轻量结果导出为适合公开提交的快照：

```powershell
python project.py export-results --run-root runs --output results
```

`runs/` 整体被 Git 忽略，避免公开快照阻塞新用户从零训练；`results/` 不包含 checkpoint、优化器状态或本机环境信息。

## 9. 编译论文

安装 TeX Live 后重新打开终端，检查：

```powershell
xelatex --version
```

编译两次以更新引用：

```powershell
xelatex -output-directory=report report/main.tex
xelatex -output-directory=report report/main.tex
```

生成的 PDF 和 LaTeX 临时文件默认不提交 Git。

## 10. 常见问题

### `torch.cuda.is_available()` 为 false

确认安装的是 CUDA wheel 而非 CPU wheel，并运行 `nvidia-smi` 检查驱动。PyTorch wheel 的 CUDA 版本不要求本机安装同版本 CUDA Toolkit，但要求显卡驱动足够新。

### Windows 无法执行激活脚本

可以不激活环境，直接调用：

```powershell
.\.venv\Scripts\python.exe project.py doctor
```

### 测试结果文件已经存在

这是预期行为。评估命令会读取已保存结果，避免无意中多次查看测试集。若开展新协议，应使用新的 run 目录。

### RTX 5060 训练结束出现原生退出码

当前实测环境在长 CUDA RNN 训练完成并写入 checkpoint 后，Windows 卸载阶段可能返回 `0xC0000409`。项目使用 CPU 重载最佳 checkpoint 复算验证集。其他机器应优先使用较新的稳定 NVIDIA 驱动；如问题持续，可用 CPU 训练或在 Linux 环境运行。
