# 本机环境状态

记录日期：2026-11-05（Asia/Shanghai）

| 项目 | 实测结果 |
| --- | --- |
| Python | 3.11.9（项目 `.venv`） |
| PyTorch | 2.11.0+cu128，来自本地 `torch-2.11.0+cu128-cp311-cp311-win_amd64.whl` |
| NumPy | 2.4.6 |
| CUDA 运行时 | 12.8（`torch.version.cuda`） |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，计算能力 12.0，8 GB |
| NVIDIA 驱动 | 577.05 |
| CUDA 可用 | `torch.cuda.is_available() == True` |
| GPU 计算 | `torch.randn`、矩阵乘法和 CPU/NumPy 往返通过 |
| 依赖检查 | `pip check`：无冲突 |
| 数据测试 | `unittest discover -s tests -v`：2 项通过 |
| XeLaTeX | 尚未安装 |

安装方式：用户已下载 PyTorch wheel；使用项目 `.venv` 的 pip 从本地 wheel 安装，依赖与 NumPy 经阿里云 PyPI 镜像安装。该环境适合进入 LSTM/GRU 模型实现和小样本过拟合检查。模型实验运行时仍应单独记录环境和随机种子。
