# 环境准备

训练使用 Linux 和 NVIDIA GPU。SFT/OPD、教师服务、Flash 与购物环境分别安装依赖，避免 Torch、vLLM 和 NumPy 版本相互覆盖。完整安装命令见 [快速开始](../README.md#快速开始)。

## Python 与依赖

| 环境 | 用途 | 配置依据 |
|---|---|---|
| SFT/OPD 客户端 | 学生采样、SFT 与蒸馏更新 | Torch 2.8.0+cu128、PEFT 0.21.2、Accelerate 1.15.0、Liger 0.8.2 |
| 教师服务 | FP8 模型及 token 条件概率 | 独立 vLLM 环境，按教师接口校验 |
| Flash | veRL 训练及异步 vLLM 采样 | Torch 2.11.0、veRL 0.8.0、vLLM 0.25.1、Ray 2.56.1 |
| ShopSimulator | 商品页面、工具接口与 SQLite 搜索 | 环境目录中的 requirements.txt |

SFT/OPD 的精确依赖见 [sft-client.txt](../requirements/sft-client.txt)，Flash 见 [pyproject.toml](../flashreinforce/pyproject.toml) 的 `grpo` 依赖组。两者固定同一个 Transformers 提交。根目录准备工具要求 Python 3.11 或更高版本；README 使用 Python 3.12 训练、Python 3.11 运行环境服务。

Flash 的 veRL 补丁工具检查安装路径，因此虚拟环境应建在 `flashreinforce/.venv`，`python.flash` 应指向其中的 Python。SFT 与 Flash 都包含 `shopping_grpo` 包，运行时使用各自的工作目录和 `PYTHONPATH`，不要混用。`tools/preview.py` 会给出这些设置。

## 教师服务

将 Qwen3.8-27B-FP8 下载到本地目录，在路径配置中设置 `models.teacher` 和 `python.teacher`。OPD 会分阶段启动教师，读取学生实际采样 token 的条件对数概率。

`opd/assets/calibration/` 提供固定教师提示和模型配置检查信息。部署时需核对聊天模板、token ID 对齐及有限 logprob。教师服务尚未提供完整的依赖锁文件，应先通过这组接口检查再运行 OPD。

## 商品数据与服务

两个环境副本各含约 20 MB 的商品压缩包。准备工具会核对哈希、解压约 140 MB 的商品 JSON，并建立搜索索引。使用已安装环境依赖的 Python，在项目根目录执行：

```bash
.venvs/shopsim/bin/python tools/prepare_environment.py --stage sft --build-index
.venvs/shopsim/bin/python tools/prepare_environment.py --stage flashreinforce --build-index
```

在两个独立终端启动服务：

```bash
PATH="$PWD/.venvs/shopsim/bin:$PATH" SHOPSIM_PORT=5700 bash sft/environments/ShopSimulator/shop_env/start.sh
```

```bash
PATH="$PWD/.venvs/shopsim/bin:$PATH" SHOPSIM_PORT=5721 SHOPSIM_ENV_SLOTS=40 bash flashreinforce/environments/ShopSimulator/shop_env/start.sh
```

Flash 默认有 32 个在途采样任务，40 个环境槽位为验证预留了空间。路径配置中的服务 URL 应与启动端口一致。

## veRL 补丁

安装 Flash 依赖后，在根目录执行：

```bash
flashreinforce/.venv/bin/python flashreinforce/scripts/apply_verl_dynamic_sampling_patch.py
```

工具检查 veRL 版本与源码哈希后应用固定补丁。训练入口还会运行 `check_grpo_runtime.py`，核对环境、依赖和异步接口。`flashreinforce/outputs/preparation/v1/runtime-wheel/source/verl` 是 CPU 测试使用的源码快照；实际训练使用安装到虚拟环境中的 veRL。

## CPU 检查

根目录检查不加载模型：

```bash
python -X utf8 tools/verify_local.py
python -X utf8 -m pytest tests -q
```

各阶段测试在对应目录运行：

| 目录 | 命令 |
|---|---|
| `sft/` | `PYTHONPATH=src python -m pytest tests -q` |
| `opd/runtime/` | `python -m pytest test_objectives.py -q` |
| `flashreinforce/` | `PYTHONPATH=src:tests python -m pytest tests -q` |

测试依赖包括 pytest、Torch、NumPy、safetensors、PyArrow、Hydra。Windows 的 `PYTHONPATH` 使用分号分隔，读取中文元数据时启用 `-X utf8`。CPU 检查验证数学实现、数据边界和运行入口；GPU 加载及显存需求由实际模型运行验证。
