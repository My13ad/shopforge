# 训练流水线

ShopForge 按 SFT、参考评测、OPD、FlashREINFORCE 的顺序运行。每个阶段使用独立的输出目录，下一阶段读取前一阶段产生的模型或评测结果。

先按 [环境准备](setup.md)安装依赖，并在 `configs/paths.local.json` 中填写模型、Python 和服务路径。在项目根目录预览命令：

```bash
python tools/preview.py sft --config configs/paths.local.json
python tools/preview.py sft-reference --config configs/paths.local.json
python tools/preview.py opd32 --config configs/paths.local.json
python tools/preview.py flashreinforce --config configs/paths.local.json
```

预览输出中的 `cwd` 是工作目录，`env` 是环境变量，`argv` 是命令参数数组。`unresolved_inputs` 列出缺少的输入。准备好输入后，在指定目录设置环境变量并执行命令；预览工具本身不启动训练。

## 1. 加权 SFT

SFT 从 Qwen3.5-2B-Base 初始化 LoRA，使用 800 条训练轨迹和 200 条开发轨迹。损失只覆盖模型回答与工具调用，说明、一般工具、规格选择、购买片段的权重分别为 `1 / 1.5 / 1.5 / 2`。

默认训练 3 个 epoch，输出到 `runs/sft-e1/adapter`。目录包含 LoRA 权重、分词器相关文件和聊天模板。`sft-e1` 是默认加权配方的目录名。

主入口为 `sft/scripts/train_lora_sft.py`。均匀监督、额外规格权重和课程学习工具也位于 `sft/scripts/`，可用于对照实验。

## 2. 为 SFT 模型生成参考评测

`sft-reference` 在 200 题开发集上评估新训练的 SFT 模型，输出到 `runs/sft-reference`。OPD 使用其中的 `run_config.json`、`sft-trajectories.jsonl`、`sft-summary.json`、`benchmark.jsonl` 和源码快照，确认输入模型与比较基线一致。

`opd/assets/reference-e1/` 保存一份基准评测供结果配对使用。每次训练新的 SFT 模型后，都应重新执行本阶段，再把生成的参考目录传给 OPD。

## 3. OPD-32

OPD 使用 Qwen3.8-27B-FP8 教师和固定的购物提示。每轮依次加载学生采样、教师评分、学生训练，控制显存占用。预览会传入模型路径、两套 Python、环境 URL、SFT 权重及参考评测。

训练任务由 `opd/assets/historical-opd32-task-ids.jsonl` 指定，共 256 题，排除开发集、Final200 和教师提示校准题。每批 8 道新任务，每题最多采集 2 个 assistant 回合，共做 32 次 AdamW 更新，学习率为 `5e-6`。

目标为即时 sampled reverse-KL importance-sampling surrogate，加 `0.1 ×` 加权 SFT 交叉熵，`discount=0`。成功和失败轨迹均参与训练。短程检查结束后，正式训练重新从指定 SFT 权重初始化。

默认输出为 `runs/opd32/round-32/training/adapter`。`--expected-parent-sha256` 必须与实际 SFT 权重及参考评测一致；预览工具会在权重文件存在时计算该值。

## 4. FlashREINFORCE

Flash 读取 OPD-32 的 LoRA 权重，在 953 道训练题上进行异步强化学习。默认每题 1 条轨迹、每批 8 题，最多更新 256 次。批次内任一轨迹缺少有效原生奖励时，整批跳过，更新计数不变。

固定的 50 题验证集每 64 次更新评测一次。至少完成 128 次更新后，连续 3 次验证未改善则早停；同分保留更早模型，初始模型也可成为最佳点。

输出位于 `runs/flash-from-opd32`，选中模型记录在 `checkpoint_selection/best-checkpoint.json`。最佳 adapter 可用于推理，恢复优化器还需相应的完整检查点。异步环境会话和在途队列不支持精确恢复。

Flash 预览带有 `--dry-run`。它用于检查配置；实际启动训练时去掉此参数。算法与调度分别见 [FlashREINFORCE](../flashreinforce/docs/flashreinforce.md) 和 [异步实现](../flashreinforce/docs/flashreinforce-async.md)。

## 数据与环境

SFT 开发集 200 题、RL 验证集 50 题和 Final200 各有用途，结果应在相同题集与推理设置下比较。训练数据按 task_id 与评测集隔离；严格成功要求完整终局、有效奖励和 `gold_purchase`。

SFT/OPD 环境使用端口 5700，Flash 使用 5721。两个副本均采用 Environment v2.1、Reward v3、observation v2、tool v2；Flash 包含规格方括号解析补丁。启动前确认命令中的 URL 对应正确的环境副本。
