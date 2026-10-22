# 监督微调

SFT 让 Qwen3.5-2B-Base 学习购物轨迹中的回答和工具调用。默认使用分段加权损失，增加规格选择和购买动作的权重。

在项目根目录生成完整命令：

```bash
python tools/preview.py sft --config configs/paths.local.json
```

实际入口为 `scripts/train_lora_sft.py --loss-mode smartad`。在本目录使用 SFT Python 环境并设置 `PYTHONPATH=src`。输入为 `data/sft/train.jsonl` 的 800 条训练轨迹和 `data/sft/validation.jsonl` 的 200 条开发轨迹。

训练完成后，在根目录执行 `sft-reference` 预览，再运行生成的评测命令。它使用 `recipes/round_task_eval.py` 生成新权重的开发集结果，供 OPD 比较与模型身份检查。

`data/sft_pure_v4/` 和课程学习脚本是可选数据与工具。默认流水线使用上述 800 条训练轨迹。安装和阶段衔接见 [环境准备](../docs/setup.md) 与 [训练流水线](../docs/pipeline.md)。
