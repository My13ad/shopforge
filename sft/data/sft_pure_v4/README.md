# DeepSeek-v4 购物轨迹池

本目录包含 1,192 条按任务去重的教师轨迹，全部以 `gold_purchase` 结束。它是可选数据池；默认 SFT 流水线使用 `data/sft/` 中的 800 条训练轨迹。

| 文件 | 内容 |
|---|---|
| `all.jsonl` | 去重后的完整轨迹 |
| `difficulty_labels.jsonl` | 任务难度与轨迹复杂度标签 |
| `duplicate_report.json` | 重复任务的来源、质量特征及保留记录 |
| `metadata.json` | 数量、哈希、标签来源和数据混合统计 |

难度标签来自 DeepSeek-v4-flash：简单 284 条、中等 798 条、困难 110 条，比例约为 23.8% / 66.9% / 9.2%。课程数据准备脚本位于 [prepare_sft_curriculum.py](../../scripts/prepare_sft_curriculum.py)。如需改变比例，可依据这些标签生成自己的划分，并检查训练与评测任务隔离。
