# FlashREINFORCE

本目录使用购物环境的终局奖励训练 Agent，支持同步批次和跨参数更新的异步采样。默认流水线从 OPD-32 的 LoRA 权重开始，每题采样 1 条轨迹，每批使用 8 个任务。

在项目根目录预览默认异步训练命令：

```bash
python tools/preview.py flashreinforce --config configs/paths.local.json
```

公共入口为 `scripts/train_grpo.py --method flashreinforce`，异步模式使用 `--flash-execution async`。直接调用该入口时默认同步；根目录预览会显式选择异步、验证选点和 256 次更新预算。

使用本目录的 `.venv` 和 `PYTHONPATH=src`，并连接 Flash 环境服务。预览带有 `--dry-run`；检查配置后，去掉该参数启动训练。

| 模块 | 作用 |
|---|---|
| `flashreinforce.py` | 奖励中心化、重要性比率与序列 KL 门控 |
| `async_fit.py` / `async_sampling.py` / `async_runtime.py` | 跨更新队列、权重同步与部分轨迹续写 |
| `flash_batching.py` | 原生奖励有效性检查及整批跳过 |
| `checkpoint_selection.py` | 验证集选点与早停 |
| `feedback_sampling.py` | 按任务清单混采难例 |

这些模块位于 `src/shopping_grpo/training/grpo/`。训练后从 `checkpoint_selection/best-checkpoint.json` 读取选中的模型。损失与调度细节见 [算法](docs/flashreinforce.md) 和 [异步实现](docs/flashreinforce-async.md)。
