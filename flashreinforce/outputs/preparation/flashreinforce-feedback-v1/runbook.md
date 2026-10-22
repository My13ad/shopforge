# 难例采样配置

本目录提供均匀采样与难例混采的对照配置。两组从同一个 adapter 开始并新建优化器，使用同一训练池、验证设置和异步调度器。该配置提供实验工具，未报告训练收益。

## 采样与预算

| 配置 | control | hard20 |
|---|---:|---:|
| 953 题背景池抽样 | 100% | 80% |
| 192 题难例通道 | 0% | 20% |
| 最大更新数 | 256 | 256 |
| 学习率 | 5e-7 | 5e-7 |
| warmup | 1 步 | 1 步 |
| 批次 / 在途上限 / 完成队列 | 8 / 32 / 64 | 8 / 32 / 64 |
| 验证间隔 / 最少更新 / patience | 64 / 128 / 3 | 64 / 128 / 3 |
| 候选批次 / 连续跳过上限 | 1024 / 16 | 1024 / 16 |

192 道难例也属于背景池，因此 20% 是额外采样通道的比例。没有占用限制时，难例成员的理论派发占比约为 control 20.15%、hard20 36.12%。实际训练贡献还受完成时间、整批跳过和 KL 门控影响，应分别统计派发、消费和更新。

## 难例清单

`pool-learnable/sampling-manifest.json` 固定训练任务和难例 ID。筛选条件为至少 4 次有效采样、至少 2 次失败、失败率不低于 35%，且至少严格成功过一次。符合条件的 228 题按收缩失败率排序后取 192 题。

清单只包含任务 ID 和数据指纹。无效奖励不计作难例失败证据；持续失败且没有成功记录的任务保持背景采样频率。`pool/` 保存另一份筛选记录，实际命令使用 `pool-learnable/`。

`confirmation/` 提供额外的 200 题评测清单，排除已知训练、开发、校准及 Final200 任务，同时排除相同商品 ASIN 和规范化需求文本。使用该清单前，应先根据 validation50 固定检查点。

## 命令预览

在 `flashreinforce/` 中执行，填写基础模型、父 adapter 和新输出目录：

```bash
PYTHONPATH=src python outputs/preparation/flashreinforce-feedback-v1/commands.py   --stage control --model "$BASE" --adapter "$PARENT_ADAPTER" --output-root "$NEW_RUN"
```

`commands.py` 打印 JSON 参数数组，默认包含 `--dry-run`。支持 reference-parent、pilot、reference-pilot-step1、control、hard20、confirm-parent、confirm-control、confirm-hard20。

运行对照前，先独立评测父模型，执行 1 步 pilot，再导出第 1 步模型做独立评测。`check_gpu_acceptance.py` 比较驻留验证与独立加载结果，并检查采样版本及权重。通过检查后，两组均从原父模型开始。训练结束后，用 `scripts/summarize_feedback_sampling.py` 汇总实际采样比例，从 `checkpoint_selection/best-checkpoint.json` 读取各组模型。

CPU 测试覆盖采样占用、批次有效性、验证屏障、早停、导出和配置解析。运行需要至少 40 个环境槽位，以及容纳模型、推理缓存和完整检查点的资源。
