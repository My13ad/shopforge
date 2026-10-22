# 异步采样与训练

`--flash-execution async` 让轨迹生成跨越参数更新：后台持续完成购物任务，训练线程从完成队列中取出完整批次。损失与奖励规则见 [FlashREINFORCE 算法](flashreinforce.md)。

## 数据流

```text
训练任务 → 在途任务池 → 完成队列 → B 条候选轨迹 → 原生奖励检查 → 参数更新
              ↑             │                                   │
              └─ 完成后补位 ┘       保留未完成轨迹 ← 暂停、同步权重、续写
```

先完成的轨迹先进入 FIFO 队列。长回合不会阻塞已完成的其他任务。训练期间采样可以继续，已完成但未消费的轨迹和在途任务跨更新保留，每条轨迹最多消费一次。队列满时暂停补位，任务流按 epoch 排空。

每个生成片段在请求 vLLM 前记录已发布的策略版本，同时保存实际采样 token 和行为 log probability。一条轨迹可以包含多个版本，训练端按 token 对齐版本、概率和策略 mask。元数据缺失、错位或包含未来版本时停止训练。

更新结束后，调度器阻止新请求进入，中断并排空旧请求，再同步权重并恢复生成。未完成的 assistant 回合把已有 token 接到 prompt 后继续，保留行为概率、工具状态和环境会话，并扣减剩余生成预算。同步失败时保持暂停并退出。

## 单卡配置

| 参数 | 默认值 | 含义 |
|---|---:|---|
| batch | 8 | 每次更新消费的完整轨迹数 |
| `async_inflight` | 32 | 同时进行的任务上限 |
| `async_queue_batches` | 8 | 完成队列最多保存的批次数 |
| rollout 显存比例 | 0.25 | 推理引擎的显存预算比例 |
| 训练微批 | 1 | 每个训练微批的轨迹数 |

训练与推理引擎驻留同一 GPU。实际吞吐和显存取决于轨迹长度、排队情况及硬件；这些设置描述单卡实现，不代表多卡吞吐表现。

环境槽位至少覆盖在途任务，同时验证时还要为验证 worker 留出会话。默认配置可使用 40 个槽位。CPU 数据加载 worker 固定为 0，避免后台线程在 CUDA 初始化后派生子进程。

每批仍执行原生奖励检查，存在无效轨迹时整批跳过。报告采样分布时应区分派发、完成、消费、批次接纳和实际更新。

## 启动

在 `flashreinforce/` 中使用配置好的 Python，并设置 `PYTHONPATH=src`：

```bash
python scripts/train_grpo.py --method flashreinforce --flash-execution async   --profile smoke --model "$BASE" --adapter "$OPD32"   --env-url "$SHOPSIM_BASE_URL" --output "$NEW_OUTPUT" --dry-run
```

异步 `smoke` 使用 953 题任务流完成 4 次更新，便于检查跨更新队列；直接入口的 `main` 默认 64 次更新。根目录预览提供 256 步与早停配置。去掉 `--dry-run` 后会实际启动训练，输出使用新目录。

`shopping_flashreinforce.async_inflight`、`async_queue_batches` 和 `async_timeout_seconds` 控制并发、队列和等待时限。`async_runtime.py` 选择专用 manager/server，`async_fit.py` 核对 veRL 源码后安装进程内 hook。异步接口指纹位于 `patches/verl-0.8.0-flash-async-sources.json`。

## 检查点与诊断

检查点保存模型、优化器和批次记录，不序列化环境会话、在途请求或完成队列。重新启动训练时，可以加载 adapter；精确恢复此前所有在途轨迹尚不支持。上下文压缩关闭，以保持 token、概率与版本对齐。

| 指标 | 用途 |
|---|---|
| `cross_update_async` | 确认启用了跨更新调度 |
| `flash_async/completions_during_update` | 统计参数更新窗口内实际完成的采样 |
| `policy_lag_max` | 记录消费轨迹相对当前策略的最大版本滞后 |
| `mixed_version_trajectories` | 统计包含多个策略版本的轨迹 |
| `sampling_interruptions` | 统计生成片段的中断次数 |

完成时间重叠说明调度窗口重叠；判断中断续写需同时查看混合版本轨迹和中断记录。CPU 测试覆盖乱序完成、补位、反压、单次消费、异常传播、token 合并及版本屏障。实际 GPU 运行还应检查梯度、权重变化和推理端权重同步。
