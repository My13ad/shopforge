# FlashREINFORCE 算法

FlashREINFORCE 每个任务只采样一条轨迹，用环境终局奖励训练策略。ShopForge 在 veRL 中实现批次奖励中心化、token 重要性比率和序列 KL 门控，复用购物环境、工具协议及 LoRA 权重。

公共入口为 `scripts/train_grpo.py --method flashreinforce`。直接调用时默认同步，增加 `--flash-execution async` 可启用跨更新异步队列。主结果见 [项目 README](../../README.md#实验结果)。

## 优势与策略损失

每批包含 B 个不同任务，每题一条轨迹。先计算完整批次的优势 `A_i = R_i - mean(R)`，再拆成微批次。奖励低于批内均值时优势为负；这与任务是否严格成功是不同概念。

```text
p_it = exp(rollout_log_prob_it)
q_it = exp(learner_log_prob_it)
d_it = KL(Bernoulli(p_it) || Bernoulli(q_it))
D_i  = mean_policy_tokens(d_it)
m_i  = stopgrad(D_i <= delta)
rho_it = exp(clamp(learner_log_prob_it - stopgrad(rollout_log_prob_it), -30, 30))
loss = -(1/B) * sum_i m_i * A_i * mean_policy_tokens(rho_it)
```

KL 计算将概率裁剪到 `[1e-6, 1-1e-6]`。门控按整条轨迹决定，工具观察和 padding 不参与损失。被拒绝的轨迹仍计入原始批次分母 B，每条轨迹的 token 分母也保持不变。优势、行为概率和门控停止梯度。

实现采用直接 ratio 的数值保护，log ratio 超过 ±30 后该处策略梯度为零。有效策略 token 的 NaN/Inf 会触发错误。这里的 Bernoulli KL 比较采样 token 与其余词表的概率质量，衡量的是二元投影中的差异。

## 购物任务配置

| 设置 | GRPO 对照 | FlashREINFORCE |
|---|---|---|
| 初始化 | OPD-32 | OPD-32 |
| 训练池 | 953 题 | 953 题 |
| 每次更新轨迹 | 2 题 × 4 条 | 8 题 × 1 条 |
| 优势 | 同一任务组内中心化 | 整个批次中心化 |
| 策略目标 | PPO clipped surrogate | token importance sampling |
| 归一化 | token mean | 轨迹内平均，再按 B 平均 |
| 更新次数 | 由对照配置指定 | 每批 1 次 |
| 采样 | temperature 0.7 / top-p 0.9 | temperature 1 / top-p 1 / top-k -1 |
| 序列门控 | PPO 配置 | 平均 Bernoulli KL ≤ 1e-3 |

基础配方使用 LoRA r16/alpha32、学习率 `1e-6`、上下文 `24,576`、每轮最多 512 token、每题最多 35 步。采样器保存实际的 `processed_logprobs`，关闭截断采样与 repetition penalty，以匹配训练时使用的行为分布。

`--flash-batch-size` 同时调整批次和完整 mini batch。序列阈值可通过 `shopping_flashreinforce.sequence_kl_threshold` 修改。各方法的候选采样量和跳过量不同，比较计算成本时应同时记录生成轨迹、token、接纳率和耗时。

## 原生奖励与整批检查

训练使用 Reward v3 的原始分数，有效失败轨迹保留负奖励。`flash-whole-batch-native-v1` 在计算优势前检查整个候选批次：只要有一条环境无效、奖励不可核验或截断轨迹，整批跳过，然后继续读取任务流。

跳过不推进优化器、学习率调度、更新计数或验证间隔。每条 uid 最多消费一次；全批奖励相同仍执行一次更新，此时策略梯度为零，但优化器动量或 weight decay 仍可能改变参数。

入口默认采用 `--flash-invalid-batch-policy skip_batch`，也可选择 `error` 在遇到无效批次时停止。以下参数限制候选总数及连续跳过次数：

```text
shopping_flashreinforce.max_candidate_batches=256
shopping_flashreinforce.max_consecutive_skipped_batches=16
```

根目录流水线预览将候选总数上限设为 1024。每批写入 `flash_batch_admission` 事件，记录接纳决定、无效原因和累计数量。整批检查改变了采样分布，分析结果时应保留这些统计。

## veRL 与环境适配

`flashreinforce.py` 注册优势与损失，`flashreinforce_config.py` 提供兼容 veRL `PolicyLossConfig` 的类型配置。Flash 专用 hook 保留实际 `rollout_log_probs`，避免被原生 bypass 模式覆盖。预检拒绝与该方法不兼容的 G4、多次更新及采样设置。

固定的 veRL V4 补丁提供逐任务结果和更新诊断，`flash_batching.py` 核对源码哈希后在进程内安装训练循环 hook。升级依赖前需重新检查这些接口。

Flash 环境使用 `shopping-action-parser-v1`，修复 `action[argument]` 中规格文本带方括号的解析，例如 `[0-800度]`。补丁保留 Reward v3 的价格与评分规则，环境清单记录补丁前后的文件哈希。动作守卫最多允许三次连续拒绝，并把合法目标和最近拒绝记录写入轨迹。

## 运行方式

在 `flashreinforce/` 中使用该阶段 Python，设置实际模型和输出路径：

```bash
export PYTHONPATH="$PWD/src"
python scripts/train_grpo.py --method flashreinforce --profile probe   --model "$BASE" --adapter "$OPD32" --output "$PROBE" --dry-run
python scripts/train_grpo.py --method flashreinforce --profile smoke   --model "$BASE" --adapter "$OPD32" --output "$SMOKE" --dry-run
python scripts/train_grpo.py --method flashreinforce --profile main   --model "$BASE" --adapter "$OPD32" --output "$MAIN" --dry-run
```

同步 `probe` 使用 60 题做无梯度诊断；`smoke` 使用 8 条轨迹更新一次；`main` 默认更新 64 次，在 32/64 步保存与验证。输出目录使用新目录或空目录，去掉 `--dry-run` 后执行。根目录 `tools/preview.py flashreinforce` 提供的是另一组明确配置：异步、最多 256 次更新和验证选点。

原生续训按更新数推导数据位置，与整批跳过消耗的数据流不一致，因此该模式要求 `trainer.resume_mode=disable`。可以从指定 adapter 新建训练；异步队列的恢复限制见 [异步实现](flashreinforce-async.md)。

## 上游对应与测试

方法参考 *FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models*，源码对照固定为 Molt 提交 `725598ee595b1eb9fe09ec14e6956dd164dbf08c`。保留的 [loss.py](../references/flashreinforce-upstream/labs-molt-main/molt/models/loss.py) 用于损失与梯度比较。

Molt 使用 detached correction 与当前策略比率组合；本实现使用直接 ratio。保护区间内损失与梯度有对照测试，极端区间另有反例测试。ShopForge 采用单卡 B8 和 LoRA 配方，实验结果对应这一配置。

`tests/test_flashreinforce.py` 覆盖中心化、梯度方向、掩码、门控分母及数值保护；`tests/test_flash_batch_admission.py` 用真实 CPU AdamW 和固定 veRL 训练循环检查整批跳过、更新计数和保存间隔。运行方式见 [环境准备](../../docs/setup.md#cpu-检查)。
