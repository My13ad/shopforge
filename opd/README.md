# 在线策略蒸馏

OPD 先让学生在购物环境中生成动作，再由 Qwen3.8-27B-FP8 教师为学生实际生成的 token 评分。学生采样、教师评分、学生训练分阶段加载。

在项目根目录生成 OPD-32 命令：

```bash
python tools/preview.py opd32 --config configs/paths.local.json
```

主入口为 `runtime/worker.py`，各阶段实现位于 `runtime/phases.py`。运行时使用同级 `sft/` 的源码和数据。默认使用固定的 256 道训练题，完成 32 次参数更新。

`assets/calibration/` 保存教师提示、50 道校准题及模型配置。`assets/reference-e1/` 保存基准评测；新的 SFT 权重应先生成自己的参考评测。OPD 会检查实际权重、参考结果和预期 SHA-256 是否一致。

`worker.py` 包含 GPU 监控，`--preparation-only` 仍需要可用的监控环境。仅做 CPU 数据准备时可使用 `runtime/phases.py prepare`，并按其帮助提供 tokenizer、模型元数据和其他参数。

安装与输入关系见 [训练流水线](../docs/pipeline.md)。优化目标的对照源码位于 `provenance/verified-opd32/`，相关测试在 `runtime/test_objectives.py`。
