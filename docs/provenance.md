# 代码与数据来源

ShopForge 将 SFT、OPD 和 FlashREINFORCE 按独立运行环境组织。模型通过 LoRA adapter 衔接，任务清单和文件指纹用于确认运行输入。

## SFT 与 OPD

`sft/` 包含轨迹处理、分段加权损失和模型评测。默认训练数据为 800 条训练轨迹与 200 条开发轨迹。

`opd/runtime/` 提供可配置的 OPD-32 入口：基模、教师、两套 Python、环境 URL、父权重和参考评测均由参数指定。256 道任务采用固定顺序，并重新检查训练、开发、评测和校准任务之间的隔离。

`provenance/verified-opd32/` 保存用于对照的 OPD 源码。根目录测试逐 AST 比较 `importance_loss`、教师评分提取、任务选择及 `rollout/score/train/analyse` 核心函数，确保路径适配没有改变优化目标。

## FlashREINFORCE

Flash 实现基于 veRL 0.8.0，算法参考 Molt 提交 `725598ee595b1eb9fe09ec14e6956dd164dbf08c`。保留的上游文件位于 `flashreinforce/references/flashreinforce-upstream/labs-molt-main/`，用于损失和梯度对照测试。

项目还保留一份 veRL 源码快照，供 CPU 测试构造训练循环和 Hydra 配置。运行时补丁检查安装版本与预期源码哈希；更新依赖时需重新检查兼容性。算法实现、单卡设置与上游差异见 [方法说明](../flashreinforce/docs/flashreinforce.md)。

## ShopSimulator

两个环境副本的 [EMBEDDED_SOURCE.json](../sft/environments/ShopSimulator/EMBEDDED_SOURCE.json) 记录上游仓库和提交。压缩商品数据随项目提供；解压文件与搜索索引通过准备工具生成。

Flash 环境使用 `shopping-action-parser-v1`，修正规格文本包含方括号时的动作解析。补丁不改变 Reward v3 的分数或价格规则。原始 engine 保存在 `provenance/flash-native-engine.py`，两份运行环境的指纹由本地检查核对。

## 模型与许可

基础模型和教师模型由使用者单独下载，路径写入本地配置。模型、第三方源码和商品数据各自遵循其来源的条款；项目原创代码的 Apache-2.0 许可及第三方声明见 [许可说明](../THIRD_PARTY_NOTICES.md)。
