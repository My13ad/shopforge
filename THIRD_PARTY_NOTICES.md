# 第三方代码与数据

ShopForge 的原创代码采用根目录的 [Apache-2.0](LICENSE) 许可。下列组件保留原作者声明与适用条款；项目许可不替代第三方代码、数据或模型的许可。

| 组件 | 项目中的位置 | 来源与许可信息 |
|---|---|---|
| Molt / FlashREINFORCE 参考源码 | `flashreinforce/references/flashreinforce-upstream/labs-molt-main/` | NVIDIA-NeMo/labs-molt，固定提交 `725598ee595b1eb9fe09ec14e6956dd164dbf08c`；保留 Apache-2.0、NOTICE 和上游第三方声明 |
| veRL 测试源码快照 | `flashreinforce/outputs/preparation/v1/runtime-wheel/source/` | veRL 0.8.0，Apache-2.0；文件内保留各贡献者版权声明 |
| ShopSimulator 环境与商品数据 | `sft/environments/ShopSimulator/`、`flashreinforce/environments/ShopSimulator/` | ShopAgent-Team/ShopSimulator；版本见各目录的 EMBEDDED_SOURCE.json |
| Qwen 模型 | 使用者提供的本地模型目录 | 模型权重单独下载，按对应模型的许可使用 |

ShopSimulator 的内嵌快照未附独立许可证文件，上游仓库也未提供可识别的根目录许可。这里保留来源说明，不将环境及商品数据重新声明为 Apache-2.0。

训练数据中的商品描述及教师生成轨迹保留其来源属性。模型、依赖库和数据的使用范围应按各自条款判断。

Molt 原始声明见 [NOTICE](flashreinforce/references/flashreinforce-upstream/labs-molt-main/NOTICE) 与 [THIRD_PARTY_NOTICES.md](flashreinforce/references/flashreinforce-upstream/labs-molt-main/THIRD_PARTY_NOTICES.md)。veRL 快照的许可证位于 [LICENSE](flashreinforce/outputs/preparation/v1/runtime-wheel/source/LICENSE)。
