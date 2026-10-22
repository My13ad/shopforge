# ShopSimulator 购物环境

本目录提供商品页面、搜索索引、Reward v3 和 `/api/shop_agent` 工具服务，供 ShopForge 的 FlashREINFORCE 阶段使用。环境版本为 v2.1，上游仓库与提交记录在 [EMBEDDED_SOURCE.json](EMBEDDED_SOURCE.json)。

先按 [环境准备](../../../docs/setup.md) 创建根目录的 `.venvs/shopsim`，安装 `shop_env/requirements.txt`。以下命令从项目根目录执行：

```bash
.venvs/shopsim/bin/python tools/prepare_environment.py --stage flashreinforce --build-index
PATH="$PWD/.venvs/shopsim/bin:$PATH" SHOPSIM_PORT=5721 SHOPSIM_ENV_SLOTS=40 bash flashreinforce/environments/ShopSimulator/shop_env/start.sh
```

准备工具校验压缩包并生成商品 JSON 和 SQLite 索引。服务运行期间保持终端打开，训练配置中的 URL 使用 `http://127.0.0.1:5721`。

商品解压文件、生成索引和运行日志由本地准备或运行产生。Flash 副本还包含规格文本方括号的动作解析修复。
