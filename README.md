# 新加坡 Major 第一阶段 Pick’Em

## 任意十格查询与全体填法对标

下载[离线查询页](https://github.com/Canyun259/cs2-singapore-major-pickem/releases/download/2026-11-18/pickem-lookup.html)，用新版浏览器打开。每换一队，页面都会显示至少命中五格的模拟通过率、全局并列排名、严格低于当前方案的百分位，以及更高、同分和更低的填法数。[完整 CSV 压缩包](https://github.com/Canyun259/cs2-singapore-major-pickem/releases/download/2026-11-18/all-tickets-csv.zip)含 120 张分表，按两支 3–0 队伍分组。

在同一批 500,000 次完整赛事模拟中，逐一评估了全部 10,090,080 种合法填法。样本第一名的 3–0 为 PVISION 和 Liquid，普通晋级为 NIP、HEROIC、FaZe、Eternal Fire、100 Thieves、M80，0–3 为 GOSH 和 Chinggis Warriors。主样本通过率 31.5218%，另用 200,000 次独立模拟复核为 31.3955%。全部填法的中位通过率为 7.2668%。排名和百分位均以主样本为准，样本第一名与相近方案的次序可能随采样改变。

- [预测与历史回测报告](output/pdf/2026新加坡Major第一阶段-PickEm预测与回测报告.pdf)
- [双架方案报告](output/pdf/2026新加坡Major第一阶段-PickEm双架方案.pdf)
- [遍历结果摘要及独立验证](all-tickets/summary.json)

如需从头重做全量查询数据，在仓库根目录运行 `node hedge.cjs` 和 `node all-tickets/sample.cjs`，然后进入 `all-tickets`，用支持 OpenMP 的 C++17 编译器运行 `g++ -O3 -std=c++17 -fopenmp enumerate.cpp -o enumerate`。执行 `./enumerate stage.bin counts.bin` 后，用已安装 NumPy 的 Python 运行 `python export.py` 生成离线查询页和全部 CSV。

## 预测模型

输入冻结于 2026-11-18。Node.js 24 可直接运行，不需要安装依赖。解压后在此目录运行 `node simulate.cjs` 重算预测，运行 `node backtest.cjs` 重做滚动回测。预测会写入 singapore-result.json，回测会写入 backtest.json。原始经验模型可用 `node legacy.cjs` 复现。

## 实验设计

哥本哈根和上海用于首次训练。奥斯汀、布达佩斯和科隆分别作为下一届留出比赛，每次只用之前已结束的届次拟合。排名冻结在该届首轮记录，近况、地图和直接交手使用开赛日前 90 天的可用比赛，近况最多 20 场。队名沿用比赛档案，历史窗口可能包含阵容调整前的比赛。单图胜负作为训练标签，留出预测时整届只使用赛前队伍快照；测试阶段的实际选图和后续真实对手不会输入十格模拟。三个测试阶段都按真实胜负重放并检查了对阵。首轮依据当时的赛程和种子。

比较了共同特征的原权重基线、统一缩放单图对数胜算的版本，以及分别拟合排名、近期战绩、交手、地图和选手状态的六参数版本。采用带先验惩罚的逻辑回归。缩放模型先验中心为 1，标准差为 0.5，六参数先验见 backtest.cjs。三届留出集也用于比较方案，尚无额外赛事独立验证选模结果。对数损失和 Brier 分数越低越好。

历史选手数据以赛前最近两场每场 ADR 最高两名队员的均值估计，并向中性值收缩。仅纳入能取到本队全场统计且有五行有效记录的场次；缺失场次不参与计算，全缺失队次取中性值。历史输入无法一致恢复上次的三个月焦点选手 Rating，基线省略该项。六参数版本使用 ADR，最终采用的缩放版本将选手项置零。历史 BP 用可获得的分图出场次数估计选禁偏好，完整比赛级历史地图覆盖有限。新加坡预测使用原始快照里的实际选禁频率。选禁行为参数没有重新拟合。

训练样本为 5 届第一阶段的 165 场、254 张实战地图。每次预测中，各对阵、各赛制抽样 500 次选禁。历史作业用 12,000 次模拟搜索及另 30,000 次检验；新加坡用 60,000 次搜索及另 100,000 次检验。作业搜索采用多起点两两换位，以同一模拟中至少猜中五格为目标。输出按 3–0、3–1/3–2、0–3 排列。随机种子固定。模拟概率只反映指定模型。

## 数据与复现

- events.json 保存八组历史首轮对阵及双方 HLTV、VRS 排名，各条附 HLTV 原始页。
- history.json 保存可直接回测的赛前快照和赛果。历史比赛来源为 [Aokua/csgopred 的 SQLite 快照](https://github.com/Aokua/csgopred)，其数据快照按 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布。此处提取、校验并汇总了所需的记录。原比赛数据来自 HLTV 等来源。
- map-corrections.json 保存八场按 HLTV 原始页补齐的分图胜负。原始库的未打地图、比分不一致记录不计入地图特征。
- players.json 按队列出最近两场选手 ADR 和 HLTV 比赛编号，可在 https://www.hltv.org/matches/ 后加编号定位。data.json 包含新加坡每队三个月分图胜率、选禁图频率及样本量，地图顺序是 Cache、Dust2、Mirage、Inferno、Nuke、Ancient、Anubis。
- 如需重建 history.json，从上述仓库下载 data/csgopred.db 后运行 `python prepare.py 数据库路径`。一般复现只需要压缩包里的 JSON 快照。

