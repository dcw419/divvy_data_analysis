
# 🚲 Divvy 共享单车商业策略与数据挖掘分析引擎

**—— 基于时空特征与单体经济模型 (UE) 的极寒淡季精细化运营研究**

## 📌 项目简介 (Executive Summary)

本项目以芝加哥 Divvy 共享单车 2026 年 1 月（极寒淡季）的真实运营数据为研究对象，构建了一套从**底层数据清洗、时空特征诊断、财务 UE 测算到统计学假设检验**的全链路商业分析引擎。
项目通过严谨的数据归因分析，量化了“结构性流失”与“摩擦性损耗”，并以**后端命令行 (CLI)** 的形式提供了高扩展性的策略诊断工具。

## 🎯 核心业务突破点 (Business Highlights)

1. **重构资产评价体系**：引入单体经济模型 (Unit Economics)，证明了电单车（E-bike）虽周转率极高但深陷高换电成本的“低毛利陷阱”，打破了“高频即高利”的业务幻觉。
2. **潮汐网络流量模型**：基于图论节点守恒定律计算 `Net Flow`，精准定位早高峰住宅区“断供”与 CBD“淤积”的供需非对称性，提出“红包车反向调度”降本策略。
3. **统计学实证归因**：引入计量经济学最小二乘法，通过 t 检验与 p 值，对气温、促销等变量进行显著性检验，为动态定价（10分钟暖冬一口价）提供数理支撑。

---
## 数据获取
```bash
aws s3 sync s3://divvy-tripdata "下载位置" --no-sign-request --exclude "*" --include "2022*" --include "2023*" --include "2024*" --include "2025*" --include "202601*"
```
## 📂 核心代码结构与函数说明 (Core Functions)

本项目采用模块化设计，分为控制层 (`main.py`) 与 业务逻辑层 (`analysis_ops.py`)。

### 1. 业务逻辑层 `analysis_ops.py`

包含了所有核心的分析与可视化绘图函数（所有图表均实现了 Data-Ink Ratio 最大化，内置极值自动标注与商业结论文本框）：

* **`analyze_hourly_bimodal(df, output_dir)`**
* **功能**: 绘制 24 小时工作日与周末的通勤双峰平滑曲线。
* **洞察**: 自动捕获早晚高峰极值点并标注，证明极寒天气下休闲场景消亡，业务退化为“通勤短途刚需”。


* **`analyze_winter_strategy(df, output_dir)`**
* **功能**: 绘制散客群体骑行时长分布直方图。
* **洞察**: 标定 10 分钟阈值红线，量化极短途场景占比（>65%），推导“10分钟一口价”策略。


* **`analyze_asset_efficiency(df, output_dir, target_year, target_month)`**
* **功能**: 绘制双坐标轴图（总单量柱状图 vs. 平均时长折线图）。
* **洞察**: 揭示不同车型的周转率异质性，电单车在淡季承担了超过 65% 的运力吞吐。


* **`analyze_station_intelligence_strategy(df, output_dir, target_year, target_month)`**
* **功能**: 计算全城站点的 `Net Flow` (Inflow - Outflow)，绘制散点诊断矩阵。
* **洞察**: 自动高亮识别“严重淤积 (Net Flow > +20)”与“严重缺车”的 Top 3 异常站点，为线下卡车提供精确调度坐标。


* **`analyze_unit_economics_and_margin(df, output_dir, target_year, target_month)`**
* **功能**: 核心财务引擎。根据真实阶梯定价与折旧/换电成本，计算不同车型的 ARPU、单均成本与毛利率。
* **洞察**: 绘制财务堆叠柱状图，直观剥离营业收入中的“水分（成本）”，呈现真实毛利。



### 2. 控制层 `main.py`

基于 `argparse` 构建的后端 CLI 入口，支持选择性调用特定维度的分析，极大地提升了工程扩展性。

---

## ⚙️ 安装与运行指南 (Installation & Usage)

### 环境依赖

```bash
pip install pandas numpy matplotlib seaborn statsmodels

```

### 快速启动 (CLI Commands)

打开终端 (Terminal/CMD)，进入项目根目录。

**1. 运行全局标准分析（默认 2026年1月，生成所有图表）**

```bash
python main.py

```

**2. 按时间切片，选择性运行特定模块（例如：仅运行单体经济 UE 测算）**

```bash
python main.py --year 2026 --month 1 --task ue

```

**3. 执行 OLS 统计回归检验（需要你在 main.py 中添加该 task）**

```bash
python main.py --task regression

```

**4. 查看后端 API 帮助文档**

```bash
python main.py --help

```

---

## 📈 统计学与数学模型支持 (Methodology)

* **Net Flow 节点质量守恒定律**:


* **Unit Economics (单体经济模型)**:


* **多元线性回归与假设检验**:

## 可视化结果与策略制定
![fig1](/figures/cluster_analysis.png "聚类方法得到不同类别的用户画像")
*图1 聚类方法得到不同地区的用户画像，绿色为休闲区，红色为核心通勤点，蓝色为均衡生活区*

    P.S.具体可视化的成果均在文件夹`conclusion/analysis.md`中展示
---


