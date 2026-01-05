# A股连板分析工具项目

## 项目概述

这是一个用于分析A股市场连续涨停股票（连板股）的Python工具，包含数据获取、分析、可视化功能。项目主要功能包括：

- 从多个数据源获取A股股票日线数据
- 识别涨停股票并计算连续涨停天数
- 分析股票概念题材和行业信息
- 生成连板天梯图和晋级率统计
- 提供交互式HTML可视化界面

## 技术栈

- **语言**: Python 3.12+
- **数据处理**: pandas, numpy
- **数据获取**: pytdx, akshare
- **数据存储**: parquet格式
- **可视化**: ECharts (前端), HTML/CSS/JavaScript
- **HTTP请求**: requests

## 项目结构

```
lgbp2/
├── main.py                 # 主程序入口，包含数据获取、分析、生成等完整流程
├── requirements.txt        # 项目依赖
├── concept_ladder.html     # 连板天梯可视化前端界面
├── concept_ladder - 副本.html  # HTML文件备份
├── depend/                 # 依赖模块目录
│   ├── __init__.py         # 包初始化文件
│   ├── config.py           # 配置管理模块
│   ├── di_container.py     # 依赖注入容器
│   ├── interfaces.py       # 接口定义模块
│   ├── services.py         # 服务实现模块
│   ├── backup_manager.py   # 数据备份管理模块
│   └── monitoring.py       # 监控指标模块
├── data/                   # 数据存储目录
│   ├── kline_data.js       # K线数据JS文件
│   ├── ladder_data.js      # 连板数据JS文件
│   ├── limit_up_ladder.parquet  # 连板分析结果数据
│   └── stock_daily_latest.parquet  # 股票日线数据
└── function/               # 功能模块目录
    ├── stock_concepts.py   # 股票概念题材获取模块
    ├── update_html.py      # HTML文件更新工具
    ├── update_project.py   # 项目更新工具
    └── generate_kline_data.py  # K线数据生成模块
```

## 核心功能

### 1. 数据获取 (DataFetcher)
- 从PyTDX和AkShare获取A股股票日线数据
- 支持增量更新，避免重复获取已有数据
- 多线程并发获取，提高效率
- 自动处理连接池和故障转移

### 2. 数据分析 (Analyzer)
- 识别涨停股票（区分主板、创业板、ST股不同涨跌幅限制）
- 计算连续涨停天数
- 识别涨停板类型（一字板、T字板、换手板）
- 获取股票概念题材信息

### 3. 数据生成
- 生成连板天梯数据 (ladder_data.js)
- 生成K线数据 (kline_data.js)
- 计算每日晋级率统计

### 4. 可视化
- 交互式连板天梯图
- 按连板天数分组展示
- 按概念题材分组展示
- K线图弹窗查看功能
- 日期导航和过滤选项

## 使用方法

### 安装依赖
```bash
pip install -r requirements.txt
```

### 命令行使用
```bash
# 获取股票日线数据
python main.py fetch

# 分析数据并生成连板信息
python main.py analyze

# 生成连板数据JS文件
python main.py generate-ladder

# 生成K线数据JS文件
python main.py generate-kline

# 生成完整HTML可视化
python main.py visualize

# 执行完整流程
python main.py full

# 指定日期范围
python main.py fetch --start-date 20241201 --end-date 20241231
```

### 前端可视化
直接在浏览器中打开 `concept_ladder.html` 文件即可查看连板天梯图。

## 数据源

- **PyTDX**: 用于获取实时股票数据
- **AkShare**: 作为备用数据源
- **东方财富API**: 用于获取股票概念题材信息

## 特色功能

1. **智能涨停识别**: 自动区分不同板块的涨跌幅限制（主板10%，创业板/科创板20%，ST股5%）
2. **多维度分析**: 按连板天数、概念题材、涨停板类型等多维度展示
3. **交互式界面**: 支持日期选择、股票筛选、K线查看等功能
4. **增量更新**: 支持增量数据获取，避免重复下载
5. **多数据源**: 支持PyTDX和AkShare双数据源，提高数据获取稳定性

## 配置选项

- **默认日期**: 程序会根据当前时间自动判断默认结束日期（交易时间前使用前一日数据）
- **线程数**: 默认使用20个线程并发获取数据
- **数据存储**: 使用parquet格式存储，节省空间且读取快速

## 文件说明

- `stock_daily_latest.parquet`: 存储所有股票的日线数据
- `limit_up_ladder.parquet`: 存储连板分析结果
- `ladder_data.js`: 供前端使用的连板数据
- `kline_data.js`: 供前端使用的K线数据
- `promotion_rates.csv`: 晋级率统计

## 开发约定

- 代码使用中文注释，便于理解
- 日期格式统一使用YYYYMMDD格式
- 股票代码格式为"代码.交易所"（如000001.SZ）
- 日志使用Python logging模块记录执行过程