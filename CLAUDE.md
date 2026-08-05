# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

A股连板天梯图可视化项目，从东方财富获取股票数据，分析涨停连板情况，并生成前端可视化页面。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 获取股票日线数据
python main.py fetch

# 分析涨停数据
python main.py analyze

# 生成连板数据JS文件 (前端可视化用)
python main.py generate-ladder

# 生成K线数据JS文件
python main.py generate-kline

# 执行完整流程 (fetch -> analyze -> generate-ladder)
python main.py full
```

## 数据流程架构

```
数据获取 (PyTDX/AkShare/Tencent) → 数据分析 (涨停识别/连板计算) → JS生成 → 前端展示
```

### 1. 数据获取层 (`depend/fetchers.py`)
- `PyTDXDataFetcher`: 使用 PyTDX 库从行情服务器获取数据（主用）
- `AkShareDataFetcher`: 使用 AkShare HTTP API（备用/降级）
- `TencentDataFetcher`: 使用腾讯行情 API (web.ifzq.gtimg.cn)，从现有数据文件读取股票列表
- `CompositeDataFetcher`: 复合获取器，优先 PyTDX，失败后依次降级 AkShare、Tencent
- `get_thread_api()`: 线程本地 PyTDX 连接管理

### 2. 数据验证与存储 (`depend/validation.py`, `depend/storage.py`)
- `DataValidator`: 数据完整性与合理性验证
- `DataStorage`: 按扩展名（parquet/csv/json）读写数据，支持备份

### 3. 数据分析层 (`main.py` `Analyzer` 类)
- `identify_limit_ups()`: 根据涨跌幅限制识别涨停股票
  - 主板/创业板 10%，ST 5%，科创板/创业板注册制 20%
- `calculate_consecutive_days()`: 计算连续涨停天数
- `identify_board_type()`: 识别板块类型（一字板/T字板/换手板）

### 4. 数据生成层
- `generate_ladder_data_for_html()` (`main.py`): 生成 `data/ladder_data.js`
- `generate_kline_data()` (`function/generate_kline_data.py`): 生成 `data/kline_data.js`

### 5. 前端 (`concept_ladder.html`)
- 加载 `data/ladder_data.js` 和 `data/kline_data.js`
- 使用 ECharts 显示 K 线图
- 纯静态 HTML，部署在 Vercel

## 关键文件路径

| 文件 | 用途 |
|------|------|
| `data/stock_daily_latest.parquet` | 原始股票日线数据 |
| `data/limit_up_ladder.parquet` | 连板分析结果 |
| `data/ladder_data.js` | 前端连板数据 |
| `data/kline_data.js` | 前端 K 线数据 |
| `function/stock_concepts.py` | 获取概念题材（东方财富 API） |
| `depend/fetchers.py` | 数据获取器（PyTDX/AkShare/Composite） |
| `depend/validation.py` | 数据验证 |
| `depend/storage.py` | 数据存储 |
| `depend/config.py` | 配置（服务器列表、并发数等） |

## 配置修改

修改 `depend/config.py`:
- `PYTDX_SERVERS`: PyTDX 行情服务器列表
- `MAX_WORKERS`: 数据获取并发数（默认 20）
- `CONCEPT_FETCH_WORKERS`: 概念获取并发数（默认 10）

## 部署

- **平台**: Vercel（静态托管）
- **配置**: `vercel.json` 将所有请求指向 `concept_ladder.html`
- **自动更新**: GitHub Actions 每日 UTC 1:00 (北京时间 9:00) 执行 `python main.py full`
