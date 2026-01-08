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
数据获取 (PyTDX/AkShare) → 数据分析 (涨停识别/连板计算) → JS生成 → 前端展示
```

### 1. 数据获取层 (`depend/services.py`)
- `PyTDXDataFetcher`: 使用 PyTDX 库从行情服务器获取数据（主用）
- `AkShareDataFetcher`: 使用 AkShare HTTP API（备用/降级）
- `CompositeDataFetcher`: 复合获取器，优先 PyTDX，失败后自动切换 AkShare

### 2. 数据分析层 (`main.py` `Analyzer` 类)
- `identify_limit_ups()`: 根据涨跌幅限制识别涨停股票
  - 主板/创业板 10%，ST 5%，科创板/创业板注册制 20%
- `calculate_consecutive_days()`: 计算连续涨停天数
- `identify_board_type()`: 识别板块类型（一字板/T字板/换手板）

### 3. 数据生成层 (`main.py`)
- `generate_ladder_data_for_html()`: 生成 `data/ladder_data.js`
- `generate_kline_data()`: 生成 `data/kline_data.js`

### 4. 前端 (`concept_ladder.html`)
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
| `depend/services.py` | 服务实现（数据获取/验证/存储） |
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
