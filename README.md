# A股连板分析工具 - Vercel部署版

这是一个用于分析A股市场连续涨停股票（连板股）的可视化工具，已配置为可在Vercel上部署的静态网站。

## 项目概述

本项目包含数据获取、分析和可视化功能，主要用于展示A股市场的连续涨停股票情况。

## 部署到Vercel

### 方法一：一键部署
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/your-username/your-repo-name)

### 方法二：手动部署

1. 将此仓库克隆到本地：
   ```bash
   git clone <your-repository-url>
   cd lgbp2
   ```

2. 安装Vercel CLI：
   ```bash
   npm install -g vercel
   ```

3. 部署到Vercel：
   ```bash
   vercel
   ```

## 数据更新

由于这是一个静态部署，数据不会自动更新。要更新数据，请按以下步骤操作：

1. 在本地运行数据获取和分析：
   ```bash
   pip install -r requirements.txt
   python main.py full
   ```

2. 这将生成最新的数据文件到 `data/` 目录

3. 重新部署到Vercel：
   ```bash
   vercel --prod
   ```

## 自动化更新（推荐）

您可以设置GitHub Actions来自动化数据更新流程：

1. 创建 `.github/workflows/update-data.yml` 文件
2. 配置定时任务（例如每天早上9点更新数据）
3. 当数据更新时自动重新部署到Vercel

## 项目结构

- `concept_ladder.html` - 主页面，包含所有前端逻辑和样式
- `data/kline_data.js` - K线数据文件
- `data/ladder_data.js` - 连板数据文件
- `vercel.json` - Vercel部署配置文件

## 功能特性

- 交互式连板天梯图
- 按连板天数分组展示
- 按概念题材分组展示
- K线图弹窗查看功能
- 日期导航和过滤选项

## 技术栈

- HTML/CSS/JavaScript
- ECharts (前端可视化)
- 静态数据文件

## 注意事项

- 此部署方案为纯静态网站，数据需要定期手动或自动更新
- 数据更新频率取决于您的需求，通常每天更新一次即可
- 如果需要实时数据，请考虑使用前后端分离的部署方案