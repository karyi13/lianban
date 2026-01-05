#!/bin/bash
# 部署脚本

echo "开始部署A股连板分析工具到Vercel..."

# 检查是否已安装Vercel CLI
if ! command -v vercel &> /dev/null; then
    echo "正在安装Vercel CLI..."
    npm install -g vercel
fi

# 登录Vercel（如果尚未登录）
vercel login

# 部署到Vercel
echo "正在部署到Vercel..."
vercel --prod

echo "部署完成！"