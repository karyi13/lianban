"""
服务实现模块 - 兼容层

原 services.py 已拆分为三个职责单一的文件：
- fetchers.py: 数据获取器 (PyTDX / AkShare / Composite)
- validation.py: 数据验证
- storage.py: 数据存储

本文件保留所有导出符号，保证旧引用兼容。
"""
from .fetchers import (
    PyTDXDataFetcher,
    AkShareDataFetcher,
    TencentDataFetcher,
    CompositeDataFetcher,
    get_thread_api,
    parse_stock_list,
)
from .validation import DataValidator
from .storage import DataStorage

__all__ = [
    'PyTDXDataFetcher',
    'AkShareDataFetcher',
    'TencentDataFetcher',
    'CompositeDataFetcher',
    'DataValidator',
    'DataStorage',
    'get_thread_api',
    'parse_stock_list',
]
