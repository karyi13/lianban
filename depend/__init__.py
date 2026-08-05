"""
depend 包 - 数据层服务模块

按职责拆分为多个文件：
- config.py: 配置
- interfaces.py: 接口定义
- fetchers.py: 数据获取器 (PyTDX / AkShare / Tencent / Composite)
- validation.py: 数据验证
- storage.py: 数据存储
- monitoring.py: 监控指标
- backup_manager.py: 数据备份
- di_container.py: 依赖注入容器
"""
from .config import config
from .di_container import container
from .fetchers import PyTDXDataFetcher, AkShareDataFetcher, TencentDataFetcher, CompositeDataFetcher, get_thread_api
from .validation import DataValidator
from .storage import DataStorage
from .backup_manager import backup_manager
from .monitoring import monitoring_manager

__all__ = [
    'config',
    'container',
    'PyTDXDataFetcher',
    'AkShareDataFetcher',
    'TencentDataFetcher',
    'CompositeDataFetcher',
    'get_thread_api',
    'DataValidator',
    'DataStorage',
    'backup_manager',
    'monitoring_manager',
]
