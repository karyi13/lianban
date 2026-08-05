"""
依赖注入容器
"""
from .fetchers import (
    PyTDXDataFetcher,
    AkShareDataFetcher,
    TencentDataFetcher,
    CompositeDataFetcher,
)
from .validation import DataValidator
from .storage import DataStorage
from .config import config


class DIContainer:
    """依赖注入容器"""

    def __init__(self):
        self._services = {}
        self._register_services()

    def _register_services(self):
        """注册服务"""
        # 注册数据获取器
        pytdx_fetcher = PyTDXDataFetcher()
        akshare_fetcher = AkShareDataFetcher()
        tencent_fetcher = TencentDataFetcher()
        composite_fetcher = CompositeDataFetcher(pytdx_fetcher, akshare_fetcher, tencent_fetcher)

        self._services['data_fetcher'] = composite_fetcher
        self._services['pytdx_fetcher'] = pytdx_fetcher
        self._services['akshare_fetcher'] = akshare_fetcher
        self._services['tencent_fetcher'] = tencent_fetcher

        # 注册数据验证器
        self._services['data_validator'] = DataValidator()

        # 注册数据存储器
        self._services['data_storage'] = DataStorage()

    def get(self, service_name: str):
        """获取服务实例"""
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not found")
        return self._services[service_name]


# 全局容器实例
container = DIContainer()
