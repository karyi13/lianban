"""
接口定义模块
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class DataFetcherInterface(ABC):
    """数据获取接口"""

    @abstractmethod
    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表"""

    @abstractmethod
    def fetch_daily_data(self, code: str, market: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取日线数据"""


class DataProcessorInterface(ABC):
    """数据处理接口"""

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理数据"""


class DataValidatorInterface(ABC):
    """数据验证接口"""

    @abstractmethod
    def validate(self, data: pd.DataFrame) -> tuple[bool, list]:
        """验证数据"""


class DataStorageInterface(ABC):
    """数据存储接口"""

    @abstractmethod
    def save(self, data: pd.DataFrame, file_path: str):
        """保存数据"""

    @abstractmethod
    def load(self, file_path: str) -> pd.DataFrame:
        """加载数据"""
