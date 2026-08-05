"""
数据存储实现模块
"""
import logging
import os
from pathlib import Path

import pandas as pd

from .backup_manager import backup_manager
from .interfaces import DataStorageInterface


logger = logging.getLogger(__name__)


def _detect_format(file_path: str) -> str:
    """根据文件扩展名推断存储格式，默认 parquet。"""
    ext = Path(file_path).suffix.lower()
    if ext == '.csv':
        return 'csv'
    if ext == '.json':
        return 'json'
    return 'parquet'


def _backup_path(file_path: str) -> str:
    """生成同格式的备份文件路径（保留目录前缀）。"""
    ext = Path(file_path).suffix
    if not ext:
        return f"{file_path}_backup"
    return f"{file_path[:-len(ext)]}_backup{ext}"


class DataStorage(DataStorageInterface):
    """数据存储实现"""

    def save(self, data, file_path: str, create_backup: bool = False):
        """保存数据到文件，根据文件扩展名选择格式"""
        # 如果目标文件存在且需要备份，则创建备份
        if create_backup and os.path.exists(file_path):
            backup_manager.create_backup(file_path)

        try:
            self._write(data, file_path)
            logger.info(f"Saved data to {file_path}")
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {e}")
            # Try to save to a backup file
            fallback_file = _backup_path(file_path)
            try:
                self._write(data, fallback_file)
                logger.info(f"Saved data to backup file: {fallback_file}")
            except Exception as backup_error:
                logger.error(f"Error saving to backup file: {backup_error}")

    def load(self, file_path: str, from_backup: bool = False) -> pd.DataFrame:
        """从文件加载数据，根据文件扩展名选择格式"""
        if from_backup:
            # 尝试从备份恢复
            backup_files = backup_manager.list_backups(f"*{os.path.basename(file_path)}*")
            if backup_files:
                # 使用最新的备份
                latest_backup = sorted(backup_files, key=os.path.getmtime, reverse=True)[0]
                backup_manager.restore_from_backup(latest_backup, file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        fmt = _detect_format(file_path)
        if fmt == 'csv':
            return pd.read_csv(file_path)
        if fmt == 'json':
            return pd.read_json(file_path)
        return pd.read_parquet(file_path)

    @staticmethod
    def _write(data, file_path: str):
        """按文件扩展名写入数据。"""
        fmt = _detect_format(file_path)
        if fmt == 'csv':
            data.to_csv(file_path, index=False)
        elif fmt == 'json':
            data.to_json(file_path, orient='records', force_ascii=False, indent=2)
        else:
            data.to_parquet(file_path, index=False)
