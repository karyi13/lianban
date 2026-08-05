"""
数据验证实现模块
"""
import pandas as pd

from .interfaces import DataValidatorInterface


class DataValidator(DataValidatorInterface):
    """数据验证实现"""

    def validate(self, df) -> tuple[bool, list]:
        """
        验证股票数据的完整性与合理性

        Args:
            df (pd.DataFrame): 股票数据DataFrame

        Returns:
            tuple: (is_valid, validation_report)
        """
        if df.empty:
            return False, ["数据为空"]

        validation_report = []
        is_valid = True

        # 检查必需列是否存在
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_report.append(f"缺少必要列: {missing_columns}")
            is_valid = False

        # 检查数据类型
        if 'date' in df.columns:
            # 尝试转换日期格式
            try:
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
                invalid_dates = df['date'].isna().sum()
                if invalid_dates > 0:
                    validation_report.append(f"无效日期格式: {invalid_dates} 条记录")
            except Exception:
                validation_report.append("日期格式转换失败")
                is_valid = False

        # 检查数值列的合理性
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                # 检查负值
                negative_values = (df[col] < 0).sum()
                if negative_values > 0:
                    validation_report.append(f"{col} 列存在 {negative_values} 个负值")

                # 检查异常值（如价格为0或异常高）
                if col in ['open', 'high', 'low', 'close']:
                    zero_prices = (df[col] == 0).sum()
                    if zero_prices > 0:
                        validation_report.append(f"{col} 列存在 {zero_prices} 个零价格")

                    # 检查价格是否合理（比如超过10000元的股票可能需要检查）
                    high_prices = (df[col] > 10000).sum()
                    if high_prices > 0:
                        validation_report.append(f"{col} 列存在 {high_prices} 个异常高价格(>10000)")

        # 检查 OHLC 关系的合理性
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            if invalid_ohlc > 0:
                validation_report.append(f"OHLC关系不合理: {invalid_ohlc} 条记录")
                is_valid = False

        # 检查重复数据
        duplicate_rows = df.duplicated(subset=['symbol', 'date']).sum()
        if duplicate_rows > 0:
            validation_report.append(f"存在 {duplicate_rows} 条重复数据")

        # 检查缺失值
        total_missing = df.isnull().sum().sum()
        if total_missing > 0:
            validation_report.append(f"存在 {total_missing} 个缺失值")

        return is_valid, validation_report
