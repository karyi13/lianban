"""
数据获取器实现模块
"""
import logging
import threading
import time
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd
import requests
from pytdx.hq import TdxHq_API

from .config import config
from .interfaces import DataFetcherInterface
from .monitoring import monitoring_manager


logger = logging.getLogger(__name__)

# Thread-local storage for PyTDX connections
thread_local = threading.local()


def get_thread_api():
    """Get or create a thread-local PyTDX API connection."""
    if not hasattr(thread_local, "api"):
        api = TdxHq_API(heartbeat=True)
        # Try connecting, falling back through the server list
        try:
            for server in config.PYTDX_SERVERS:
                if api.connect(server[0], server[1], time_out=config.REQUEST_TIMEOUT):
                    thread_local.api = api
                    return api
        except Exception:
            pass
        thread_local.api = None
    return thread_local.api


def parse_stock_list(stock_df: pd.DataFrame) -> List[Dict[str, str]]:
    """从 AkShare 股票列表 DataFrame 解析 A 股股票信息。"""
    stocks = []
    for _, row in stock_df.iterrows():
        symbol = str(row['代码'])
        name = row['名称']
        if symbol.startswith(('900', '200')):
            continue

        if symbol.startswith(('60', '68')):
            full_symbol = f"{symbol}.SH"
        elif symbol.startswith(('00', '30')):
            full_symbol = f"{symbol}.SZ"
        else:
            continue

        stocks.append({'symbol': full_symbol, 'code': symbol, 'name': name})
    return stocks


class _RequestTracker:
    """跟踪一次请求的耗时与结果，统一记录监控指标。"""

    def __init__(self):
        self.start_time = time.time()
        self.success = False
        self.error_msg = ""

    def mark_success(self, error_msg: str = ""):
        self.success = True
        self.error_msg = error_msg

    def mark_failure(self, error_msg: str):
        self.success = False
        self.error_msg = error_msg

    def record(self):
        monitoring_manager.record_request(
            success=self.success,
            response_time=time.time() - self.start_time,
            error_msg=self.error_msg,
        )


class PyTDXDataFetcher(DataFetcherInterface):
    """PyTDX数据获取实现"""

    def __init__(self):
        self.main_api = TdxHq_API()
        self.connected = False
        self.connect_main()

    def connect_main(self):
        try:
            primary_server = config.PYTDX_SERVERS[0]
            if self.main_api.connect(primary_server[0], primary_server[1], time_out=config.REQUEST_TIMEOUT):
                self.connected = True
        except Exception:
            pass

    def get_stock_list(self) -> List[Dict[str, str]]:
        """Get all A-share stocks."""
        try:
            logger.info("Fetching stock list via AkShare...")
            stock_df = ak.stock_zh_a_spot_em()
            stocks = parse_stock_list(stock_df)
            logger.info(f"Found {len(stocks)} A-share stocks.")
            return stocks
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            raise  # Re-raise the exception to trigger the retry decorator

    def fetch_daily_data(self, code: str, market: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch daily data using Thread-Local PyTDX with specific date range."""
        tracker = _RequestTracker()
        api = get_thread_api()

        if not api:
            tracker.mark_failure(f"PyTDX API connection failed for {code}")
            tracker.record()
            return None

        try:
            # market: 0 - SZ, 1 - SH
            # category: 9 - Day
            # Fetch more bars than needed to ensure we cover the date range
            data = api.get_security_bars(9, market, code, 0, 400)
            if not data:
                tracker.mark_failure(f"No data returned from PyTDX for {code}")
                tracker.record()
                return None

            df = api.to_df(data)
            df['date'] = df['datetime'].apply(lambda x: x[:10].replace('-', ''))
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            if df.empty:
                tracker.mark_success(f"Empty data for {code} in date range {start_date}-{end_date}")
                tracker.record()
                return pd.DataFrame()

            df = df.rename(columns={'vol': 'volume'})
            tracker.mark_success()
            tracker.record()
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        except Exception as e:
            error_msg = f"PyTDX fetch failed for {code}: {str(e)}"
            logger.error(error_msg)
            tracker.mark_failure(error_msg)
            tracker.record()
            return None


class AkShareDataFetcher(DataFetcherInterface):
    """AkShare数据获取实现"""

    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表 - 使用AkShare"""
        try:
            logger.info("Fetching stock list via AkShare...")
            stock_df = ak.stock_zh_a_spot_em()
            stocks = parse_stock_list(stock_df)
            logger.info(f"Found {len(stocks)} A-share stocks.")
            return stocks
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            raise

    def fetch_daily_data(self, code: str, market: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch daily data using AkShare (fallback) with specific date range."""
        tracker = _RequestTracker()

        try:
            # AkShare is HTTP based, thread-safe usually
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty:
                tracker.mark_success(f"Empty data from AkShare for {code}")
                tracker.record()
                return pd.DataFrame()

            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })
            df['date'] = df['date'].astype(str).str.replace('-', '')
            tracker.mark_success()
            tracker.record()
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        except Exception as e:
            error_msg = f"AkShare fetch failed for {code}: {str(e)}"
            logger.error(error_msg)
            tracker.mark_failure(error_msg)
            tracker.record()
            raise  # Re-raise the exception to trigger the retry decorator


class TencentDataFetcher(DataFetcherInterface):
    """腾讯行情数据获取实现

    腾讯行情接口(web.ifzq.gtimg.cn)返回不复权K线数据，字段顺序为
    [date, open, close, high, low, volume]，不包含成交额。
    """

    def __init__(self, stock_list_file: Optional[str] = None):
        self.stock_list_file = stock_list_file or config.DEFAULT_OUTPUT_FILE
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表 - 从现有数据文件读取（腾讯无完整列表接口）"""
        import os
        if not os.path.exists(self.stock_list_file):
            raise FileNotFoundError(f"Stock list file not found: {self.stock_list_file}")

        try:
            df = pd.read_parquet(self.stock_list_file)
            if df.empty:
                raise ValueError(f"Stock list file is empty: {self.stock_list_file}")

            stocks_df = df[['symbol', 'name']].drop_duplicates('symbol')
            stocks = []
            for _, row in stocks_df.iterrows():
                symbol = str(row['symbol'])
                code = symbol.split('.')[0]
                stocks.append({'symbol': symbol, 'code': code, 'name': str(row['name'])})
            logger.info(f"Found {len(stocks)} A-share stocks from {self.stock_list_file}.")
            return stocks
        except Exception as e:
            logger.error(f"Error reading stock list from {self.stock_list_file}: {e}")
            raise

    @staticmethod
    def _to_tencent_symbol(code: str, market: int) -> str:
        """转换为腾讯行情代码格式（sh/sz 前缀）"""
        code = code.zfill(6)
        if market == 1:
            return f"sh{code}"
        return f"sz{code}"

    def fetch_daily_data(self, code: str, market: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch daily data using Tencent kline API with specific date range."""
        tracker = _RequestTracker()
        tencent_symbol = self._to_tencent_symbol(code, market)

        try:
            # start_date/end_date are YYYYMMDD, API expects YYYY-MM-DD
            start_fmt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
            end_fmt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"
            params = {
                "param": f"{tencent_symbol},day,{start_fmt},{end_fmt},{config.TENCENT_KLINE_COUNT}"
            }
            res = self.session.get(config.TENCENT_KLINE_URL, params=params, timeout=config.REQUEST_TIMEOUT)
            if res.status_code != 200:
                tracker.mark_failure(f"Tencent API returned status {res.status_code} for {code}")
                tracker.record()
                return None

            data = res.json()
            stock_data = (data.get('data') or {}).get(tencent_symbol, {})
            rows = stock_data.get('day') or stock_data.get('qfqday') or []
            if not rows:
                tracker.mark_failure(f"No data returned from Tencent for {code}")
                tracker.record()
                return None

            df = pd.DataFrame(rows).iloc[:, :6]
            df.columns = ['date', 'open', 'close', 'high', 'low', 'volume']
            df = df.astype({
                'open': float, 'close': float, 'high': float, 'low': float, 'volume': float
            })
            df['date'] = df['date'].str.replace('-', '')
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            if df.empty:
                tracker.mark_success(f"Empty data for {code} in date range {start_date}-{end_date}")
                tracker.record()
                return pd.DataFrame()

            # 腾讯接口不含成交额，用 手数*100股*收盘价 近似
            df['amount'] = df['volume'] * 100 * df['close']
            tracker.mark_success()
            tracker.record()
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        except Exception as e:
            error_msg = f"Tencent fetch failed for {code}: {str(e)}"
            logger.error(error_msg)
            tracker.mark_failure(error_msg)
            tracker.record()
            return None


class CompositeDataFetcher(DataFetcherInterface):
    """复合数据获取器，结合PyTDX和AkShare"""

    def __init__(self, pytdx_fetcher: PyTDXDataFetcher, akshare_fetcher: AkShareDataFetcher,
                 tencent_fetcher: Optional[TencentDataFetcher] = None):
        self.pytdx_fetcher = pytdx_fetcher
        self.akshare_fetcher = akshare_fetcher
        self.tencent_fetcher = tencent_fetcher

    def get_stock_list(self) -> List[Dict[str, str]]:
        """获取股票列表，优先使用AkShare，失败后使用腾讯（从数据文件读取）"""
        try:
            return self.akshare_fetcher.get_stock_list()
        except Exception as e:
            logger.warning(f"AkShare stock list failed: {e}, falling back to Tencent...")
            if self.tencent_fetcher is not None:
                return self.tencent_fetcher.get_stock_list()
            raise

    def fetch_daily_data(self, code: str, market: int, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取日线数据，优先使用PyTDX，失败后使用AkShare，再失败使用腾讯"""
        # Try PyTDX first
        df = self.pytdx_fetcher.fetch_daily_data(code, market, start_date, end_date)

        # Fallback to AkShare
        if df is None or df.empty:
            try:
                df = self.akshare_fetcher.fetch_daily_data(code, market, start_date, end_date)
            except Exception as e:
                logger.warning(f"AkShare fetch failed for {code}: {e}, falling back to Tencent...")
                df = None

        # Final fallback to Tencent
        if (df is None or df.empty) and self.tencent_fetcher is not None:
            df = self.tencent_fetcher.fetch_daily_data(code, market, start_date, end_date)

        return df
