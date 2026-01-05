"""
A股连板分析工具 - 统一入口
包含数据获取、分析、可视化功能
"""
import pandas as pd
import numpy as np
from pytdx.hq import TdxHq_API
import akshare as ak
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import datetime
import time
import os
import threading
import json
import argparse
from typing import Optional
from function.stock_concepts import get_stock_concepts

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_START_DATE = '20241220'

def get_default_end_date():
    """
    Get the default end date based on current time.
    If market has closed (after 15:00), use today's date.
    If market hasn't closed (before 15:00), use yesterday's date.
    """
    now = datetime.datetime.now()
    current_time = now.time()
    
    # Market closes at 15:00
    market_close_time = datetime.time(15, 0, 0)
    
    if current_time >= market_close_time:
        # Market has closed, use today's date
        target_date = now.date()
    else:
        # Market hasn't closed, use yesterday's date
        target_date = now.date() - datetime.timedelta(days=1)
    
    # Skip weekends
    while target_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
        target_date = target_date - datetime.timedelta(days=1)
    
    return target_date.strftime('%Y%m%d')

DEFAULT_END_DATE = get_default_end_date()
DEFAULT_OUTPUT_FILE = os.path.join('data', 'stock_daily_latest.parquet')
MAX_WORKERS = 20  # Reduced workers to avoid connection spamming

# Thread-local storage for PyTDX connections
thread_local = threading.local()

def get_thread_api():
    """Get or create a thread-local PyTDX API connection."""
    if not hasattr(thread_local, "api"):
        api = TdxHq_API(heartbeat=True)
        # Try connecting
        try:
            if api.connect('121.36.81.195', 7709, time_out=10):
                thread_local.api = api
                return api
            # Fallbacks
            fallback_ips = [
                ('119.147.212.81', 7709),
                ('47.107.75.159', 7709)
            ]
            for ip, port in fallback_ips:
                if api.connect(ip, port, time_out=10):
                    thread_local.api = api
                    return api
        except:
            pass
        thread_local.api = None
    return thread_local.api

class DataFetcher:
    def __init__(self, output_file=DEFAULT_OUTPUT_FILE):
        # Initial connection just to test or get stock list
        self.main_api = TdxHq_API()
        self.connected = False
        self.output_file = output_file
        self.connect_main()

    def connect_main(self):
        try:
            if self.main_api.connect('121.36.81.195', 7709, time_out=10):
                self.connected = True
        except:
            pass

    def get_stock_list(self):
        """Get all A-share stocks."""
        try:
            logger.info("Fetching stock list via AkShare...")
            stock_df = ak.stock_zh_a_spot_em()
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

            logger.info(f"Found {len(stocks)} A-share stocks.")
            return stocks
        except Exception as e:
            logger.error(f"Error fetching stock list: {e}")
            return []

    def fetch_daily_pytdx_with_date_range(self, code, market, start_date, end_date):
        """Fetch daily data using Thread-Local PyTDX with specific date range."""
        api = get_thread_api()
        if not api:
            return None

        try:
            # market: 0 - SZ, 1 - SH
            # category: 9 - Day
            # Fetch more bars than needed to ensure we cover the date range
            data = api.get_security_bars(9, market, code, 0, 400)
            if not data:
                return None

            df = api.to_df(data)
            df['date'] = df['datetime'].apply(lambda x: x[:10].replace('-', ''))
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

            if df.empty:
                return pd.DataFrame()

            df = df.rename(columns={'vol': 'volume'})
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

        except Exception as e:
            return None

    def fetch_daily_akshare_with_date_range(self, code, symbol, start_date, end_date):
        """Fetch daily data using AkShare (fallback) with specific date range."""
        try:
            # AkShare is HTTP based, thread-safe usually
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            if df.empty:
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
            return df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        except Exception as e:
            # logger.error(f"AkShare fetch failed for {code}: {e}")
            return None

    def process_stock_with_date_range(self, stock_info, start_date, end_date):
        """Process a single stock with specified date range."""
        symbol = stock_info['symbol']
        code = stock_info['code']
        name = stock_info['name']

        market = 1 if symbol.endswith('.SH') else 0

        # Try PyTDX
        df = self.fetch_daily_pytdx_with_date_range(code, market, start_date, end_date)

        # Fallback to AkShare
        if df is None or df.empty:
            df = self.fetch_daily_akshare_with_date_range(code, symbol, start_date, end_date)

        if df is not None and not df.empty:
            df['symbol'] = symbol
            df['name'] = name
            return df

        return None

    def get_existing_data_date_range(self):
        """Get the date range of existing data in the output file."""
        if not os.path.exists(self.output_file):
            return None, None

        try:
            df = pd.read_parquet(self.output_file)
            if df.empty:
                return None, None

            # Convert date column to string format if it's not already
            df['date'] = df['date'].astype(str)
            min_date = df['date'].min()
            max_date = df['date'].max()
            return min_date, max_date
        except Exception as e:
            logger.error(f"Error reading existing data: {e}")
            return None, None

    def get_next_trading_day(self, date_str):
        """Simple function to get next trading day - accounts for weekends."""
        # This is a simplified version - in production, should check against actual trading calendar
        # For now, increment the date and skip weekends
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])

        import datetime
        current_date = datetime.date(year, month, day)

        # Keep incrementing until we find a weekday (Monday=0, Sunday=6)
        next_date = current_date + datetime.timedelta(days=1)
        while next_date.weekday() > 4:  # 5=Saturday, 6=Sunday
            next_date = next_date + datetime.timedelta(days=1)

        return next_date.strftime('%Y%m%d')

    def run(self, start_date=None, end_date=None, incremental=True):
        """
        Run the data fetcher.

        Args:
            start_date (str): Start date in format 'YYYYMMDD'. If None, uses DEFAULT_START_DATE
            end_date (str): End date in format 'YYYYMMDD'. If None, uses DEFAULT_END_DATE
            incremental (bool): Whether to perform incremental update or full refresh
        """
        self.start_date = start_date or DEFAULT_START_DATE
        self.end_date = end_date or DEFAULT_END_DATE

        # If incremental update is enabled and file exists, determine the date range to fetch
        if incremental and os.path.exists(self.output_file):
            logger.info("Incremental mode enabled. Checking existing data...")
            existing_min_date, existing_max_date = self.get_existing_data_date_range()

            if existing_min_date is not None and existing_max_date is not None:
                logger.info(f"Existing data range: {existing_min_date} to {existing_max_date}")

                # If the requested end date is not later than existing max date, no need to fetch
                if self.end_date <= existing_max_date:
                    logger.info(f"Requested end date {self.end_date} is not later than existing max date {existing_max_date}. No new data to fetch.")
                    return

                # Fetch only data from the day after existing max date to requested end date
                fetch_start_date = self.get_next_trading_day(existing_max_date)
                if fetch_start_date > self.end_date:
                    logger.info(f"No new trading days between {existing_max_date} and {self.end_date}")
                    return

                logger.info(f"Fetching incremental data from {fetch_start_date} to {self.end_date}")
                # Update the date range for fetching
                self.start_date = fetch_start_date
            else:
                logger.info("No existing data found or error reading existing data. Performing full fetch.")
        else:
            logger.info(f"Full refresh mode. Fetching data from {self.start_date} to {self.end_date}")

        stocks = self.get_stock_list()
        if not stocks:
            logger.error("No stocks found.")
            return

        all_data = []
        logger.info(f"Starting fetch for {len(stocks)} stocks with {MAX_WORKERS} workers...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_stock = {executor.submit(self.process_stock_with_date_range, stock_info, self.start_date, self.end_date): stock_info for stock_info in stocks}

            for i, future in enumerate(as_completed(future_to_stock)):
                res = future.result()
                if res is not None:
                    all_data.append(res)

                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(stocks)} stocks...")

        if all_data:
            new_data_df = pd.concat(all_data, ignore_index=True)
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_cols:
                new_data_df[col] = pd.to_numeric(new_data_df[col], errors='coerce')

            # If incremental mode and file exists, combine with existing data
            if incremental and os.path.exists(self.output_file):
                logger.info("Loading existing data to combine with new data...")
                try:
                    existing_df = pd.read_parquet(self.output_file)
                    # Combine existing and new data
                    combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
                    # Remove duplicates based on date and symbol
                    combined_df = combined_df.drop_duplicates(subset=['date', 'symbol'], keep='last')
                    # Sort by date and symbol for consistency
                    combined_df = combined_df.sort_values(['symbol', 'date']).reset_index(drop=True)
                    final_df = combined_df
                    logger.info(f"Combined {len(existing_df)} existing rows with {len(new_data_df)} new rows, total {len(final_df)} rows after deduplication")
                except Exception as e:
                    logger.error(f"Error combining existing and new data: {e}")
                    final_df = new_data_df
            else:
                final_df = new_data_df

            final_df.to_parquet(self.output_file, index=False)
            logger.info(f"Saved {len(final_df)} rows to {self.output_file}")
        else:
            logger.info("No new data fetched.")


class Analyzer:
    def __init__(self, input_file: str = DEFAULT_OUTPUT_FILE, output_ladder: str = os.path.join('data', 'limit_up_ladder.parquet'), output_promotion: str = os.path.join('data', 'promotion_rates.csv')):
        self.input_file = input_file
        self.output_ladder = output_ladder
        self.output_promotion = output_promotion
        self.df = None
        self.concepts_cache = {}

    def load_data(self):
        if not os.path.exists(self.input_file):
            logger.error(f"Input file {self.input_file} not found.")
            return False
        self.df = pd.read_parquet(self.input_file)
        return True

    def calculate_limit_price(self, row, prev_close):
        if pd.isna(prev_close):
            return np.nan

        # Determine multiplier
        if 'ST' in row['name']:
            multiplier = 1.05
        elif row['symbol'].startswith(('30', '68')): # ChiNext, STAR
            multiplier = 1.20
        else: # Main board
            multiplier = 1.10

        # Calculate limit price (Standard A-share rounding: round to 2 decimals)
        # Note: Python round() rounds to nearest even number for .5 cases sometimes.
        # Financial rounding usually requires standard rounding.
        # But simple round(x, 2) is often close enough for screening.
        # Strict formula: int(prev_close * (1 + ratio) * 100 + 0.5) / 100
        # Let's use the strict formula to be safe.

        limit = int(prev_close * multiplier * 100 + 0.49999) / 100.0
        return limit

    def identify_limit_ups(self):
        logger.info("identifying limit ups...")
        self.df.sort_values(by=['symbol', 'date'], inplace=True)

        # Vectorized prev_close
        self.df['prev_close'] = self.df.groupby('symbol')['close'].shift(1)

        # We need to apply logic row by row or vectorized?
        # Vectorized is hard due to "ST" name check and code check mix.
        # But we can split df into groups.

        # 1. ST stocks
        st_mask = self.df['name'].str.contains('ST', na=False)
        # 2. 20% stocks (30/68)
        twenty_pct_mask = self.df['symbol'].str.startswith(('30', '68'))
        # 3. 10% stocks (Rest)
        ten_pct_mask = (~st_mask) & (~twenty_pct_mask)

        # Calculate Limit Prices
        # Function to calc limit: int(prev * (1+r) * 100 + 0.5) / 100
        # We can use lambda

        def calc_limit_vec(prev_closes, ratio):
            return (np.floor(prev_closes * (1 + ratio) * 100 + 0.5) / 100)

        self.df['limit_price'] = np.nan

        # Apply 10%
        if ten_pct_mask.any():
            self.df.loc[ten_pct_mask, 'limit_price'] = calc_limit_vec(self.df.loc[ten_pct_mask, 'prev_close'], 0.10)

        # Apply 20%
        if twenty_pct_mask.any():
            self.df.loc[twenty_pct_mask, 'limit_price'] = calc_limit_vec(self.df.loc[twenty_pct_mask, 'prev_close'], 0.20)

        # Apply 5% (ST) - Note: ST overrides 20%? Usually ST on ChiNext is 20%?
        # Actually ChiNext ST is 20%. Main board ST is 5%.
        # Refinement: ST on 30xx/68xx is 20%. ST on Main is 5%.
        # So exclude 20% from ST mask for 5% rule.
        st_main_mask = st_mask & (~twenty_pct_mask)
        if st_main_mask.any():
            self.df.loc[st_main_mask, 'limit_price'] = calc_limit_vec(self.df.loc[st_main_mask, 'prev_close'], 0.05)

        # Compare
        # Allow small float error margin? Usually strict equal for limit up price,
        # but user said ">= ... - 1e-4".
        self.df['is_limit_up'] = self.df['close'] >= (self.df['limit_price'] - 0.001)

        # Filter only necessary columns for processing to save memory if needed
        # But we need full data.

    def calculate_consecutive_days(self):
        logger.info("Calculating consecutive days...")
        # Logic:
        # If is_limit_up is False, count = 0
        # If True, count = prev_count + 1

        # This is hard to vectorize fully across groups without loop, but we can use cumsum approach.
        # Group by symbol.

        # A common trick:
        # group = (is_limit_up != is_limit_up.shift()).cumsum()
        # count = groupby(group).cumcount() + 1
        # mask where is_limit_up is False => 0

        self.df['group'] = (self.df['is_limit_up'] != self.df.groupby('symbol')['is_limit_up'].shift(1)).cumsum()
        self.df['consecutive_limit_up_days'] = self.df.groupby(['symbol', 'group']).cumcount() + 1
        self.df.loc[~self.df['is_limit_up'], 'consecutive_limit_up_days'] = 0

    def identify_board_type(self):
        logger.info("Identifying board types...")
        # Default empty
        self.df['board_type'] = None

        limit_ups = self.df[self.df['is_limit_up']].copy()

        # One-word: high == low
        one_word_mask = limit_ups['high'] == limit_ups['low']
        limit_ups.loc[one_word_mask, 'board_type'] = '一字板'

        # T-shaped: open == high == close and low < close
        t_shape_mask = (limit_ups['open'] == limit_ups['high']) & \
                       (limit_ups['high'] == limit_ups['close']) & \
                       (limit_ups['low'] < limit_ups['close'])
        limit_ups.loc[t_shape_mask, 'board_type'] = 'T字板'

        # Turnover: The rest
        turnover_mask = limit_ups['board_type'].isna()
        limit_ups.loc[turnover_mask, 'board_type'] = '换手板'

        self.df.update(limit_ups)

    def process(self):
        if not self.load_data():
            return

        self.identify_limit_ups()
        self.calculate_consecutive_days()
        self.identify_board_type()

        # Filter for Ladder: consecutive >= 1 (include single-day limit-ups)
        ladder_df = self.df[self.df['consecutive_limit_up_days'] >= 1].copy()

        # Fetch concepts only for these ladder stocks
        unique_symbols = ladder_df['symbol'].unique()
        logger.info(f"Fetching concepts for {len(unique_symbols)} stocks in ladder...")

        # Fetch concepts for each stock
        def fetch_concept_for_stock(symbol):
            result = get_stock_concepts(symbol)
            if result and 'concepts' in result:
                return symbol, result['concepts']
            return symbol, []

        # Use ThreadPoolExecutor to fetch concepts in parallel
        concept_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_concept_for_stock, symbol): symbol for symbol in unique_symbols}
            for i, future in enumerate(as_completed(futures)):
                symbol, concepts = future.result()
                concept_map[symbol] = concepts
                if (i + 1) % 100 == 0:
                    logger.info(f"Fetched concepts for {i + 1}/{len(unique_symbols)} stocks...")

        # Map concepts back to ladder_df
        ladder_df['concept_themes'] = ladder_df['symbol'].map(concept_map)

        # Save Ladder
        ladder_df.to_parquet(self.output_ladder, index=False)
        logger.info(f"Saved ladder data to {self.output_ladder}")

        # Calculate Promotion Rates
        # Group by date + consecutive_days
        # For each date D, count(N board).
        # Check date D+1, how many of those stocks became N+1 board.

        stats = []
        dates = sorted(self.df['date'].unique())

        for i in range(len(dates) - 1):
            curr_date = dates[i]
            next_date = dates[i+1]

            # Get stocks with N boards today
            curr_df = self.df[self.df['date'] == curr_date]
            next_df = self.df[self.df['date'] == next_date]

            # Max consecutive days today
            max_boards = curr_df['consecutive_limit_up_days'].max()
            if pd.isna(max_boards) or max_boards == 0:
                continue

            for n in range(1, int(max_boards) + 1):
                # Stocks that were N board today
                n_board_stocks = curr_df[curr_df['consecutive_limit_up_days'] == n]['symbol'].tolist()
                total_n = len(n_board_stocks)

                if total_n == 0:
                    continue

                # Check their status tomorrow
                # They must have consecutive_limit_up_days == n + 1
                promoted_stocks = next_df[
                    (next_df['symbol'].isin(n_board_stocks)) &
                    (next_df['consecutive_limit_up_days'] == n + 1)
                ]
                promoted_count = len(promoted_stocks)

                rate = promoted_count / total_n if total_n > 0 else 0

                stats.append({
                    'date': curr_date,
                    'board_level': n,
                    'total': total_n,
                    'promoted': promoted_count,
                    'promotion_rate': rate
                })

        stats_df = pd.DataFrame(stats)
        stats_df.to_csv(self.output_promotion, index=False)
        logger.info(f"Saved promotion rates to {self.output_promotion}")


def generate_ladder_data_for_html(ladder_file: str = os.path.join('data','limit_up_ladder.parquet'), output_file: str = os.path.join('data','ladder_data.js')):
    """Generate ladder data for HTML visualization."""
    # Load the ladder data
    if not os.path.exists(ladder_file):
        print(f"{ladder_file} not found!")
        return

    df = pd.read_parquet(ladder_file)

    # Get all unique dates
    unique_dates = sorted(df['date'].unique(), reverse=True)

    # Create a dictionary to store data for each date
    all_dates_data = {}

    for date in unique_dates:
        # Filter data for this date
        date_data = df[df['date'] == date].copy()

        # Group by consecutive limit up days
        grouped = date_data.groupby('consecutive_limit_up_days').apply(
            lambda x: x[['symbol', 'name', 'close', 'consecutive_limit_up_days', 'concept_themes']].to_dict('records')
        ).to_dict()

        # Convert to the format expected by the HTML
        formatted_data = {}
        for level, stocks in grouped.items():
            formatted_data[level] = []
            for stock in stocks:
                # Convert concept_themes to list if it's a numpy array
                concepts = stock['concept_themes']
                if hasattr(concepts, 'tolist'):
                    concepts = concepts.tolist()
                elif not isinstance(concepts, list):
                    concepts = list(concepts) if concepts else []
                
                formatted_data[level].append({
                    'code': stock['symbol'],
                    'name': stock['name'],
                    'price': stock['close'],
                    'limitUpDays': stock['consecutive_limit_up_days'],
                    'conceptThemes': concepts
                })

        all_dates_data[str(date)] = formatted_data

    # Save as JS file that can be loaded by the HTML
    js_content = f"// 自动生成的连板数据文件\nwindow.LADDER_DATA = {json.dumps(all_dates_data, ensure_ascii=False)};"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)

    print(f"Generated ladder data for {len(all_dates_data)} dates to {output_file}")
    print(f"Latest date in data: {unique_dates[0] if unique_dates else 'None'}")

    # Also create a simple summary
    latest_date = unique_dates[0] if unique_dates else None
    if latest_date:
        latest_data = df[df['date'] == latest_date]
        print(f"Latest date ({latest_date}) has {len(latest_data)} limit-up stocks")
        level_counts = latest_data['consecutive_limit_up_days'].value_counts().sort_index(ascending=False)
        print("Distribution by consecutive days:")
        for level, count in level_counts.items():
            print(f"  {level}连板: {count}只")




def generate_kline_data(input_file: str = os.path.join('data','stock_daily_latest.parquet'), output_file: str = os.path.join('data','kline_data.js')):
    """Generate K-line JS data file for HTML visualization."""
    if not os.path.exists(input_file):
        print(f"{input_file} not found!")
        return

    df = pd.read_parquet(input_file)
    df_sorted = df.sort_values(['symbol', 'date'])
    df_sorted['date_formatted'] = pd.to_datetime(df_sorted['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
    grouped = df_sorted.groupby('symbol')
    kline_data = {}
    for symbol, group in grouped:
        kline_data[symbol] = {
            'name': group['name'].iloc[0] if 'name' in group.columns else '',
            'dates': group['date_formatted'].tolist(),
            'values': group[['open', 'close', 'low', 'high']].values.tolist(),
            'volumes': group['volume'].tolist()
        }
    js_content = f"// 自动生成的K线数据文件\nwindow.KLINE_DATA_GLOBAL = {json.dumps(kline_data, ensure_ascii=False)};"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    print(f"Saved K-line data to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='A股连板分析工具 - 统一入口')
    parser.add_argument('command', choices=['fetch', 'analyze', 'generate-ladder', 'generate-kline', 'visualize', 'full'], 
                        help='执行的命令: fetch(获取数据), analyze(分析数据), generate-ladder(生成阶梯数据), generate-kline(生成K线数据JS), visualize(生成可视化), full(完整流程)')
    parser.add_argument('--start-date', type=str, help='开始日期 YYYYMMDD')
    parser.add_argument('--end-date', type=str, help='结束日期 YYYYMMDD')
    parser.add_argument('--input-file', type=str, default=DEFAULT_OUTPUT_FILE, help='输入文件')
    parser.add_argument('--output-file', type=str, help='输出文件名')
    parser.add_argument('--incremental', action='store_true', default=True, help='是否增量更新')
    parser.add_argument('--full-refresh', action='store_true', help='完全刷新模式')

    args = parser.parse_args()

    # Determine output file name
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = DEFAULT_OUTPUT_FILE  # Use the default file name 'stock_daily_latest.parquet'

    if args.command == 'fetch':
        fetcher = DataFetcher(output_file=output_file)
        fetcher.run(
            start_date=args.start_date,
            end_date=args.end_date,
            incremental=not args.full_refresh
        )
    elif args.command == 'analyze':
        analyzer = Analyzer(input_file=args.input_file)
        analyzer.process()
    elif args.command == 'generate-ladder':
        generate_ladder_data_for_html()
    elif args.command == 'visualize':
        generate_html()
    elif args.command == 'generate-kline':
        out_js = args.output_file if args.output_file else os.path.join('data','kline_data.js')
        generate_kline_data(input_file=args.input_file, output_file=out_js)
    elif args.command == 'full':
        # 执行完整流程
        logger.info("开始执行完整分析流程...")
        
        # 1. 获取数据
        fetcher = DataFetcher(output_file=output_file)
        fetcher.run(
            start_date=args.start_date,
            end_date=args.end_date,
            incremental=not args.full_refresh
        )
        
        # 2. 分析数据
        analyzer = Analyzer(input_file=output_file)
        analyzer.process()
        
        # 3. 生成阶梯数据
        generate_ladder_data_for_html()
        
        # 4. 生成可视化
        generate_html()
        
        logger.info("完整分析流程完成！")


if __name__ == "__main__":
    main()