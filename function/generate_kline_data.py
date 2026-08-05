"""
K线数据生成模块 - 生成前端可视化所需 JS 文件

本模块是 generate_kline_data 的唯一实现，main.py 从中导入。
"""
import json
import logging
import os

import pandas as pd

from depend.config import config
from utils.logging_utils import performance_monitor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_kline_data(input_file: str = config.DEFAULT_OUTPUT_FILE, output_file: str = config.DEFAULT_KLINE_JS_FILE, chunk_size: int = config.CHUNK_SIZE):
    """Generate K-line JS data file for HTML visualization."""
    logger.info("Starting K-line data generation", {
        "input_file": input_file,
        "output_file": output_file,
        "chunk_size": chunk_size
    })

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    # 由于pandas的read_parquet不支持chunksize参数，我们直接加载整个文件
    timer_id = performance_monitor.start_timer("load_kline_data")
    df = pd.read_parquet(input_file)
    load_duration = performance_monitor.end_timer(timer_id)
    logger.info(f"Loaded K-line data: {len(df)} records", {
        "duration_seconds": round(load_duration, 4),
        "columns_count": len(df.columns)
    })

    timer_id = performance_monitor.start_timer("process_kline_data")
    df_sorted = df.sort_values(['symbol', 'date'])
    df_sorted['date_formatted'] = pd.to_datetime(df_sorted['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
    grouped = df_sorted.groupby('symbol')
    kline_data = {}
    total_symbols = len(grouped)

    for i, (symbol, group) in enumerate(grouped):
        kline_data[symbol] = {
            'name': group['name'].iloc[0] if 'name' in group.columns else '',
            'dates': group['date_formatted'].tolist(),
            'values': group[['open', 'close', 'low', 'high']].values.tolist(),
            'volumes': group['volume'].tolist()
        }

        # Log progress every 100 symbols
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{total_symbols} symbols...")

    process_duration = performance_monitor.end_timer(timer_id)
    logger.info(f"Completed processing K-line data", {
        "duration_seconds": round(process_duration, 4),
        "symbols_count": len(kline_data)
    })

    timer_id = performance_monitor.start_timer("save_kline_js_file")
    js_content = f"// 自动生成的K线数据文件\nwindow.KLINE_DATA_GLOBAL = {json.dumps(kline_data, ensure_ascii=False)};"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)
    save_duration = performance_monitor.end_timer(timer_id)

    logger.info(f"Saved K-line data to {output_file}", {
        "duration_seconds": round(save_duration, 4),
        "symbols_count": len(kline_data)
    })


if __name__ == "__main__":
    generate_kline_data()
