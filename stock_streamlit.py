import pandas as pd
import os
import numpy as np
import csv
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper

import json

use_local_data = True
if use_local_data == False:
    print("not  local")
    os.environ["ALPHAVANTAGE_API_KEY"] = os.getenv('ALPHAVANTAGE_KEY')
    alpha_vantage = AlphaVantageAPIWrapper()
    stock_data = alpha_vantage._get_time_series_weekly("IBM")
else:
    print("using local")
    f = open('weekly.json')
    stock_data = json.load(f)

def parse_data_stock(stock_data):
    df = pd.DataFrame(stock_data)
    count = 0
    year = 2023
    parsed_stock_data = pd.DataFrame()
    for index, row in df.iterrows():
        if count > 4:
            current_year = index[:4]
            current_closing_price = row['Weekly Time Series']['4. close']
            volume = row['Weekly Time Series']['5. volume']
            if int(current_year) == year:
                row_to_append = pd.DataFrame({'key': ['year', 'volume', 'closing_price'], 'value': [year, volume, current_closing_price]})
                parsed_stock_data = pd.concat([parsed_stock_data, row_to_append])
                year = year - 1
        count += 1
    return parsed_stock_data

parsed_stock_data = parse_data_stock(stock_data)

