import pandas as pd
import os
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

def generate_weekly_csv_data_stock(stock_data):
    df = pd.DataFrame(stock_data)
    count = 0
    year = 2023
    parsed_volume_data = pd.DataFrame()
    i= 0
    week_number = 1248
    for index, row in df.iterrows():
        if count > 4:
            closing_price = row['Weekly Time Series']['5. volume']
            volume = row['Weekly Time Series']['5. volume']
            if week_number > 0 and year <= 2023:
                row_to_append = pd.DataFrame([{'week_number': week_number, 'volume': volume, 'closing_price': closing_price}])
                parsed_volume_data = pd.concat([parsed_volume_data, row_to_append])
                year = year - 1
                week_number = week_number - 1
        count += 1
    return parsed_volume_data

parsed_volume_data = generate_weekly_csv_data_stock(stock_data)

parsed_volume_data.to_csv('csv_weekly_data.csv', index=False)


