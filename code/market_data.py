import pandas as pd
import numpy as np
from pandas_datareader import data


# download market benchmark data
# indices: ^dji, ^spx
class MarketData:
    def __init__(self, index='^spx'):
        self.index = index
        return None

    # startDate, endDate in yyyymmdd format
    def _get_market_index(self, start_date, end_date):
        url = "https://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d"
        url = url.format(self.index, start_date, end_date)
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    # returns panel
    def _get_historical_prices(self, tickers, start_date, end_date):
        pf = data.DataReader(tickers, "google", start_date, end_date)
        pf = pf.astype(np.float32)
        return pf

    # return all stocks and index in one panel
    def get_data(self, tickers, start_date, end_date):
        pf = self._get_historical_prices(tickers, start_date, end_date)
        idx = self._get_market_index(start_date, end_date)
        pf.ix[:, :, 'market'] = idx
        return pf
