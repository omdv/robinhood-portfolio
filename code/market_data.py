import pandas as pd
import numpy as np
from pandas_datareader import data


# download market benchmark data
# indices: ^dji, ^spx
class MarketData:
    def __init__(self, datafile='', index='^spx'):
        self.datafile = datafile
        self.index = index

    # startDate, endDate in yyyymmdd format
    def _get_market_index(self, start_date, end_date):
        url = "https://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d"
        url = url.format(self.index, start_date, end_date)
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df

    # returns panel using goodle finance server, pause is required to avoid ban
    def _get_historical_prices(self, tickers, start_date, end_date):
        pf = data.DataReader(tickers, "google", start_date, end_date, pause=10)
        pf = pf.astype(np.float32)
        return pf

    # download treasury bills yields for different periods for capm model
    def _get_treasury_yields(self, start_date, end_date):
        tickers = ["TB4WK", "TB3MS", "TB1YR"]
        pf = data.DataReader(tickers, "fred", start_date, end_date)
        return pf

    # return all stocks and index in one panel
    def download_market_data(self, tickers, start_date, end_date):
        pf = self._get_historical_prices(tickers, start_date, end_date)
        df = self._get_market_index(start_date, end_date)
        tb = self._get_treasury_yields(start_date, end_date)
        pf.ix[:, :, 'market'] = df
        pf.to_hdf(self.datafile, 'market')
        tb.to_hdf(self.datafile, 'treasury_bills')
        return pf


if __name__ == '__main__':
    print("Testing MarketData")
    md = MarketData(datafile='../data/data.h5')
    md.download_market_data(['BND'], '20170701', '20170720')
