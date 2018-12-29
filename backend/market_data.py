import os
import pandas as pd
import requests as rq
import pandas_datareader.data as web
from io import StringIO


# download market benchmark data
# indices: ^dji, ^spx
class MarketData:
    def __init__(self, datafile='', index='^spx'):
        self.datafile = datafile
        self.index = index
        self._date_fmt = '{:%Y%m%d}'

    # convert dates to required format
    def _dates(self, date):
        return self._date_fmt.format(date)

    # startDate, endDate in yyyymmdd format
    def _get_market_index(self, start_date, end_date):
        url = "https://stooq.com/q/d/l/?s={}&d1={}&d2={}&i=d"
        url = url.format(self.index, start_date, end_date)
        content = rq.get(url=url, verify=False).content
        df = pd.read_csv(StringIO(content.decode('utf8')))
        # check if empty - e.g. update existing over weekend
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except KeyError:
            print('Warning: Market index data is empty!')
            None
        # convert columns names to lower() in all frames
        df.columns = [c.lower() for c in df.columns]
        return df

    # returns panel with TIINGO, expecting TIINGO_API_KEY in env
    def _get_historical_prices(self, tickers, start_date, end_date):

        try:
            api_key = os.getenv('TIINGO_API_KEY')
        except KeyError:
            print("Missing TIINGO_API_KEY")

        # Get data from TIINGO
        pf = web.get_data_tiingo(
            symbols=tickers,
            api_key=api_key,
            start=start_date,
            end=end_date)
        pf = pf.to_panel()
        pf = pf.swapaxes(1, 2)
        return pf

    # return all stocks and index in one panel
    def download_save_market_data(self, tickers, start_date, end_date,
                                  update_existing=False):
        start_date_str = self._date_fmt.format(start_date)
        end_date_str = self._date_fmt.format(end_date)
        print("Downloading market data for {}-{}".format(
            start_date_str, end_date_str))

        # add market index
        pf = self._get_historical_prices(
            tickers,
            start_date.date(),
            end_date.date())
        pf.loc[:, :, 'market'] = self._get_market_index(
            start_date_str, end_date_str)

        if update_existing:
            new_dict = {}
            pf_old = pd.read_hdf(self.datafile, 'market')
            pf_new = pd.concat([pf_old, pf], axis=1)
            for it in pf_new.items:
                new_dict[it] = pf_new.loc[it].drop_duplicates().sort_index()
            pf = pd.Panel(new_dict)
        else:
            pf.to_hdf(self.datafile, 'market')
        return pf


if __name__ == '__main__':
    print("Testing MarketData")
    md = MarketData(datafile='../data/data.h5')
    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    df = md._get_market_index(
        df_ord.date.min(),
        df_ord.date.max())

    pf = md.download_save_market_data(
        df_ord.symbol.unique(),
        df_ord.date.min(),
        df_ord.date.max())
