import pandas as pd
import numpy as np
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
        # get content
        content = rq.get(url=url, verify=False).content
        df = pd.read_csv(StringIO(content.decode('utf8')))
        # check if empty - e.g. update existing over weekend
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        except:
            print('Warning: Market index data is empty!')
            None
        return df

    # returns panel using goodle finance server, pause is required to avoid ban
    def _get_historical_prices(self, tickers, start_date, end_date):

        # MorningStar provides a multiindex DF, so we need to convert it to
        # panelframe consistent with other routines
        pf = web.DataReader(tickers, 'morningstar', start_date, end_date)
        pf = pf.to_panel()
        pf = pf.swapaxes(1, 2)

        ### STOOQ section - working, but stooq has strict daily limits
        # # need to append ".US" to every symbol to read from stooq
        # # start and end dates are not implemented for stooq
        # st = StooqDailyReader(
        #     symbols=[i+'.US' for i in tickers],
        #     start=start_date, end=end_date,
        #     retry_count=3, pause=0.001, session=None, chunksize=25)
        # pf = st.read()
        # pf = pf.astype(np.float32)
        # # change tickers back to Robinhood style
        # pf.minor_axis = [i[:-3] for i in pf.minor_axis]
        # st.close()
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
    pf = md._get_historical_prices(
        # df_ord.symbol.unique(),
        ['BND', 'VTI'],
        df_ord.date.min(),
        df_ord.date.max())
