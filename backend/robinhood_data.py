import pandas as pd
import numpy as np
from backend.robinhood_api import RobinhoodAPI
from backend.auth import user, password


class RobinhoodData:
    """
    Wrapper to download orders and dividends from Robinhood accounts
    Downloads two dataframes and saves to datafile
    ----------
    Parameters:
    datafile : location of h5 datafile
    """
    def __init__(self, datafile):
        self.datafile = datafile

    def _login(self):
        self.client = RobinhoodAPI()
        self.client.login(username=user, password=password)
        return self

    # private method for getting all orders
    def _fetch_json_by_url(self, url):
        return self.client.session.get(url).json()

    # deleting sensitive or redundant fields
    def _delete_sensitive_fields(self, df):
        for col in ['account', 'url', 'id', 'instrument']:
            if col in df:
                del df[col]
        return df

    # download positions and all fields requiring RB client
    def _download_positions(self):
        positions = self.client.positions()
        positions = [x for x in positions['results']]
        df = pd.DataFrame(positions)
        df['symbol'] = df['instrument'].apply(
            self.client.get_symbol_by_instrument)
        df['name'] = df['instrument'].apply(
            self.client.get_name_by_instrument)
        self.df_pos = self._delete_sensitive_fields(df)
        return self

    # download orders and fields requiring RB client
    def _download_orders(self):
        orders = []
        past_orders = self.client.order_history()
        orders.extend(past_orders['results'])
        while past_orders['next']:
            next_url = past_orders['next']
            past_orders = self._fetch_json_by_url(next_url)
            orders.extend(past_orders['results'])
        df = pd.DataFrame(orders)
        df['symbol'] = df['instrument'].apply(
            self.client.get_symbol_by_instrument)
        df.sort_values(by='created_at', inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.df_ord = self._delete_sensitive_fields(df)
        return self

    # download dividends and fields requiring RB client
    def _download_dividends(self):
        dividends = self.client.dividends()
        dividends = [x for x in dividends['results']]
        df = pd.DataFrame(dividends)
        df['symbol'] = df['instrument'].apply(
            self.client.get_symbol_by_instrument)
        df.sort_values(by='paid_at', inplace=True)
        df.reset_index(inplace=True, drop=True)
        self.df_div = self._delete_sensitive_fields(df)
        return self

    # process orders
    def _process_orders(self):
        # assign to df and reduce the number of fields
        df = self.df_ord.copy()
        fields = [
            'created_at',
            'average_price', 'price', 'cumulative_quantity', 'fees',
            'symbol', 'side']
        df = df[fields]

        # convert types
        for field in ['average_price', 'price',
                      'cumulative_quantity', 'fees']:
            df[field] = pd.to_numeric(df[field])
        for field in ['created_at']:
            df[field] = pd.to_datetime(df[field])

        # add days
        df['date'] = df['created_at'].apply(
            lambda x: pd.tslib.normalize_date(x))

        # quantity accounting for side of transaction
        df['signed_quantity'] = np.where(
            df.side == 'buy',
            df['cumulative_quantity'],
            -df['cumulative_quantity'])
        df['signed_quantity'] = df['signed_quantity'].astype(np.int64)

        # calculate cost_basis at the moment of the order
        df['cost_basis'] = df['signed_quantity'] * df['average_price']

        # group by days
        df = df.groupby(['date', 'symbol'], as_index=False).sum()

        # cumsum by symbol
        # df['total_quantity'] = df.groupby('symbol').signed_quantity.cumsum()
        # df['total_cost_basis'] = df.groupby('symbol').cost_basis.cumsum()
        self.df_ord = df
        return self

    # process_orders
    def _process_dividends(self):
        df = self.df_div.copy()

        # convert types
        for field in ['amount', 'position', 'rate']:
            df[field] = pd.to_numeric(df[field])
        for field in ['paid_at', 'payable_date']:
            df[field] = pd.to_datetime(df[field])

        # add days
        df['date'] = df['paid_at'].apply(
            lambda x: pd.tslib.normalize_date(x))
        self.df_div = df
        return self

    def download_robinhood_data(self):
        self._login()

        self._download_dividends()._process_dividends()
        self._download_orders()._process_orders()

        self.df_div.to_hdf(self.datafile, 'dividends')
        self.df_ord.to_hdf(self.datafile, 'orders')
        return (self.df_div, self.df_ord)


if __name__ == "__main__":
    case = 'read'

    if case == 'download':
        rd = RobinhoodData('../data/data.h5')
        df_div, df_ord = rd.download_robinhood_data()

    if case == 'read':
        df_div = pd.read_hdf('../data/data.h5', 'dividends')
        df_ord = pd.read_hdf('../data/data.h5', 'orders')
