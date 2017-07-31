import pandas as pd
import numpy as np
from robinhood_api import RobinhoodAPI
from auth import user, password
from functools import reduce


class RobinhoodData:
    def __init__(self, datafile):
        self.datafile = '../data/data.h5'

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

        # calculate cost_basis at the moment of the order
        df['cost_basis'] = df['signed_quantity'] * df['average_price']

        # group by days
        df = df.groupby(['date', 'symbol'], as_index=False).sum()

        # cumsum by symbol
        df['total_quantity'] = df.groupby('symbol').signed_quantity.cumsum()
        df['total_cost_basis'] = df.groupby('symbol').cost_basis.cumsum()
        self.df_ord = df

    # process_orders
    def _process_dividends(self):
        df = self.df_div.copy()
        fields = [
            'paid_at',
            'amount',
            'symbol']
        df = df[fields]

        # convert types
        for field in ['amount']:
            df[field] = pd.to_numeric(df[field])
        for field in ['paid_at']:
            df[field] = pd.to_datetime(df[field])

        # add days
        df['date'] = df['paid_at'].apply(
            lambda x: pd.tslib.normalize_date(x))

        # cumsum by symbol
        df['total_amount'] = df.groupby('symbol').amount.cumsum()
        self.df_div = df

    def download_robinhood_data(self):
        self._login()

        self._download_dividends()._process_dividends()
        self._download_orders()._process_orders()

        self.df_div.to_hdf(self.datafile, 'dividends')
        self.df_ord.to_hdf(self.datafile, 'orders')
        return (self.df_div, self.df_ord)


# auxiliary for getting all orders
def fetch_json_by_url(rb_client, url):
    return rb_client.session.get(url).json()


# Setup and login
def login():
    rb = RobinhoodAPI()
    rb.login(username=user, password=password)
    return rb


def get_positions(rb):
    positions = rb.positions()
    positions = [x for x in positions['results']]
    df = pd.DataFrame(positions)
    df['symbol'] = df['instrument'].apply(rb.get_symbol_by_instrument)
    df['name'] = df['instrument'].apply(rb.get_name_by_instrument)
    # TODO last_trade_price is probably redundant
    df['last_trade_price'] = df['symbol'].apply(rb.last_trade_price)
    del df['account']
    return df


# TODO add position type
def process_positions(df):
    # change types
    for field in ['average_buy_price', 'intraday_average_buy_price',
                  'intraday_quantity', 'quantity']:
            df[field] = pd.to_numeric(df[field])
    for field in ['created_at', 'updated_at']:
        df[field] = pd.to_datetime(df[field])

    # process positions
    df['absGain'] =\
        (df['last_trade_price'] - df['average_buy_price']) *\
        df['quantity']
    df['value'] = df['last_trade_price'] * df['quantity']
    df['pctTotal'] = df['value'] / df['value'].sum()
    df['relGain'] = df['absGain'] / df['value']
    return df


def get_orders(rb):
    orders = []
    past_orders = rb.order_history()
    orders.extend(past_orders['results'])
    while past_orders['next']:
        next_url = past_orders['next']
        past_orders = fetch_json_by_url(rb, next_url)
        orders.extend(past_orders['results'])
    df = pd.DataFrame(orders)
    df['symbol'] = df['instrument'].apply(rb.get_symbol_by_instrument)

    # redundant?
    # df['last_trade_price'] = df['symbol'].apply(rb.last_trade_price)

    df.sort_values(by='created_at', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # privacy
    del df['account']
    del df['url']
    return df


def process_orders(df):
    # convert types
    for field in ['average_price', 'price', 'stop_price', 'quantity',
                  'cumulative_quantity', 'fees']:
        df[field] = pd.to_numeric(df[field])
    for field in ['created_at', 'updated_at']:
        df[field] = pd.to_datetime(df[field])

    # add days
    df['date'] = df['created_at'].apply(
        lambda x: pd.tslib.normalize_date(x))

    # quantity accounting for side of transaction
    df['signed_quantity'] = np.where(
        df.side == 'buy',
        df['cumulative_quantity'],
        -df['cumulative_quantity'])

    # calculate cost_basis at the moment of the order
    df['cost_basis'] = df['signed_quantity'] * df['average_price']

    # group by days
    df = df.groupby(['date', 'symbol'], as_index=False).sum()

    # cumsum by symbol
    df['total_quantity'] = df.groupby('symbol').signed_quantity.cumsum()
    df['total_cost_basis'] = df.groupby('symbol').cost_basis.cumsum()
    return df


def get_history_for_symbols(rb, symbols):
    dfs = []
    ncols = ['volume', 'open_price', 'close_price', 'high_price', 'low_price']
    for s in symbols:
        res = rb.get_historical_quotes(s, 'week', '5year')
        df = pd.DataFrame(res['results'][0]['historicals'])
        for c in ncols:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['symbol'] = s
        df['price'] = (df["close_price"] + df["open_price"]) / 2
        dfs.append(df)

    df = reduce((lambda x, y: pd.concat([x, y], axis=0)), dfs)
    df['begins_at'] = pd.to_datetime(df['begins_at'])
    return df


def get_dividends(rb):
    dividends = rb.dividends()
    dividends = [x for x in dividends['results']]
    df = pd.DataFrame(dividends)
    df['symbol'] = df['instrument'].apply(rb.get_symbol_by_instrument)
    del df['account']
    return df


def process_dividends(df):
    for field in ['position', 'amount', 'rate', 'withholding']:
        df[field] = pd.to_numeric(df[field])
    for field in ['record_date', 'payable_date', 'paid_at']:
        df[field] = pd.to_datetime(df[field])

    # add days
    df['date'] = df['paid_at'].apply(
        lambda x: pd.tslib.normalize_date(x))
    df.sort_values(by='date', inplace=True)

    # cumsum by symbol
    df['total_amount'] = df.groupby('symbol').amount.cumsum()
    return df


# merging orders and prices
def process_portfolio(df_ord, df_prc):
    df = pd.merge_asof(
        df_ord, df_prc[['begins_at']],
        left_on='created_at', right_on='begins_at')
    return df


# adding portfolio orders into a price panel
# currently hacky, done via a dummy dataframe
# not sure if there is a better way
def merge_market_with_orders(df_ord, pf):
    pf['total_quantity'] = 0
    pf['total_cost_basis'] = 0
    for key in pf.minor_axis[:-1]:
        df1 = pf.loc[:, :, key]
        df2 = df_ord[df_ord['symbol'] == key]
        df2.set_index('date', inplace=True)
        df = pd.merge(
            df1, df2[['total_quantity', 'total_cost_basis']],
            left_index=True, right_index=True, how='left')
        df.rename(columns={
            'total_quantity_y': 'total_quantity',
            'total_cost_basis_y': 'total_cost_basis'}, inplace=True)
        df.drop('total_quantity_x', axis=1, inplace=True)
        df.drop('total_cost_basis_x', axis=1, inplace=True)
        # now propagate values from last observed and then fill rest with zeros
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        pf.ix[:, :, key] = df
    # portfolio calculations
    pf['current_price'] = pf['total_quantity'] * pf['Close']
    pf['current_ratio'] =\
        (pf['current_price'].T / pf['current_price'].sum(axis=1)).T
    return pf


def merge_market_with_dividends(df_div, pf):
    pf['total_dividends'] = 0
    for key in pf.minor_axis[:-1]:
        df1 = pf.loc[:, :, key]
        df2 = df_div[df_div['symbol'] == key]
        df2.set_index('date', inplace=True)
        df = pd.merge(
            df1, df2[['total_amount']],
            left_index=True, right_index=True, how='left')
        df.drop('total_dividends', axis=1, inplace=True)
        df.rename(columns={'total_amount': 'total_dividends'}, inplace=True)
        # now propagate values from last observed and then fill rest with zeros
        df.fillna(0, inplace=True)
        pf.ix[:, :, key] = df
    return pf


if __name__ == "__main__":
    case = 'read'

    if case == 'download':
        rd = RobinhoodData('../data/data.h5')
        df_div, df_ord = rd.download_robinhood_data()

    if case == 'read':
        df_div = pd.read_hdf('../data/data.h5', 'dividends')
        df_ord = pd.read_hdf('../data/data.h5', 'orders')
        pf = pd.read_hdf('../data/data.h5', 'panel')



    # if case == 'download':
    #     rb = login()
    #     df_pos = process_positions(get_positions(rb))
    #     df_ord = process_orders(get_orders(rb))
    #     df_div = process_dividends(get_dividends(rb))
    #     df_prc = get_history_for_symbols(rb, df_ord['symbol'].unique())

    #     df_pos.to_hdf('../data/data.h5', 'positions')
    #     df_ord.to_hdf('../data/data.h5', 'orders')
    #     df_div.to_hdf('../data/data.h5', 'dividends')
    #     df_prc.to_hdf('../data/data.h5', 'prices')




    # if case == 'update':
    #     df_div = process_dividends(pd.read_hdf('../data/data.h5', 'dividends'))
    #     df_pos = process_positions(pd.read_hdf('../data/data.h5', 'positions'))
    #     df_ord = process_orders(pd.read_hdf('../data/data.h5', 'orders'))
    #     df_prc = pd.read_hdf('../data/data.h5', 'prices')

    #     # df_ptf = process_portfolio(df_ord, df_prc)

    #     df_pos.to_hdf('../data/data.h5', 'positions')
    #     df_ord.to_hdf('../data/data.h5', 'orders')
    #     df_div.to_hdf('../data/data.h5', 'dividends')

    # # working on portfolio
    # pf = merge_market_with_orders(df_ord, pf)
    # pf = merge_market_with_dividends(df_div, pf)
