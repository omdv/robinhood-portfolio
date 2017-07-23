import pandas as pd
import numpy as np
from robinhood_api import RobinhoodAPI
from market_data import MarketData
from auth import user, password
from functools import reduce


# calculating portfolio performance
class PortfolioPerformance():
    def __init__(self, risk_free):
        self.risk_free = risk_free
        return None

    # beta for a provided panel, index value should be last column
    def beta_by_covar(self, pf):
        betas = dict()
        # dates = pd.to_datetime([start_date, end_date])
        pf.ix['Daily_change'] = (pf.ix['Close'] - pf.ix['Close'].shift(1))\
            / pf.ix['Close'].shift(1) * 100
        covar = np.cov(pf.ix['Daily_change'][1:], rowvar=False, ddof=0)
        variance = np.var(pf.ix['Daily_change'])['market']
        for i, j in enumerate(pf.ix['Daily_change'].columns):
            betas[j] = covar[-1, i] / variance
        return betas

    # beta for a provided panel based on simple return
    def beta_by_return(self, pf):
        return None

    # alpha by capm model
    def alpha_by_capm():
        return None


class RobinhoodData:
    def __init__(self):
        return None

    def login(self):
        self.client = RobinhoodAPI()
        self.client.login(username=user, password=password)
        return self


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

    # get asset type
    # TODO plug for testing
    df.loc[:, 'asset_type'] = 'ETF'
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
    df['last_trade_price'] = df['symbol'].apply(rb.last_trade_price)
    df.sort_values(by='created_at', inplace=True)
    df.reset_index(inplace=True, drop=True)

    # privacy
    del df['account']
    return df


def process_orders(df):
    # convert types
    for field in ['average_price', 'price', 'stop_price', 'quantity',
                  'cumulative_quantity', 'fees']:
        df[field] = pd.to_numeric(df[field])
    for field in ['created_at', 'updated_at']:
        df[field] = pd.to_datetime(df[field])

    # quantity accounting for side of transaction
    df['signed_quantity'] = np.where(
        df.side == 'buy',
        df['cumulative_quantity'],
        -df['cumulative_quantity'])

    # calculate cost_basis at the moment of the order
    df['cost_basis'] = df['signed_quantity'] * df['average_price']

    # cumsum by symbol
    df['cum_quantity_symbol'] = df.groupby('symbol').signed_quantity.cumsum()
    return df


# def get_history_for_symbols_1(rb, symbols):
#     dfs = []
#     for s in symbols:
#         res = rb.get_historical_quotes(s, 'week', '5year')
#         df = pd.DataFrame(res['results'][0]['historicals'])

#         df = df[['begins_at', 'volume', 'open_price', 'close_price']]
#         for c in ['volume', 'open_price', 'close_price']:
#             df[c] = pd.to_numeric(df[c], errors='coerce')

#         df.columns =\
#             ['_'.join([s, c]) if c != 'begins_at' else c for c in df.columns]
#         df[s] = (df[s + "_close_price"] + df[s + "_open_price"]) / 2
#         dfs.append(df)

#     df = reduce((lambda x, y: pd.merge(x, y, on='begins_at', how='left')), dfs)
#     df['begins_at'] = pd.to_datetime(df['begins_at'])
#     return df


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
    return df


# merging orders and prices
def process_portfolio(df_ord, df_prc):
    df = pd.merge_asof(
        df_ord, df_prc[['begins_at']],
        left_on='created_at', right_on='begins_at')
    return df


if __name__ == "__main__":
    case = 'update'

    if case == 'download':
        rb = login()
        df_pos = process_positions(get_positions(rb))
        df_ord = process_orders(get_orders(rb))
        df_div = process_dividends(get_dividends(rb))
        df_prc = get_history_for_symbols(rb, df_ord['symbol'].unique())

        df_pos.to_hdf('../data/data.h5', 'positions')
        df_ord.to_hdf('../data/data.h5', 'orders')
        df_div.to_hdf('../data/data.h5', 'dividends')
        df_prc.to_hdf('../data/data.h5', 'prices')

    if case == 'read':
        df_div = pd.read_hdf('../data/data.h5', 'dividends')
        df_pos = pd.read_hdf('../data/data.h5', 'positions')
        df_ord = pd.read_hdf('../data/data.h5', 'orders')
        df_prc = pd.read_hdf('../data/data.h5', 'prices')

    if case == 'update':
        df_div = process_dividends(pd.read_hdf('../data/data.h5', 'dividends'))
        df_pos = process_positions(pd.read_hdf('../data/data.h5', 'positions'))
        df_ord = process_orders(pd.read_hdf('../data/data.h5', 'orders'))
        df_prc = pd.read_hdf('../data/data.h5', 'prices')

        # df_ptf = process_portfolio(df_ord, df_prc)

        df_pos.to_hdf('../data/data.h5', 'positions')
        df_ord.to_hdf('../data/data.h5', 'orders')
        df_div.to_hdf('../data/data.h5', 'dividends')

    # read panel
    pf = pd.read_hdf('../data/data.h5', 'panel')
