import pandas as pd
import numpy as np
from robinhood_api import RobinhoodAPI
from market_data import MarketData
from auth import user, password
from functools import reduce


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
    return df


# merging orders and prices
def process_portfolio(df_ord, df_prc):
    df = pd.merge_asof(
        df_ord, df_prc[['begins_at']],
        left_on='created_at', right_on='begins_at')
    return df


if __name__ == "__main__":
    case = 'read'

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

        # mb = MarketData()
        # pf = mb.get_data(
        #     df_ord.symbol.unique(),
        #     df_ord.date.min().strftime("%Y%m%d"),
        #     df_ord.date.max().strftime("%Y%m%d"))
        # pf.to_hdf('../data/data.h5', 'panel')

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

    # working on portfolio
    pf = pd.read_hdf('../data/data.h5', 'panel')

    # adding portfolio orders into a price panel
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

    pf['current_price'] = pf['total_quantity'] * pf['Close']
    pf['current_ratio'] =\
        (pf['current_price'].T / pf['current_price'].sum(axis=1)).T
