# TODO refactor to class
import pandas as pd
from Robinhood import Robinhood
from auth import user, password
from functools import reduce


# auxiliary for getting all orders
def fetch_json_by_url(rb_client, url):
    return rb_client.session.get(url).json()


# Setup and login
def login():
    rb = Robinhood()
    rb.login(username=user, password=password)
    return rb


def get_positions(rb):
    positions = rb.positions()
    positions = [x for x in positions['results']]
    df = pd.DataFrame(positions)
    df['symbol'] = df['instrument'].apply(rb.get_symbol_by_instrument)
    df['name'] = df['instrument'].apply(rb.get_name_by_instrument)
    df['last_trade_price'] = df['symbol'].apply(rb.last_trade_price)
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
    return df


def process_orders(df):
    for field in [
        'average_price', 'price', 'stop_price', 'quantity',
        'cumulative_quantity', 'fees'
    ]:
        df[field] = pd.to_numeric(df[field])
    for field in ['created_at', 'updated_at']:
        df[field] = pd.to_datetime(df[field])
    df.sort_values(by='created_at', inplace=True)
    # for field in df.columns:
    #     if df[field].dtype == 'O':
    #         print(field)
    #         df[field] = df[field].astype('|S')
    return df


def get_history_for_symbols(rb, symbols):
    dfs = []
    for s in symbols:
        res = rb.get_historical_quotes(s, 'week', '5year')
        df = pd.DataFrame(res['results'][0]['historicals'])
        df.columns =\
            ['_'.join([s, c]) if c != 'begins_at' else c for c in df.columns]
        dfs.append(df)
    df = reduce((lambda x, y: pd.merge(x, y, on='begins_at', how='left')), dfs)
    df['begins_at'] = pd.to_datetime(df['begins_at'])
    return df


def get_dividends(rb):
    dividends = rb.dividends()
    dividends = [x for x in dividends['results']]
    df = pd.DataFrame(dividends)
    df['symbol'] = df['instrument'].apply(rb.get_symbol_by_instrument)
    return df


def process_dividends(df):
    for field in ['position', 'amount', 'rate', 'withholding']:
        df[field] = pd.to_numeric(df[field])
    for field in ['record_date', 'payable_date', 'paid_at']:
        df[field] = pd.to_datetime(df[field])
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

        df_pos.to_hdf('../data/data.h5', 'positions')
        df_ord.to_hdf('../data/data.h5', 'orders')
        df_div.to_hdf('../data/data.h5', 'dividends')
