# TODO refactor to class
from Robinhood import Robinhood
import pandas as pd
from auth import user, password


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
    df['balance'] =\
        (df['last_trade_price'] - df['average_buy_price']) *\
        df['quantity']
    df['subtotal'] = df['last_trade_price'] * df['quantity']
    df['ratio'] = df['subtotal'] / df['subtotal'].sum()
    df['return'] = df['balance'] / df['subtotal']
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
    return df


def process_orders(df):
    for field in ['average_price', 'price', 'stop_price', 'quantity']:
        df[field] = pd.to_numeric(df[field])
    for field in ['created_at', 'updated_at']:
        df[field] = pd.to_datetime(df[field])
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

        df_pos.to_hdf('../data/data.h5', 'positions')
        df_ord.to_hdf('../data/data.h5', 'orders')
        df_div.to_hdf('../data/data.h5', 'dividends')

    if case == 'read':
        df_div = pd.read_hdf('../data/data.h5', 'dividends')
        df_pos = pd.read_hdf('../data/data.h5', 'positions')
        df_ord = pd.read_hdf('../data/data.h5', 'orders')

    if case == 'update':
        df_div = process_dividends(pd.read_hdf('../data/data.h5', 'dividends'))
        df_pos = process_positions(pd.read_hdf('../data/data.h5', 'positions'))
        df_ord = process_orders(pd.read_hdf('../data/data.h5', 'orders'))

        df_pos.to_hdf('../data/data.h5', 'positions')
        df_ord.to_hdf('../data/data.h5', 'orders')
        df_div.to_hdf('../data/data.h5', 'dividends')
