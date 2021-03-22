"""
Download and prepare robinhood data
"""
import pandas as pd
import numpy as np


class RobinhoodData:
    """
    Wrapper to download orders and dividends from Robinhood accounts
    Downloads two dataframes and saves to datafolder
    ----------
    Parameters:
    datafolder : location of data
    client : connected pyrh client
    """
    def __init__(self, datafolder, client):
        self.datafolder = datafolder
        self.client = client

    def _get_symbol_from_instrument_url(self, url):
        return self._fetch_json_by_url(url)['symbol']

    # private method for getting all orders
    def _fetch_json_by_url(self, url):
        return self.client.session.get(url).json()

    # deleting sensitive or redundant fields
    def _delete_sensitive_fields(self, df):
        for col in ['account', 'url', 'id', 'instrument']:
            if col in df:
                del df[col]
        return df

    # download orders and fields requiring RB client
    def _download_orders(self):
        print("Downloading orders from Robinhood")
        orders = []
        past_orders = self.client.order_history()
        orders.extend(past_orders['results'])
        while past_orders['next']:
            next_url = past_orders['next']
            past_orders = self._fetch_json_by_url(next_url)
            orders.extend(past_orders['results'])
        df = pd.DataFrame(orders)
        df['symbol'] = df['instrument'].apply(
            self._get_symbol_from_instrument_url)
        df.sort_values(by='created_at', inplace=True)
        df.reset_index(inplace=True, drop=True)
        df_ord = self._delete_sensitive_fields(df)
        return df_ord

    # download dividends and fields requiring RB client
    def _download_dividends(self):
        print("Downloading dividends from Robinhood")
        dividends = self.client.dividends()
        dividends = [x for x in dividends['results']]
        df = pd.DataFrame(dividends)
        if df.shape[0] > 0:
            df['symbol'] = df['instrument'].apply(
                self._get_symbol_from_instrument_url)
            df.sort_values(by='paid_at', inplace=True)
            df.reset_index(inplace=True, drop=True)
            df_div = self._delete_sensitive_fields(df)
        else:
            df_div = pd.DataFrame(columns=['symbol', 'amount', 'position',
                                           'rate', 'paid_at', 'payable_date'])
        return df_div

    # process orders
    def _process_orders(self, df_ord):
        # assign to df and reduce the number of fields
        df = df_ord.copy()
        fields = [
            'created_at',
            'average_price', 'cumulative_quantity', 'fees',
            'symbol', 'side']
        df = df[fields]

        # convert types
        for field in ['average_price', 'cumulative_quantity', 'fees']:
            df[field] = pd.to_numeric(df[field])
        for field in ['created_at']:
            df[field] = pd.to_datetime(df[field])

        # normalize dates
        idx = pd.Index(df['created_at']).normalize()
        df['date'] = idx

        # rename columns for consistency
        df.rename(columns={
            'cumulative_quantity': 'current_size'
        }, inplace=True)

        # quantity accounting for side of transaction for cumsum later
        df['signed_size'] = np.where(
            df.side == 'buy',
            df['current_size'],
            -df['current_size'])
        df['signed_size'] = df['signed_size'].astype(np.int64)

        # initialize columns
        df['realized_gains'] = 0.
        df['current_cost_basis'] = 0.

        return df

    # process_orders
    def _process_dividends(self, df_div):
        df = df_div.copy()

        # convert types
        for field in ['amount', 'position', 'rate']:
            df[field] = pd.to_numeric(df[field])
        for field in ['paid_at', 'payable_date']:
            df[field] = pd.to_datetime(df[field])

        # normalize dates
        idx = pd.Index(df['paid_at']).normalize()
        df['date'] = idx
        return df

    def _generate_positions(self, df_ord):
        """
        Process orders dataframe and generate open and closed positions.
        For all open positions close those which were later sold, so that
        the cost_basis for open can be calculated correctly. For closed
        positions calculate the cost_basis based on the closed open positions.
        Note: the olders open positions are first to be closed. The logic here
        is to reduce the tax exposure.
        -----
        Parameters:
        - Pre-processed df_ord
        Return:
        - Two dataframes with open and closed positions correspondingly
        """
        # prepare dataframe for open and closed positions
        df_open = df_ord[df_ord.side == 'buy'].copy()
        df_closed = df_ord[df_ord.side == 'sell'].copy()

        # create a new column for today's position size
        df_open['final_size'] = df_open['current_size']
        df_closed['final_size'] = df_closed['current_size']

        # main loop
        for i_closed, row_closed in df_closed.iterrows():
            sell_size = row_closed.final_size
            sell_cost_basis = 0
            for i_open, _ in df_open[
                    (df_open.symbol == row_closed.symbol) &
                    (df_open.date < row_closed.date)].iterrows():

                new_sell_size = sell_size - df_open.loc[i_open, 'final_size']
                new_sell_size = 0 if new_sell_size < 0 else new_sell_size

                new_open_size = df_open.loc[i_open, 'final_size'] - sell_size
                new_open_size = new_open_size if new_open_size > 0 else 0

                # updating open positions
                df_open.loc[i_open, 'final_size'] = new_open_size

                # updating closed positions
                df_closed.loc[i_closed, 'final_size'] = new_sell_size
                sold_size = sell_size - new_sell_size
                sell_cost_basis +=\
                    df_open.loc[i_open, 'average_price'] * sold_size
                sell_size = new_sell_size

            # assign a cost_basis to the closed position
            df_closed.loc[i_closed, 'current_cost_basis'] = -sell_cost_basis

        # calculate cost_basis for open positions
        df_open['current_cost_basis'] =\
            df_open['current_size'] * df_open['average_price']
        df_open['final_cost_basis'] =\
            df_open['final_size'] * df_open['average_price']

        # calculate capital gains for closed positions
        if df_closed.shape[0] != 0:
            df_closed['realized_gains'] =\
                df_closed['current_size'] * df_closed['average_price'] +\
                df_closed['current_cost_basis']
            df_closed['final_cost_basis'] = 0

        return df_open, df_closed

    def download(self, orders=None, dividends=None):
        """
        Download and parse method
        """
        if dividends is None:
            dividends = self._download_dividends()
        df_div = self._process_dividends(dividends)
        df_div.to_pickle(self.datafolder + "dividends.pkl")

        if orders is None:
            orders = self._download_orders()
        df_ord = self._process_orders(orders)
        df_ord.to_pickle(self.datafolder + "orders.pkl")

        df_open, df_closed = self._generate_positions(df_ord)
        df_open.to_pickle(self.datafolder + "open.pkl")
        df_closed.to_pickle(self.datafolder + "closed.pkl")

        return df_div, df_ord, df_open, df_closed

    def demo_orders(self):
        """
        Generate demo orders dataframe for testing
        """
        orders = pd.DataFrame(index=range(11))
        orders['created_at'] = pd.Timestamp('2018-01-02', tz='UTC')
        orders['date'] = pd.Timestamp('2018-01-02', tz='UTC')
        orders['symbol'] = ['MSFT', 'AAPL', 'CVX', 'XOM', 'BND', 'CAT', 'BA', 'TIF', 'BAC', 'JPM', 'MSFT']
        orders['current_size'] = 100
        orders['signed_size'] = 100
        orders['average_price'] = 100.0
        orders['fees'] = 0
        orders['cumulative_quantity'] = 100
        orders['side'] = 'buy'
        orders.to_pickle(self.datafolder + 'orders.pkl')

        # one sell order
        orders.loc[10, 'side'] = 'sell'
        orders.loc[10, 'average_price'] = 120.
        orders.loc[10, 'created_at'] = pd.Timestamp('2020-01-03', tz='UTC')
        orders.loc[10, 'date'] = pd.Timestamp('2020-01-03', tz='UTC')
        orders.loc[10, 'cumulative_quantity'] = 20
        orders.loc[10, 'signed_size'] = -20
        orders.loc[10, 'current_size'] = 20
        orders.loc[10, 'fees'] = 0
        return orders

    def demo_dividends(self):
        """
        Generate demo dividends dataframe for testing
        """
        dividends = pd.DataFrame(index=range(10))
        dividends['symbol'] = ['MSFT', 'AAPL', 'CVX', 'XOM', 'BND', 'CAT', 'BA', 'TIF', 'BAC', 'JPM']
        dividends['position'] = 100
        dividends['amount'] = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        dividends['rate'] = dividends['amount'] / dividends['position']
        dividends['paid_at'] = pd.Timestamp('2019-01-15', tz='UTC')
        dividends['payable_date'] = pd.Timestamp('2019-01-02', tz='UTC')
        return dividends


if __name__ == "__main__":
    pass
