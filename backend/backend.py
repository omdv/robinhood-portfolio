import pandas as pd
from collections import OrderedDict
from pickle import dump, load
from backend.portfolio_model import PortfolioModels
from backend.robinhood_data import RobinhoodData
from backend.market_data import MarketData


class BackendClass(object):
    """
    Backend wrapper class, provides wrappers to donwload robinhood and market
    data, provides access to portfolio models. Mostly deals with UI logic.
    --------
    datafile: path to hdf datafile
    """
    def __init__(self, datafile, userfile):
        self.datafile = datafile
        self.userfile = userfile
        self.df_returns = None
        self.daily_returns = None
        self.user = {}
        self.portfolio = {}
        self.trades = {}
        self.stock = {}
        self.markowitz = {}
        self._date_fmt = '{:%d-%m-%Y}'

        # read dataframe
        self._df_ord = pd.read_hdf(datafile, 'orders')
        self._df_div = pd.read_hdf(datafile, 'dividends')
        self._market = pd.read_hdf(datafile, 'market')

        # handle user dictionary
        self._init_user_dict()
        self._validate_user_dict()

    def _init_user_dict(self):
        # try to load the user dict or re-initiate if all fails
        try:
            with open(self.userfile, 'rb') as f:
                self.user = load(f)
            self.user["today"] = pd.Timestamp("today")
        except:
            # check the dates for robinhood data
            self.user['rb_dates'] = [
                self._df_ord.date.min(),
                max(self._df_ord.date.max(), self._df_div.date.max())]

            '''
            # check the dates for market data
            self.user['mkt_dates'] = [
                self._market.major_axis.min(),
                self._market.major_axis.max()]
            '''
            # use the fresh start for market when pickle is deleted
            self.update_market_data(fresh_start=True)

            # get today date
            self.user['today'] = pd.Timestamp("today")
            # pickle the dictionary
            self._pickle_user_dict()
        return self

    def _validate_user_dict(self):
        # check the consistency of the dates
        if (
            (self.user['mkt_dates'][0] > self.user['rb_dates'][0]) or
            (self.user['mkt_dates'][1] < self.user['rb_dates'][1])
        ):
            print('Market data is not consistent with Robinhood data')
            self.update_market_data(fresh_start=True)
        # check if dates match the actual data
        elif (
            (self.user['mkt_dates'][0] != self._market.major_axis.min()) or
            (self.user['mkt_dates'][1] != self._market.major_axis.max())
        ):
            print('Dates are not consistent with dataset')
            self.update_market_data(fresh_start=True)
        return self

    def _pickle_user_dict(self):
        # dump dates and user config
        with open(self.userfile, 'wb') as f:
            dump(self.user, f)

    def update_market_data(self, **keyword_parameters):
        """
        if 'fresh_start' is passed than use the rb_dates from
        user dict to download the entire history from scratch, otherwise
        download only new dates in addition to the existing set
        """
        md = MarketData(self.datafile)

        # check if symbols match
        s1 = list(self._df_ord.symbol.unique())
        s1.sort()
        s2 = list(self._market.minor_axis)
        s2.sort()

        if ('fresh_start' in keyword_parameters) or (s1 != s2):
            min_date = self.user['rb_dates'][0]
            max_date = pd.Timestamp("today")
            self._market = md.download_save_market_data(
                self._df_ord.symbol.unique(), min_date, max_date)
            self.user['mkt_dates'] = [min_date, max_date]
        else:
            min_date = self.user['mkt_dates'][1]
            max_date = pd.Timestamp("today")
            self._market = md.download_save_market_data(
                self._df_ord.symbol.unique(), min_date, max_date,
                update_existing=True)
            self.user['mkt_dates'] = [self.user['mkt_dates'][0], max_date]

        self._pickle_user_dict()
        self.calculate_all()
        return self

    def update_robinhood_data(self, user, password):
        rd = RobinhoodData(self.datafile)
        self._df_div, self._df_ord, _, _ =\
            rd.download_robinhood_data(user, password)
        self.user['rb_dates'] = [
            self._df_ord.date.min(), pd.Timestamp("today")]

        '''
        destroy RobinhoodData instance after download
        to protect user information and delete token
        '''
        rd = 0

        self._pickle_user_dict()
        self.update_market_data()
        self.calculate_all()
        return self

    def _get_daily_portfolio_panel(self):
        """
        Generate a panelframe with daily portfolio changes
        """
        self._ptfm = PortfolioModels(self.datafile)
        self._panel = self._ptfm.daily_portfolio_changes().panelframe
        return self

    def _get_latest_portfolio_snapshot(self):
        # use only the last row
        df = self._panel.iloc[:, -1, :-1]

        columns_to_names = OrderedDict([
            ('cum_size', ['Shares', '{:,.0f}']),
            ('current_weight', ['Portfolio weight', '{:.2f}%']),
            ('cum_cost_basis', ['Current cost basis', '{:,.2f}']),
            ('cum_value_close', ['Current value', '{:,.2f}']),
            ('cum_realized_gain', ['Realized gain', '{:,.2f}']),
            ('cum_dividends', ['Dividends', '{:,.2f}']),
            ('cum_unrealized_gain', ['Unrealized gain', '{:,.2f}']),
            ('cum_total_return', ['Total return', '{:,.2f}']),
            ('current_return_rate', ['Total return rate', '{:,.2f}%'])
        ])

        # convert ratios to percent
        df['current_weight'] = df['current_weight'] * 100

        # add total row
        df = df.copy()  # avoid chained assignment warning
        df.loc['Portfolio', :] = df.sum(axis=0)
        df.loc['Portfolio', 'current_return_rate'] =\
            df.loc['Portfolio', 'cum_total_return'] /\
            df.loc['Portfolio', 'cum_cost_basis'] * 100

        # re-order
        df = df[list(columns_to_names.keys())]

        # format
        df = df.apply(
            lambda x: x.map(columns_to_names[x.name][1].format), axis=0)

        # rename for HTML
        df.columns =\
            df.columns.to_series().apply(lambda x: columns_to_names[x][0])

        # apply styles
        self.portfolio['returns'] = df.style.\
            set_table_attributes('border=1 class="dataframe"').\
            render()

        return self

    def _get_sell_orders(self):
        """
        Get three best/worst closed positions by realized gains
        """
        df = pd.read_hdf(self.datafile, 'closed')
        df1 = df.nlargest(min(3, df.shape[0]), 'realized_gains')
        df2 = df.nsmallest(min(3, df.shape[0]), 'realized_gains')
        df = pd.concat([df1, df2]).sort_values(by='realized_gains')
        df['buy_price'] = df['current_cost_basis'] / df['signed_size']

        # create unique index to avoid issues with styler
        df.reset_index(inplace=True)

        columns_to_names = OrderedDict([
            ('date', 'Date'),
            ('symbol', 'Security'),
            ('current_size', 'Shares'),
            ('buy_price', 'Average buy price'),
            ('average_price', 'Average sell price'),
            ('realized_gains', 'Realized gain')
        ])

        self.trades['closed'] = pd.DataFrame()
        for k in columns_to_names:
            self.trades['closed'][columns_to_names[k]] = df[k]

        # apply styles
        self.trades['closed'] = self.trades['closed'].style.\
            set_table_attributes('border=1 class="dataframe"').\
            bar(subset=['Realized gain'], align='mid',
                color=['#fc8d59', '#91bfdb']).\
            set_table_styles([
                {'selector': '.row_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}]).\
            format({
                'Date': self._date_fmt,
                'Shares': '{:,.0f}',
                'Average buy price': '{:,.2f}',
                'Average sell price': '{:,.2f}',
                'Realized gain': "{:+,.2f}"
            }).\
            render()
        return self

    def _get_buy_orders(self):
        """
        Get three best/worst open positions by unrealized gains
        """
        market_prices = self._panel['Close'].iloc[-1]

        df = pd.read_hdf(self.datafile, 'open')
        df['current_price'] =\
            df.apply(lambda x: market_prices[x.symbol], axis=1)
        df['unrealized_gains'] =\
            (df['current_price'] - df['average_price']) * df['final_size']

        df1 = df.nlargest(min(3, df.shape[0]), 'unrealized_gains').copy()
        df2 = df.nsmallest(min(3, df.shape[0]), 'unrealized_gains').copy()
        df = pd.concat([df1, df2]).sort_values(by='unrealized_gains')

        # create unique index to avoid issues with styler
        df.reset_index(inplace=True)

        columns_to_names = OrderedDict([
            ('date', 'Date'),
            ('symbol', 'Security'),
            ('final_size', 'Shares'),
            ('average_price', 'Average buy price'),
            ('current_price', 'Current market price'),
            ('unrealized_gains', 'Unrealized gain')
        ])

        self.trades['open'] = pd.DataFrame()
        for k in columns_to_names:
            self.trades['open'][columns_to_names[k]] = df[k]

        # apply styles
        self.trades['open'] = self.trades['open'].style.\
            set_table_attributes('border=1 class="dataframe"').\
            bar(subset=['Unrealized gain'], align='mid',
                color=['#fc8d59', '#91bfdb']).\
            set_table_styles([
                {'selector': '.row_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}]).\
            format({
                'Date': self._date_fmt,
                'Shares': '{:,.0f}',
                'Average buy price': '{:,.2f}',
                'Current market price': '{:,.2f}',
                'Unrealized gain': "{:+,.2f}"
            }).\
            render()
        return self

    def _get_all_orders(self):
        cl = pd.read_hdf(self.datafile, 'closed')
        op = pd.read_hdf(self.datafile, 'open')
        mkt = self._panel['Close'].iloc[-1]

        cl['average_buy_price'] = cl['current_cost_basis'] / cl['signed_size']
        cl.rename(columns={'average_price': 'average_sell_price'},
                  inplace=True)

        op['current_price'] =\
            op.apply(lambda x: mkt[x.symbol], axis=1)
        op['unrealized_gains'] =\
            (op['current_price'] - op['average_price']) * op['final_size']
        op.rename(columns={'average_price': 'average_buy_price'},
                  inplace=True)

        ord = pd.concat([cl, op]).sort_values(by='date')
        columns_to_names = OrderedDict([
            ('date', 'Date'),
            ('symbol', 'Security'),
            ('signed_size', 'Order size'),
            ('final_size', 'Current size'),
            ('current_price', 'Current market price'),
            ('average_buy_price', 'Average buy price'),
            ('average_sell_price', 'Average sell price'),
            ('realized_gains', 'Realized gain'),
            ('unrealized_gains', 'Unrealized gain')
        ])

        self.trades['all'] = pd.DataFrame()
        for k in columns_to_names:
            self.trades['all'][columns_to_names[k]] = ord[k]

        self.trades['all'] = self.trades['all'].style.\
            set_table_attributes(
                'border=1 class="dataframe orders"' +
                'style="display:none"').\
            set_table_styles([
                {'selector': '.row_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}]).\
            format({
                'Date': self._date_fmt,
                'Execution size': '{:,.0f}',
                'Final size': '{:,.0f}',
                'Average buy price': '{:,.2f}',
                'Average sell price': '{:,.2f}',
                'Current market price': '{:,.2f}',
                'Realized gain': '{:,.2f}',
                'Unrealized gain': '{:,.2f}'
            }).\
            render()

        self.trades['total_orders'] = ord.shape[0]
        self.trades['open_orders'] = op.shape[0]
        self.trades['closed_orders'] = cl.shape[0]
        self.trades['fees'] = round(ord['fees'].sum(), 2)

        return self

    def _format_portfolio_stats_series(self, df, horizontal=True):
        pct_val = [
            "Total return",
            "Market return",
            "Annual return",
            "Annual volatility",
            "Max drawdown",
            "Daily value at risk",
            "Alpha"
        ]
        for col in pct_val:
            try:
                df[col] = df[col] * 100
            except:
                None

        # format values
        df = df.to_frame().apply(
            lambda x:
            '{:.2f}'.format(x[0]) if x.name not in pct_val
            else '{:.2f}%'.format(x[0]), axis=1)

        # apply styles
        if horizontal:
            df = df.to_frame().transpose().style
        else:
            df = df.to_frame().style

        df = df.set_table_attributes('border=1 class="dataframe"')

        if horizontal:
            df = df.set_table_styles([
                {'selector': '.row_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}])
        else:
            df = df.set_table_styles([
                {'selector': '.col_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}])

        return df.render()

    def _format_stock_stats_frame(self, df):

        # apply styles
        pct_val = [
            "Annual return",
            "Annual volatility",
            "Max drawdown",
            "Daily value at risk",
            "Alpha"
        ]

        for col in pct_val:
            df[col] = df[col] * 100

        # parse percentage values to strings
        for i in df.columns:
            df[i] = df[i].apply(
                lambda x: '{:.2f}'.format(x) if i not in pct_val
                else '{:.2f}%'.format(x))

        def color_values(val):
            if val[0] == '-':
                return 'background-color: {}'.format('#fc8d59')
            else:
                return 'background-color: {}'.format('#91bfdb')

        # apply styles
        res = df.style.\
            set_table_attributes('border=1 class="dataframe"').\
            applymap(lambda x: color_values(x),
                     subset=['Alpha', 'Annual return', 'Sharpe ratio']).\
            render()
        return res

    def _get_portfolio_stats(self):
        """
        Get actual portfolio stats
        """
        self.portfolio['daily'] = self._panel['cum_total_return'].sum(axis=1)

        pf_stats = self._ptfm.actual_portfolio_stats()
        self.portfolio['stats'] = self._format_portfolio_stats_series(pf_stats)

        self.portfolio['total_return'] =\
            '{:.2f}%'.format(pf_stats['Total return'])
        self.portfolio['annual_return'] =\
            '{:.2f}%'.format(pf_stats['Annual return'])
        self.portfolio['market_return'] =\
            '{:.2f}%'.format(pf_stats['Market return'])

        return self

    def _get_stock_stats(self):
        df = self._ptfm.stock_risk_analysis(False)
        self.stock['risk'] = self._format_stock_stats_frame(df)
        self.stock['corr'], _ = self._ptfm.stock_correlation_matrix()
        return self

    def _get_markowitz(self):
        mrk = self._ptfm.markowitz_portfolios()
        for c in mrk:
            c['stats'] = self._format_portfolio_stats_series(c['stats'], False)
            c['weights'] = c['weights'].apply(
                lambda x: '{:.2f}%'.format(x*100))
            c['weights'] = c['weights'].to_frame().style.\
                set_table_attributes('border=1 class="dataframe"').\
                set_table_styles([
                    {'selector': '.col_heading',
                        'props': [('display', 'none')]},
                    {'selector': '.blank.level0',
                        'props': [('display', 'none')]}]).\
                render()

        self.markowitz = mrk
        return self

    def calculate_all(self):
        """
        Run all calculations and save values to internal class props
        --------
        Returns:
        """
        self._get_daily_portfolio_panel()
        self._get_latest_portfolio_snapshot()
        self._get_portfolio_stats()
        self._get_stock_stats()
        self._get_buy_orders()
        self._get_sell_orders()
        self._get_all_orders()
        self._get_markowitz()
        return self


if __name__ == '__main__':
    bc = BackendClass('../data/data.h5')
    bc = bc.calculate_all()
