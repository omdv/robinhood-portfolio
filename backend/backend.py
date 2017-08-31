import pandas as pd
from datetime import datetime
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
    def __init__(self, datafile):
        self.datafile = datafile
        self.df_returns = None
        self.daily_returns = None

    def download_save_data(self):
        rd = RobinhoodData(self.datafile)
        self._df_ord, self._df_div = rd.download_robinhood_data()

        md = MarketData(self.datafile)
        md.download_save_market_data(
            self.df_ord.symbol.unique(),
            self.df_ord.date.min().strftime('%Y%m%d'),
            pd.Timestamp("today").strftime('%Y%m%d'))
        return self

    def _get_daily_portfolio_panel(self):
        """
        Generate a panelframe with daily portfolio changes
        """
        self._ptfm = PortfolioModels(self.datafile)
        self._df_ord = pd.read_hdf(self.datafile, 'orders')
        self._panel = self._ptfm.daily_portfolio_changes().panelframe
        return self

    def _get_daily_total_returns(self):
        """
        Generate a series of cumulative daily  returns for a plot
        """
        self.daily_returns = self._panel['cum_total_return'].sum(axis=1)
        return self

    def _get_latest_portfolio_snapshot(self):
        # use only the last row
        df = self._panel.iloc[:, -1, :-1]
        columns = [
            'cum_size',
            'current_weight',
            'cum_cost_basis',
            'cum_value',
            'cum_unrealized_gain',
            'cum_realized_gain',
            'cum_dividends',
            'cum_total_return',
            'current_return_rate'
        ]
        df = df[columns]
        # convert ratios to percent
        df['current_weight'] = df['current_weight'] * 100

        # add total row
        df = df.copy()  # avoid chained assignment warning
        df.loc['Portfolio', :] = df.sum(axis=0)
        df.loc['Portfolio', 'current_return_rate'] =\
            df.loc['Portfolio', 'cum_total_return'] /\
            df.loc['Portfolio', 'cum_cost_basis'] * 100

        # how to deal with closed positions
        # df['current_return_rate'].fillna(method='ffill', inplace=True)
        # df.loc[df['cum_size'] == 0, 'current_return_rate'] = 0

        # # fix market return rate
        # df.loc['market', :] = ''
        # market_return = (
        #     self._panel['Close', :, 'market'][-1] -
        #     self._panel['Close', :, 'market'][0])
        # df.loc['market', 'current_return_rate'] =\
        #     market_return / self._panel['Close', :, 'market'][0] * 100
        # df.rename(index={'market': 'S&P 500'}, inplace=True)

        # rename for HTML
        df.rename(columns={
            'cum_size': 'Shares',
            'cum_cost_basis': 'Current cost basis, $',
            'cum_dividends': 'Dividends, $',
            'cum_value': 'Current value, $',
            'current_weight': 'Portfolio weight, %',
            'cum_unrealized_gain': 'Unrealized gain, $',
            'cum_realized_gain': 'Realized gain, $',
            'cum_total_return': 'Total return, $',
            'current_return_rate': 'Return rate, %'
        }, inplace=True)

        # apply styles
        # self.df_returns = df.style.\
        #     set_table_attributes('border=1 class="dataframe"').\
        #     format({
        #         'Shares': '{:,.0f}',
        #         'Current cost basis, $': '{:,.2f}',
        #         'Dividends, $': '{:,.2f}',
        #         'Current value, $': '{:,.2f}',
        #         'Portfolio weight, %': '{:.2f}',
        #         'Unrealized gain, $': '{:,.2f}',
        #         'Realized gain, $': "{:,.2f}",
        #         'Total return, $': "{:,.2f}",
        #         'Return rate, %': '{:.2f}'
        #     }).\
        #     render()
        self.df_returns = df

        return self

    def _get_stock_risk(self):
        self.df_stock_risk = self._ptfm.stock_risk_analysis()

        self.df_stock_risk.rename(columns={
            'alpha': "Jensen's alpha, %",
            'beta': 'Beta',
            'returns_mean': 'Monthly return mean, %',
            'returns_var': 'Monthly return variance, %'
        }, inplace=True)
        return self

    def _get_stock_correlations(self):
        self.df_stock_corr, _ = self._ptfm.stock_correlation_matrix()
        return self

    def _get_closed_positions(self):
        """
        Get three best/worst closed positions by realized gains
        """
        self._df_closed = pd.read_hdf(self.datafile, 'closed')
        df1 = self._df_closed.nlargest(3, 'realized_gains')
        df2 = self._df_closed.nsmallest(3, 'realized_gains')
        df = pd.concat([df1, df2]).sort_values(by='realized_gains')
        df['buy_price'] = df['current_cost_basis'] / df['signed_size']

        columns = [
            'date', 'symbol', 'current_size', 'buy_price',
            'average_price', 'realized_gains']
        df = df[columns]
        df.rename(columns={
            'date': 'Date',
            'symbol': 'Security',
            'current_size': 'Shares',
            'buy_price': 'Average buy price, $',
            'average_price': 'Average sell price, $',
            'realized_gains': 'Realized gain, $'
            }, inplace=True)

        # apply styles
        self.df_closed_positions = df.style.\
            set_table_attributes('border=1 class="dataframe"').\
            bar(subset=['Realized gain, $'], align='mid',
                color=['#fc8d59', '#91bfdb']).\
            set_table_styles([
                {'selector': '.row_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}]).\
            format({
                'Date': '{:%Y-%m-%d}',
                'Shares': '{:,.0f}',
                'Average buy price, $': '{:,.2f}',
                'Average sell price, $': '{:,.2f}',
                'Realized gain, $': "{:+,.2f}"
            }).\
            render()
        return self

    def _get_open_positions(self):
        """
        Get three best/worst open positions by unrealized gains
        """
        market_prices = self._panel['Close'].iloc[-1]

        self._df_open = pd.read_hdf(self.datafile, 'open')
        df = self._df_open.copy()
        df['current_price'] =\
            df.apply(lambda x: market_prices[x.symbol], axis=1)
        df['unrealized_gains'] =\
            (df['current_price'] - df['average_price']) * df['final_size']

        df1 = df.nlargest(3, 'unrealized_gains').copy()
        df2 = df.nsmallest(3, 'unrealized_gains').copy()
        df = pd.concat([df1, df2]).sort_values(by='unrealized_gains')

        columns = [
            'date', 'symbol', 'final_size',
            'average_price', 'current_price', 'unrealized_gains']
        df = df[columns]
        df.rename(columns={
            'date': 'Date',
            'symbol': 'Security',
            'final_size': 'Shares',
            'average_price': 'Average buy price, $',
            'current_price': 'Current market price, $',
            'unrealized_gains': 'Unrealized gain, $'
            }, inplace=True)

        # apply styles
        self.df_open_positions = df.style.\
            set_table_attributes('border=1 class="dataframe"').\
            bar(subset=['Unrealized gain, $'], align='mid',
                color=['#fc8d59', '#91bfdb']).\
            set_table_styles([
                {'selector': '.row_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}]).\
            format({
                'Date': '{:%Y-%m-%d}',
                'Shares': '{:,.0f}',
                'Average buy price, $': '{:,.2f}',
                'Current market price, $': '{:,.2f}',
                'Unrealized gain, $': "{:+,.2f}"
            }).\
            render()
        return self

    def _get_portfolio_stats(self):
        """
        Get actual portfolio stats
        """
        pf_stats = self._ptfm.actual_portfolio_stats()

        pct_val = [
            "Total return",
            "Market return",
            "Annual return",
            "Annual volatility",
            "Max drawdown",
            "Daily value at risk"
        ]
        for col in pct_val:
            pf_stats[col] = pf_stats[col] * 100

        # format values
        pf_stats = pf_stats.to_frame().apply(
            lambda x:
            '{:.2f}'.format(x[0]) if x.name not in pct_val
            else '{:.2f}%'.format(x[0]), axis=1)

        # apply styles
        self.ptf_stats = pf_stats.to_frame().style.\
            set_table_attributes('border=1 class="dataframe"').\
            set_table_styles([
                {'selector': '.col_heading',
                 'props': [('display', 'none')]},
                {'selector': '.blank.level0',
                 'props': [('display', 'none')]}
            ]).\
            render()
        return self

    def calculate_all(self):
        """
        Run all calculations and save values to internal class props
        --------
        Returns:
        """
        self._get_daily_portfolio_panel()
        self._get_latest_portfolio_snapshot()
        self._get_daily_total_returns()
        self._get_stock_risk()
        self._get_stock_correlations()
        self._get_closed_positions()
        self._get_open_positions()
        self._get_portfolio_stats()
        return self


if __name__ == '__main__':
    bc = BackendClass('../data/data.h5')
    bc = bc.calculate_all()
