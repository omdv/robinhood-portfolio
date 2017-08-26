import pandas as pd
from backend.portfolio_model import PortfolioModels
from backend.robinhood_data import RobinhoodData
from backend.market_data import MarketData


class BackendClass(object):
    """
    Main backend class, provides wrappers to donwload robinhood and market
    data, provides access to portfolio models
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
        self._ptfm = PortfolioModels(self.datafile)
        self._df_ord = pd.read_hdf(self.datafile, 'orders')
        self._panel = self._ptfm.daily_portfolio_changes(
            self._df_ord.date.min(),
            pd.Timestamp("today").strftime('%Y%m%d')).panelframe
        return self

    def _get_daily_returns_series(self):
        self.daily_returns = self._panel['current_total_return'].sum(axis=1)
        return self

    def _get_latest_portfolio_snapshot(self):
        # use only the last row
        df = self._panel.iloc[:, -1, :]     
        columns = [
            'total_quantity',
            'current_ratio',
            'cost_basis',
            'current_value',
            'current_capital_gain',
            'total_dividends',
            'current_total_return',
            'current_return_rate'
        ]
        df = df[columns]
        # convert ratios to percent
        df['current_ratio'] = df['current_ratio'] * 100

        # add summary row
        df = df.copy()  # avoid chained assignment warning
        df.loc['Summary', :] = df.sum(axis=0)

        df.loc['Summary', 'current_return_rate'] =\
            df.loc['Summary', 'current_total_return'] /\
            df.loc['Summary', 'cost_basis'] * 100

        # zero-out return rate for closed positions
        df.loc[df['total_quantity'] == 0, 'current_return_rate'] = 0

        # fix market return rate
        df.loc['market', :] = '-'
        df.loc['market', 'current_total_return'] = (
            self._panel['Close', :, 'market'][-1] -
            self._panel['Close', :, 'market'][0])
        df.loc['market', 'current_return_rate'] = (
            df.loc['market', 'current_total_return']) /\
            self._panel['Close', :, 'market'][0] * 100
        df.rename(index={'market': '^SPX'}, inplace=True)

        # rename for HTML
        df.rename(columns={
            'total_quantity': 'Total shares',
            'cost_basis': 'Current cost basis, $',
            'total_dividends': 'Dividends, $',
            'current_value': 'Current value, $',
            'current_ratio': 'Portfolio ratio, %',
            'current_capital_gain': 'Capital gain, $',
            'current_total_return': 'Total return, $',
            'current_return_rate': 'Return rate, % (*)'
        }, inplace=True)

        self.df_returns = df
        return self

    def _get_stock_risk(self):
        self.df_stock_risk, self.df_stock_correlations =\
            self._ptfm.stock_risk_analysis()
        return self

    def calculate_all(self):
        """
        Run all calculations and save values to internal class props
        --------
        Returns:
        """
        self._get_daily_portfolio_panel()
        self._get_latest_portfolio_snapshot()
        self._get_daily_returns_series()
        self._get_stock_risk()

        # get stock properties
        # df_stocks, df_covar, df_corr, ptf_dict =\
        #     ptfm.calc_portfolio_performance()
        # self.df_stocks = pd.DataFrame()
        # self.df_covar = pd.DataFrame()
        # self.df_corr = pd.DataFrame()
        return self


if __name__ == '__main__':
    bc = BackendClass('../data/data.h5')
    bc = bc.calculate_all()
