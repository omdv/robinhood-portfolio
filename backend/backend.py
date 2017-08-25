import pandas as pd
import numpy as np
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
        self.panel = None

    def download_save_data(self):
        rd = RobinhoodData(self.datafile)
        self.df_ord, df_div = rd.download_robinhood_data()

        md = MarketData(self.datafile)
        md.download_save_market_data(
            self.df_ord.symbol.unique(),
            self.df_ord.date.min().strftime('%Y%m%d'),
            self.df_ord.date.max().strftime('%Y%m%d'))
        return self

    def calc_all_values(self):
        # get panel
        df_ord = pd.read_hdf(self.datafile, 'orders')
        ptfm = PortfolioModels(self.datafile)

        # prepare returns dataframe
        self.panel = ptfm.calc_daily_returns(
            df_ord.date.min(),
            df_ord.date.max()).panelframe

        # use only the last row
        df = self.panel.iloc[:, -1, :-1]
        df = df[
            ['total_quantity', 'total_cost_basis', 'total_dividends',
             'current_price', 'current_ratio', 'current_return_raw',
             'current_return_div', 'current_roi_raw', 'current_roi_div']]
        # convert quantities to integer
        df['total_quantity'] = df['total_quantity'].astype(np.int32)
        # convert ratios to percent
        df['current_ratio'] = df['current_ratio'] * 100
        df['current_roi_raw'] = df['current_roi_raw'] * 100
        df['current_roi_div'] = df['current_roi_div'] * 100
        # add summary row
        df = df.copy()  # avoid chained assignment warning
        df.loc['Summary', :] = df.sum(axis=0)
        df.loc['Summary', 'current_roi_raw'] =\
            df.loc['Summary', 'current_return_raw'] /\
            df.loc['Summary', 'total_cost_basis'] * 100
        df.loc['Summary', 'current_roi_div'] =\
            df.loc['Summary', 'current_return_div'] /\
            df.loc['Summary', 'total_cost_basis'] * 100
        # rename for HTML
        df.rename(columns={
            'total_quantity': 'Total shares',
            'total_cost_basis': 'Total cost basis, $',
            'total_dividends': 'Cumulative dividends, $',
            'current_price': 'Current market price, $',
            'current_ratio': 'Ratio in portfolio, %',
            'current_return_raw': 'Capital gain, $',
            'current_return_div': 'Total return, $',
            'current_roi_raw': 'RoR w/o dividends, %',
            'current_roi_div': 'RoR with dividends, %'}, inplace=True)
        self.df_returns = df

        # get stock properties
        df_stocks, df_covar, df_corr, ptf_dict =\
            ptfm.calc_portfolio_performance()

        


        return self
