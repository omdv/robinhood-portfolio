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
        self.df_ord = None
        self.ptf_daily = None

    def download_save_data(self):
        rd = RobinhoodData(self.datafile)
        self.df_ord, df_div = rd.download_robinhood_data()

        md = MarketData(self.datafile)
        md.download_market_data(
            self.df_ord.symbol.unique(),
            self.df_ord.date.min().strftime('%Y%m%d'),
            self.df_ord.date.max().strftime('%Y%m%d'))
        return self

    def calculate_portfolio_performance(self):
        self.df_ord = pd.read_hdf(self.datafile, 'orders')
        ptfm = PortfolioModels(self.datafile)
        self.ptf_daily = ptfm.calc_daily_returns(
            self.df_ord.date.min(),
            self.df_ord.date.max()).panelframe
        return self
