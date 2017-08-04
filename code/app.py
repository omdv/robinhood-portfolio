import pandas as pd
from portfolio_data import PortfolioData
from portfolio_model import PortfolioModels


if __name__ == "__main__":
    datafile = '../data/data.h5'
    ptd = PortfolioData(datafile)
    ptd.prepare_portfolio_data('read', 'read')

    ptm = PortfolioModels(datafile)
    pf = ptm.calc_returns().pf
