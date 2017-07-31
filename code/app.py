import pandas as pd
from portfolio_data import PortfolioData
from portfolio_model import PortfolioModels


if __name__ == "__main__":
    ptf = PortfolioData('../data/data.h5')
    pf = ptf.prepare_portfolio_data('read', 'read')
