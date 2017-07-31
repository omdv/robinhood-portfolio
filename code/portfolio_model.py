# TODO:
# - account for RB fee when selling position
import numpy as np
import pandas as pd


# calculating portfolio performance
class PortfolioModels():
    def __init__(self, datafile):
        self.datafile = datafile
        self.pf = pd.read_hdf(self.datafile, 'portfolio')
        return None

    def calc_returns(self):
        pf = self.pf
        # portfolio calculations
        pf['current_price'] = pf['total_quantity'] * pf['Close']
        pf['current_ratio'] =\
            (pf['current_price'].T / pf['current_price'].sum(axis=1)).T
        pf['current_return_raw'] = pf['current_price'] - pf['total_cost_basis']
        pf['current_return_div'] = pf['current_price'] + pf['total_dividends']\
            - pf['total_cost_basis']
        pf['current_roi'] = pf['current_return_div'] / pf['total_cost_basis']
        self.pf = pf
        return self

    # beta for a provided panel, index value should be last column
    def calc_beta_by_covar(self):
        pf = self.pf
        betas = dict()
        # dates = pd.to_datetime([start_date, end_date])
        pf.ix['Daily_change'] = (pf.ix['Close'] - pf.ix['Close'].shift(1))\
            / pf.ix['Close'].shift(1) * 100
        covar = np.cov(pf.ix['Daily_change'][1:], rowvar=False, ddof=0)
        variance = np.var(pf.ix['Daily_change'])['market']
        for i, j in enumerate(pf.ix['Daily_change'].columns):
            betas[j] = covar[-1, i] / variance
        self.betas = betas
        return self

    # beta for a provided panel based on simple return
    def calc_beta_by_return(self, pf):
        return None

    # alpha by capm model
    def alpha_by_capm():
        return None


if __name__ == '__main__':
    ptf = PortfolioPerformance('../data/data.h5')
    pf = ptf.calc_returns().pf
