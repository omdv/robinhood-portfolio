# TODO:
# - account for RB fee when selling position
import numpy as np


# calculating portfolio performance
class PortfolioPerformance():
    def __init__(self, risk_free):
        self.risk_free = risk_free
        return None

    # beta for a provided panel, index value should be last column
    def beta_by_covar(self, pf):
        betas = dict()
        # dates = pd.to_datetime([start_date, end_date])
        pf.ix['Daily_change'] = (pf.ix['Close'] - pf.ix['Close'].shift(1))\
            / pf.ix['Close'].shift(1) * 100
        covar = np.cov(pf.ix['Daily_change'][1:], rowvar=False, ddof=0)
        variance = np.var(pf.ix['Daily_change'])['market']
        for i, j in enumerate(pf.ix['Daily_change'].columns):
            betas[j] = covar[-1, i] / variance
        return betas

    # beta for a provided panel based on simple return
    def beta_by_return(self, pf):
        return None

    # alpha by capm model
    def alpha_by_capm():
        return None


if __name__ == "__main__":
    None
