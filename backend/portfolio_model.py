# TODO:
# 1. take into account the RB fee when selling position
import numpy as np
import pandas as pd


# calculating portfolio performance
class PortfolioModels():
    def __init__(self, datafile):
        self.datafile = datafile
        self.panelframe = None
        self.return_todate = None
        return None

    def _merge_market_with_orders(self, df_ord, pf):
        """
        Helper for merging orders with panel frame with market data
        """
        pf['total_quantity'] = 0
        pf['total_cost_basis'] = 0
        # loop over tickers, except the last one, which is market
        for key in pf.minor_axis[:-1]:
            df1 = pf.loc[:, :, key]
            df2 = df_ord[df_ord['symbol'] == key]
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['total_quantity', 'total_cost_basis']],
                left_index=True, right_index=True, how='left')
            df.rename(columns={
                'total_quantity_y': 'total_quantity',
                'total_cost_basis_y': 'total_cost_basis'}, inplace=True)
            df.drop('total_quantity_x', axis=1, inplace=True)
            df.drop('total_cost_basis_x', axis=1, inplace=True)
            # now propagate values from last observed
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            pf.ix[:, :, key] = df
        return pf

    def _merge_market_with_dividends(self, df_div, pf):
        """
        Helper to merge the market frame with dividends
        """
        pf['total_dividends'] = 0
        for key in pf.minor_axis[:-1]:
            df1 = pf.loc[:, :, key]
            df2 = df_div[df_div['symbol'] == key]
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['total_amount']],
                left_index=True, right_index=True, how='left')
            df.drop('total_dividends', axis=1, inplace=True)
            df.rename(
                columns={'total_amount': 'total_dividends'}, inplace=True)
            # now propagate values from last observed
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            pf.ix[:, :, key] = df
        return pf

    def _prepare_portfolio_for_date_range(self, start_date, end_date):
        """
        Prepare portfolio panelframe for a given range of dates
        by merging orders and dividends with stock prices and
        calculating cumulative values
        -------------
        Parameters:
        start_date: as datetime[64]
        """
        df_ord = pd.read_hdf(self.datafile, 'orders')
        df_div = pd.read_hdf(self.datafile, 'dividends')
        pf = pd.read_hdf(self.datafile, 'market')

        # create subset based on date range
        df_ord = df_ord[(df_ord['date'] >= start_date) &
                        (df_ord['date'] <= end_date)].copy()
        df_div = df_div[(df_div['date'] >= start_date) &
                        (df_div['date'] <= end_date)].copy()
        pf = pf[:, start_date:end_date, :].copy()

        # calculate cumulative quantitities and cost basis
        df_ord['total_quantity'] =\
            df_ord.groupby('symbol').signed_quantity.cumsum()
        df_ord['total_cost_basis'] =\
            df_ord.groupby('symbol').cost_basis.cumsum()
        df_div['total_amount'] = df_div.groupby('symbol').amount.cumsum()

        # merge both with market
        pf = self._merge_market_with_orders(df_ord, pf)
        pf = self._merge_market_with_dividends(df_div, pf)

        self.panelframe = pf
        return self

    # process self.pf to add daily returns and related info
    def calc_daily_returns(self, start_date, end_date):

        # prepare the portfolio panel
        self._prepare_portfolio_for_date_range(start_date, end_date)
        pf = self.panelframe

        # portfolio calculations
        pf['current_price'] = pf['total_quantity'] * pf['Close']
        pf['current_ratio'] =\
            (pf['current_price'].T / pf['current_price'].sum(axis=1)).T
        pf['current_return_raw'] = pf['current_price'] - pf['total_cost_basis']
        pf['current_return_div'] = pf['current_return_raw'] +\
            pf['total_dividends']
        pf['current_roi'] = pf['current_return_div'] / pf['total_cost_basis']

        # fix the current roi for positions with zero holdings
        # TODO

        # calculate final return
        self.return_todate = self.panelframe['current_return_div', -1, :].sum()

        # assign to panelframe
        self.panelframe = pf
        return self

    # beta for a provided panel, index value should be last column
    def calc_beta_by_covar(self):
        pf = self.panelframe
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
    def alpha_by_capm(self):
        """
        Calculate alpha as per CAPM
        www.alphagamma.eu/finance/how-to-calculate-alpha-of-your-portfolio/
        -------------
        Parameters:
        """
        pf = self.panelframe
        tb = pd.read_hdf(self.datafile, 'treasury_bills')

        # get portfolio age to estimate durations
        portfolio_age = pf['Open'].index.max() - pf['Open'].index.min()
        portfolio_age = portfolio_age.days

        if portfolio_age > 365:
            print(None)
        return None


if __name__ == '__main__':
    start_date = pd.to_datetime('07/07/2016')
    end_date = pd.to_datetime('07/03/2017')

    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    start_date = df_ord.date.min()
    end_date = df_ord.date.max()

    ptf = PortfolioModels('../data/data.h5')
    pf = ptf.calc_daily_returns(start_date, end_date).panelframe
    ptf.calc_full_return()
