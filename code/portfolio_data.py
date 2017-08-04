import pandas as pd
from robinhood_data import RobinhoodData
from market_data import MarketData

"""
this is likely not required - if merging is moved under portfolio model then
this is just a wrapper around marketData and portfolioData
"""


class PortfolioData():
    """
    Returns Panel of portfolio data, with symbols covering all symbols
    im portfolio, over date range, start to end.
    Parameters
    ----------
    datafile : location of h5 datafile
    """

    def __init__(self, datafile, mode='read'):
        self.datafile = datafile
        self.mode = mode

    # adding portfolio orders into a price panel
    # currently hacky
    def _merge_market_with_orders(self, df_ord, pf):
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

    # adding dividends to the market data panel
    def _merge_market_with_dividends(self, df_div, pf):
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

    def prepare_portfolio_data(self, robinhood_mode='read',
                               market_mode='read'):
        """
        robinhood_mode : string, (defaults to 'read')
        if read just download the previously downloaded data
        if download - download and save to datafile
        """
        if robinhood_mode == 'download':
            rd = RobinhoodData(self.datafile)
            df_div, df_ord = rd.download_robinhood_data()
        elif robinhood_mode == 'read':
            df_div = pd.read_hdf(self.datafile, 'dividends')
            df_ord = pd.read_hdf(self.datafile, 'orders')
        if market_mode == 'download':
            md = MarketData(self.datafile)
            pf = md.download_market_data(
                df_ord.symbol.unique(),
                df_ord.date.min().strftime("%Y%m%d"),
                df_ord.date.max().strftime("%Y%m%d"))
        elif market_mode == 'read':
            pf = pd.read_hdf(self.datafile, 'market')

        # merge with portfolio
        pf = self._merge_market_with_dividends(df_div, pf)
        pf = self._merge_market_with_orders(df_ord, pf)
        pf.to_hdf(self.datafile, 'portfolio')
        return pf


if __name__ == "__main__":
    ptf = PortfolioData('../data/data.h5')
    pf = ptf.prepare_portfolio_data('read', 'read')
