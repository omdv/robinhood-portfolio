# TODO:
# 1. take into account the RB fee when selling position
import numpy as np
import pandas as pd
import empyrical as emp


# calculating portfolio performance
class PortfolioModels():
    def __init__(self, datafile):
        self.datafile = datafile
        self.panelframe = None
        return None

    def _merge_market_with_orders(self, df_ord, pf):
        """
        Helper for merging orders with panel frame with market data
        """
        pf['cum_size'] = 0
        pf['cum_cost_basis'] = 0
        pf['cum_realized_gain'] = 0
        # loop over tickers, except the last one, which is market
        for key in pf.minor_axis[:-1]:
            df1 = pf.loc[:, :, key]
            df2 = df_ord[df_ord['symbol'] == key].copy()
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['cum_size', 'cum_cost_basis', 'cum_realized_gain']],
                left_index=True, right_index=True, how='left')
            df.rename(columns={
                'cum_size_y': 'cum_size',
                'cum_cost_basis_y': 'cum_cost_basis',
                'cum_realized_gain_y': 'cum_realized_gain'}, inplace=True)

            df.drop('cum_size_x', axis=1, inplace=True)
            df.drop('cum_cost_basis_x', axis=1, inplace=True)
            df.drop('cum_realized_gain_x', axis=1, inplace=True)

            # now propagate values from last observed
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            pf.loc[:, :, key] = df
        return pf

    def _merge_market_with_dividends(self, df_div, pf):
        """
        Helper to merge the market frame with dividends
        """
        pf['cum_dividends'] = 0
        pf['dividend_rate'] = 0
        for key in pf.minor_axis[:-1]:
            df1 = pf.loc[:, :, key]
            df2 = df_div[df_div['symbol'] == key].copy()
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['cum_dividends', 'rate']],
                left_index=True, right_index=True, how='left')

            df.drop('cum_dividends_x', axis=1, inplace=True)
            df.drop('dividend_rate', axis=1, inplace=True)

            df.rename(columns={
                'cum_dividends_y': 'cum_dividends',
                'rate': 'dividend_rate'}, inplace=True)

            # now propagate values from last observed
            df['cum_dividends'].fillna(method='ffill', inplace=True)
            df['cum_dividends'].fillna(0, inplace=True)
            pf.loc[:, :, key] = df
        return pf

    def _prepare_portfolio(self):
        """
        Prepare portfolio panelframe
        by merging orders and dividends with stock prices and
        calculating cumulative values
        -------------
        Parameters:
        - None, using df_ord, df_div and market prices
        """

        # read frames for internal use
        # df_ord = pd.read_hdf(self.datafile, 'orders')
        df_div = pd.read_hdf(self.datafile, 'dividends')
        df_open = pd.read_hdf(self.datafile, 'open')
        df_closed = pd.read_hdf(self.datafile, 'closed')
        pf = pd.read_hdf(self.datafile, 'market')

        # try this out
        # TODO: if works - df_ord may be replaced
        df_ord = pd.concat([df_open, df_closed]).sort_index()

        # calculate cumulative size and cost basis
        df_ord['cum_size'] =\
            df_ord.groupby('symbol').signed_size.cumsum()

        # cost basis for closed orders is equal to the one for original open
        # position, so cumulative does not include any gains or losses from
        # closing orders
        df_ord['cum_cost_basis'] =\
            df_ord.groupby('symbol').current_cost_basis.cumsum()

        # aggregate orders on the same day
        func = {
            'average_price': 'mean',
            'current_cost_basis': 'sum',
            'current_size': 'sum',
            'fees': 'sum',
            'final_cost_basis': 'sum',
            'final_size': 'sum',
            'signed_size': 'sum',
            'cum_size': 'sum',
            'cum_cost_basis': 'sum',
            'realized_gains': 'sum'}
        df_ord = df_ord.groupby(['date', 'symbol'], as_index=False).agg(func)

        # calculate cumulative size and cost basis
        df_ord['cum_size'] =\
            df_ord.groupby('symbol').signed_size.cumsum()
        df_ord['cum_cost_basis'] =\
            df_ord.groupby('symbol').current_cost_basis.cumsum()
        df_ord['cum_realized_gain'] =\
            df_ord.groupby('symbol').realized_gains.cumsum()

        # fix the average price, so it is weighted mean
        df_ord['average_price'] =\
            df_ord['cum_cost_basis'] / df_ord['cum_size']

        # calculate cumulative dividends
        df_div['cum_dividends'] = df_div.groupby('symbol').amount.cumsum()

        # merge both with market
        pf = self._merge_market_with_orders(df_ord, pf)
        pf = self._merge_market_with_dividends(df_div, pf)

        # replace null stock prices using backfill to avoid issues with
        # Daily_change and beta calculations
        close_price = pf['Close']
        close_price.values[close_price.values == 0] = np.nan
        close_price.fillna(method='bfill', inplace=True)
        pf['Close'] = close_price

        self.panelframe = pf
        return self

    def daily_portfolio_changes(self):
        """
        Calculate daily prices, cost-basis, ratios, returns, etc.
        Used for plotting and also showing the final snapshot of
        the portfolio
        -------------
        Parameters:
        - None
        Return:
        - Panelframe with daily return values. Used for plotting
        and for html output
        """

        # prepare the portfolio panel
        self._prepare_portfolio()
        pf = self.panelframe

        """
        Main daily portfolio properties
        """
        # dividend yield
        pf['dividend_yield'] = pf['dividend_rate'] / pf['Close'] * 100

        # cumulative current value of the position for the given security
        pf['cum_value'] = pf['cum_size'] * pf['Close']

        # current weight of the given security in the portfolio - matrix
        pf['current_weight'] =\
            (pf['cum_value'].T / pf['cum_value'].sum(axis=1)).T

        # unrealized gain on open positions
        pf['cum_unrealized_gain'] = pf['cum_value'] - pf['cum_cost_basis']

        # capital gain on closed positions
        pf['cum_total_return'] = pf['cum_unrealized_gain'] +\
            pf['cum_dividends'] + pf['cum_realized_gain']

        # calculate ROI
        pf['current_return_rate'] =\
            (pf['cum_total_return'] / pf['cum_cost_basis'] * 100).\
            where(pf['cum_size'] != 0)

        # assign to panelframe
        self.panelframe = pf
        return self

    # USE empyrical instead
    def stock_risk_analysis_legacy(self, risk_free_return=0):
        """
        Calculate monthly returns and run all main portfolio performance
        calculations, such as mean returns, std, portfolio mean, std, beta, etc
        References:
        1. p. 137 of Modern Portfolio Theory and Investment Analysis
        edition 9
        2. faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        -------------
        Parameters:
        - None
        - Use only stock prices
        Return:
        - Dataframe of properties for each security in portfolio
        """

        # get monthly changes for all stocks
        stock_monthly_change = self.stock_monthly_returns()

        # get mean values and std by security
        returns_mean = stock_monthly_change.mean(axis=0)
        returns_std = stock_monthly_change.std(axis=0)
        returns_var = stock_monthly_change.var(axis=0)

        # get covariance matrix
        returns_covar = np.cov(
            stock_monthly_change.values, rowvar=False, ddof=1)

        # get correlation matrix
        std_products = np.dot(
            returns_std.values.reshape(-1, 1),
            returns_std.values.reshape(1, -1))
        returns_corr = returns_covar / std_products

        # get betas for each stock
        stock_betas = np.round(returns_covar[-1] / returns_std['market']**2, 5)

        # get alphas for each stock, ref. [1], convert to annual value
        stock_alphas = returns_mean - stock_betas *\
            (returns_mean['market'] - risk_free_return) -\
            risk_free_return
        stock_alphas = stock_alphas * 12

        # construct dataframes with stock properties
        df_stocks = pd.DataFrame({
            'returns_mean': returns_mean,
            'returns_var': returns_var,
            'beta': stock_betas,
            'alpha': stock_alphas
        })

        df_corr = pd.DataFrame(
            returns_corr,
            columns=returns_mean.keys(),
            index=returns_mean.keys())

        # use empyrical to get monthly returns for portfolio
        ptf_monthly_returns = self.ptf_monthly_returns()
        df_stocks.loc['portfolio', 'returns_mean'] = ptf_monthly_returns.mean()
        df_stocks.loc['portfolio', 'returns_var'] = ptf_monthly_returns.var()

        ptf_alpha_beta = emp.alpha_beta(
            ptf_monthly_returns,
            stock_monthly_change['market'],
            period='monthly')
        df_stocks.loc['portfolio', 'alpha'] = ptf_alpha_beta[0]
        df_stocks.loc['portfolio', 'beta'] = ptf_alpha_beta[1]

        return df_stocks, df_corr

    def stock_correlation_matrix(self):
        """
        Calculate stock correlation matrix
        References:
        1. p. 137 of Modern Portfolio Theory and Investment Analysis
        edition 9
        2. faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        -------------
        Parameters:
        - None
        - Use stock prices, div yields and portfolio weights
        Return:
        - Correlation dataframe
        """

        # get monthly changes for all stocks
        stock_returns = self.stock_monthly_returns()
        stock_returns['portfolio'] = self.ptf_monthly_returns()

        # get mean values and std by security
        returns_mean = stock_returns.mean(axis=0)
        returns_std = stock_returns.std(axis=0)

        # get covariance matrix
        returns_covar = np.cov(
            stock_returns.values, rowvar=False, ddof=1)

        # get correlation matrix
        std_products = np.dot(
            returns_std.values.reshape(-1, 1),
            returns_std.values.reshape(1, -1))
        returns_corr = returns_covar / std_products

        df_corr = pd.DataFrame(
            returns_corr,
            columns=returns_mean.keys(),
            index=returns_mean.keys())

        return df_corr

    def stock_risk_analysis(self, if_risk_free_return=True):
        """
        Calculate risk properties for every security in the portfolio
        using empyrical library.
        Results are consistent with self-written routine
        References:
        1. p. 137 of Modern Portfolio Theory and Investment Analysis
        edition 9
        2. faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        -------------
        Parameters:
        - If include risk_free_return or not
        - Using stock prices, weight ratios and div yield
        Return:
        - Dataframe of properties for each security in portfolio
        """

        # get monthly changes for all stocks
        stock_returns = self.stock_monthly_returns()

        # get risk free return
        if if_risk_free_return:
            risk_free_return = self.risk_free_return()/12
        else:
            risk_free_return = 0

        # get mean values and std by security
        returns_mean = stock_returns.mean(axis=0)
        returns_var = stock_returns.var(axis=0)

        # construct resulting dataframe
        df_stocks = pd.DataFrame({
            'returns_mean': returns_mean,
            'returns_var': returns_var,
        })

        df_stocks[['alpha', 'beta']] = stock_returns.\
            apply(lambda x: emp.alpha_beta(
                x, stock_returns['market'],
                risk_free_return, period='monthly')).\
            apply(pd.Series)

        # use empyrical to get monthly returns for portfolio
        ptf_monthly_returns = self.ptf_monthly_returns()
        df_stocks.loc['portfolio', 'returns_mean'] = ptf_monthly_returns.mean()
        df_stocks.loc['portfolio', 'returns_var'] = ptf_monthly_returns.var()

        ptf_alpha_beta = emp.alpha_beta(
            ptf_monthly_returns, stock_returns['market'],
            risk_free_return, period='monthly')
        df_stocks.loc['portfolio', 'alpha'] = ptf_alpha_beta[0]
        df_stocks.loc['portfolio', 'beta'] = ptf_alpha_beta[1]

        return df_stocks

    def risk_free_return(self, period='monthly'):
        """
        Risk free return based on T-bills.
        -------------
        Parameters:
        - period: monthly, quarterly or annual
        Returns:
        - annualized value of return based on period
        """
        tb = pd.read_hdf(self.datafile, 'treasury_bills')
        TBILLS_PERIODS = {
            'yearly': 'TB1YR',
            'monthly': 'TB4WK',
            'quarterly': 'TB3MS'
        }
        return tb[TBILLS_PERIODS[period]].mean()

    def stock_monthly_capital_gain(self):
        """
        Monthly capital gain for all stocks, market index and portfolio
        -------------
        Parameters:
        - none
        - Using stock prices
        Returns:
        - dataframe with monthly returns in % by symbol
        """
        pf = self.panelframe

        # monthly changes in stock_prices prices
        stock_prices = pf['Close']
        stock_month_start = stock_prices.groupby([
            lambda x: x.year,
            lambda x: x.month]).first()
        stock_month_end = stock_prices.groupby([
            lambda x: x.year,
            lambda x: x.month]).last()
        stock_monthly_return = (stock_month_end - stock_month_start) /\
            stock_month_start * 100

        return stock_monthly_return

    def stock_monthly_div(self):
        """
        Monthly dividend yield in %
        Note that div are accounted for only if position existed
        TODO: add a div payout dataframe
        -------------
        Parameters:
        - none
        - Using dividend yield
        Returns:
        - dataframe with monthly dividend yield in $ by symbol
        """
        pf = self.panelframe
        stock_monthly_div_yield = pf['dividend_yield'].groupby([
            lambda x: x.year,
            lambda x: x.month]).mean()
        stock_monthly_div_yield.fillna(0, inplace=True)
        return stock_monthly_div_yield

    def stock_monthly_returns(self):
        """
        Monthly returns = capital gain + dividend yields for all symbols
        -------------
        Parameters:
        - none
        Returns:
        - dataframe with monthly returns in % by symbol
        """
        return self.stock_monthly_capital_gain() + self.stock_monthly_div()

    def ptf_monthly_returns(self):
        """
        monthly changes in portfolio value
        using indirect calculation with mean ratios
        TODO - implement a more accurate method
        -------------
        Parameters:
        - none
        - Using stock prices, portfolio weights on every day and div yield
        Returns:
        - dataframe with monthly returns in % by symbol
        """
        stock_monthly_change = self.stock_monthly_returns()
        ptf_monthly_ratio = self.panelframe['current_weight'].groupby([
            lambda x: x.year,
            lambda x: x.month]).mean()
        ptf_monthly_returns = (
            stock_monthly_change * ptf_monthly_ratio).sum(1)
        return ptf_monthly_returns

    def annual_returns(self):
        """
        Annualized returns for all stocks
        -------------
        Parameters:
        - none
        Returns:
        - merge with other returns
        """
        return emp.annual_return(
            self.stock_monthly_returns()/100, period='monthly')


if __name__ == '__main__':
    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    df_div = pd.read_hdf('../data/data.h5', 'dividends')
    df_open = pd.read_hdf('../data/data.h5', 'open')
    df_closed = pd.read_hdf('../data/data.h5', 'closed')
    pf = pd.read_hdf('../data/data.h5', 'market')

    ptf = PortfolioModels('../data/data.h5')
    pf = ptf.daily_portfolio_changes().panelframe

    # this section uses only stock prices, div yields and weights
    df_risk = ptf.stock_risk_analysis(True)
    df_corr = ptf.stock_correlation_matrix()
