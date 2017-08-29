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
        pf['total_quantity'] = 0
        pf['cost_basis'] = 0
        # loop over tickers, except the last one, which is market
        for key in pf.minor_axis[:-1]:
            df1 = pf.loc[:, :, key]
            df2 = df_ord[df_ord['symbol'] == key]
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['total_quantity', 'cost_basis']],
                left_index=True, right_index=True, how='left')
            df.rename(columns={
                'total_quantity_y': 'total_quantity',
                'cost_basis_y': 'cost_basis'}, inplace=True)
            df.drop('total_quantity_x', axis=1, inplace=True)
            df.drop('cost_basis_x', axis=1, inplace=True)
            # now propagate values from last observed
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            pf.loc[:, :, key] = df
        return pf

    def _merge_market_with_dividends(self, df_div, pf):
        """
        Helper to merge the market frame with dividends
        """
        pf['total_dividends'] = 0
        pf['dividend_rate'] = 0
        for key in pf.minor_axis[:-1]:
            df1 = pf.loc[:, :, key]
            df2 = df_div[df_div['symbol'] == key]
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['total_amount', 'rate']],
                left_index=True, right_index=True, how='left')
            df.drop('total_dividends', axis=1, inplace=True)
            df.drop('dividend_rate', axis=1, inplace=True)
            df.rename(columns={
                'total_amount': 'total_dividends',
                'rate': 'dividend_rate'}, inplace=True)
            # now propagate values from last observed
            df['total_dividends'].fillna(method='ffill', inplace=True)
            df['total_dividends'].fillna(0, inplace=True)
            pf.loc[:, :, key] = df
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
        df_ord['cost_basis'] =\
            df_ord.groupby('symbol').cost_basis.cumsum()
        df_div['total_amount'] = df_div.groupby('symbol').amount.cumsum()

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

    def daily_portfolio_changes(self, start_date, end_date):
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
        self._prepare_portfolio_for_date_range(start_date, end_date)
        pf = self.panelframe

        # portfolio calculations
        pf['dividend_yield'] = pf['dividend_rate'] / pf['Close'] * 100
        pf['current_value'] = pf['total_quantity'] * pf['Close']
        pf['current_ratio'] =\
            (pf['current_value'].T / pf['current_value'].sum(axis=1)).T
        pf['current_capital_gain'] = pf['current_value'] - pf['cost_basis']
        pf['current_total_return'] = pf['current_capital_gain'] +\
            pf['total_dividends']

        # zero out cumulative cost_basis
        pf['cost_basis'] = pf['cost_basis'].\
            where(pf['total_quantity'] != 0).fillna(0)

        # calculate ROI
        # TODO - account for closed positions
        pf['current_return_rate'] =\
            (pf['current_total_return'] / pf['cost_basis'] * 100).\
            where(pf['total_quantity'] != 0).fillna(method='ffill')

        # assign to panelframe
        self.panelframe = pf
        return self

    # TODO: refactor to apply
    def _process_all_orders(df_ord):
        df_ord['cum_quantity_to_date'] = df_ord.groupby('symbol').signed_quantity.cumsum()

        # dfn = df_ord
        # dfn['qdelta'] = dfn.apply(lambda x: , axis=1)

        df_open = pd.DataFrame()
        df_closed = pd.DataFrame()
        # iterate over all symbols - replace by groupby?
        for sym in df_ord.symbol.unique():
            df = df_ord[df_ord.symbol == sym].copy()
           
            # initiate open and closed positions DF
            df_open = df_open.append(df[df.signed_quantity > 0])
            df_closed = df_closed.append(df[df.signed_quantity < 0])

        return None


    def orders_performance(self):
        """
        Calculate performance of every order
        Parameters:
        None
        Return:
        Dataframe with returns for every execution
        """
        df = pd.read_hdf(self.datafile, 'orders')

        # get latest market prices
        last_prices = self.panelframe['Closed'].iloc[-1]

        # calculate P&L for open and closed positions
        df['current_price'] = df.apply(
            lambda x:
            x['cumulative_quantity'] * last_prices[x['symbol']], axis=1)

        # open P&L is straightforward
        df['Open P&L'] = df.apply(
            lambda x: (x['current_price'] - x['cost_basis']) if
            x['signed_quantity'] > 0 else 0, axis=1)

        # to calculate close P&L we need to determine the average weighted
        # price of shares bought till to date of sale
        df['cum_cost_basis_open_todate'] = df.apply(
            lambda x:
            df.loc[
                (df.symbol == x.symbol) &
                (df.date <= x.date) &
                (df.signed_quantity > 0), 'cost_basis'
            ].sum(), axis=1)

        df['cum_quantity_open_todate'] = df.apply(
            lambda x:
            df.loc[
                (df.symbol == x.symbol) &
                (df.date <= x.date), 'signed_quantity'
            ].sum(), axis=1)

        return df

    # NEW - CHECKED
    def stock_risk_analysis(self, risk_free_return=0):
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

    def risk_analysis_empyrical(self):
        # get monthly changes for all stocks
        stock_returns = self.stock_monthly_returns()

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
                x, stock_returns['market'], period='monthly')).\
            apply(pd.Series)

        # use empyrical to get monthly returns for portfolio
        ptf_monthly_returns = self.ptf_monthly_returns()
        df_stocks.loc['portfolio', 'returns_mean'] = ptf_monthly_returns.mean()
        df_stocks.loc['portfolio', 'returns_var'] = ptf_monthly_returns.var()

        ptf_alpha_beta = emp.alpha_beta(
            ptf_monthly_returns,
            stock_returns['market'],
            period='monthly')
        df_stocks.loc['portfolio', 'alpha'] = ptf_alpha_beta[0]
        df_stocks.loc['portfolio', 'beta'] = ptf_alpha_beta[1]

        return df_stocks



    # USED
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

    def ptf_monthly_returns(self):
        # monthly changes in portfolio value
        # using indirect calculation with mean ratios
        # TODO - implement a more accurate method
        stock_monthly_change = self.stock_monthly_returns()
        ptf_monthly_ratio = self.panelframe['current_ratio'].groupby([
            lambda x: x.year,
            lambda x: x.month]).mean()
        ptf_monthly_returns = (
            stock_monthly_change * ptf_monthly_ratio).sum(1)
        return ptf_monthly_returns


if __name__ == '__main__':
    start_date = pd.to_datetime('07/07/2016')
    end_date = pd.to_datetime('07/03/2017')

    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    start_date = df_ord.date.min()
    end_date = df_ord.date.max()

    ptf = PortfolioModels('../data/data.h5')
    pf = ptf.daily_portfolio_changes(start_date, end_date).panelframe
    df1, df2 = ptf.stock_risk_analysis(ptf.risk_free_return()/12.)
    df3 = ptf.risk_analysis_empyrical()
