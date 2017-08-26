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

    # OLD - REMOVE
    def calc_beta_by_covar(self):
        """
        Calculate betas for a provided panel with market index
        being the last column
        TODO: if one of the stocks was not traded for any portion
        of the considered duration it will have a lower beta since the daily
        price changes for the days when it was not trading will be zero,
        i.e. no influence of the market
        -------------
        Parameters:
        - None
        Return:
        - Total portfolio beta for every time index
        """
        pf = self.panelframe
        # betas_dict = dict()
        betas_list = list()

        # calculate daily change, covariance and betas
        pf['Daily_change'] = (pf['Close'] - pf['Close'].shift(1))\
            / pf['Close'].shift(1) * 100
        covar = np.cov(pf.loc['Daily_change'][1:], rowvar=False, ddof=0)
        market_variance = np.var(pf.loc['Daily_change'])['market']
        for i, j in enumerate(pf.loc['Daily_change'].columns):
            # betas_dict[j] = covar[-1, i] / market_variance
            betas_list.append(covar[-1, i] / market_variance)

        # daily beta and mean beta
        daily_beta = np.dot(pf['current_ratio'], betas_list)
        ptf_beta = daily_beta.mean()
        self.beta = ptf_beta
        return ptf_beta

    # NEW - CHECKED
    def stock_risk_analysis(self):
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
        # pf = self.panelframe
        # ptf_returns = pf['current_total_return'].sum(1)

        # get monthly changes for all stocks
        stock_monthly_change = self.monthly_returns()

        # get mean values and std by security
        returns_mean = stock_monthly_change.mean(axis=0)
        returns_std = stock_monthly_change.std(axis=0)

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
        stock_alphas = returns_mean - stock_betas * returns_mean['market']
        stock_alphas = stock_alphas * 12

        # construct dataframes with stock properties
        df_stocks = pd.DataFrame({
            'returns_mean': returns_mean,
            'returns_std': returns_std,
            'beta': stock_betas,
            'alpha': stock_alphas
        })

        df_corr = pd.DataFrame(
            returns_corr,
            columns=returns_mean.keys(),
            index=returns_mean.keys())

        # portfolio ratios at the end of every month
        # ratios_month_end = ratios.groupby([
        #     lambda x: x.year, lambda x: x.month]).last()
        # ptf_returns_monthly = stock_monthly_change * ratios_month_end
        # ptf_returns_monthly = ptf_returns_monthly.sum(axis=1)
        # ptf_month_start = ptf_returns.groupby([
        #     lambda x: x.year,
        #     lambda x: x.month]).first()
        # ptf_month_end = ptf_returns.groupby([
        #     lambda x: x.year,
        #     lambda x: x.month]).last()
        # cost_month_start = pf['cost_basis'].sum(1).groupby([
        #     lambda x: x.year,
        #     lambda x: x.month]).first()

        # ptf_monthly_change = (ptf_month_end - ptf_month_start) \
        #     / cost_month_start * 100

        # # calculate portofolio values using the last known ratio
        # ptf_ratio = pf['current_ratio'].iloc[-1].values
        # ptf_std = np.sqrt(np.dot(
        #     np.dot(ptf_ratio.reshape(1, -1), returns_covar),
        #     ptf_ratio.reshape(-1, 1)))
        # ptf_beta = np.dot(stock_betas, ptf_ratio)
        # ptf_alpha = np.dot(stock_alphas, ptf_ratio)
        # ptf_return = ptf_monthly_change[1:].mean(axis=0)

        # df_stocks.loc['portfolio', 'returns_mean'] = ptf_return
        # df_stocks.loc['portfolio', 'returns_std'] = ptf_std[0][0]
        # df_stocks.loc['portfolio', 'beta'] = ptf_beta
        # df_stocks.loc['portfolio', 'alpha'] = ptf_alpha

        # df_covar = pd.DataFrame(
        #     returns_covar,
        #     columns=returns_mean.keys(),
        #     index=returns_mean.keys())

        return df_stocks, df_corr

    def calc_alpha_by_capm(self):
        """
        Calculate alpha as per CAPM
        www.alphagamma.eu/finance/how-to-calculate-alpha-of-your-portfolio/
        TODO: taking a mean of daily_betas
        TODO:
        -------------
        Parameters:
        - None
        Returns:
        - Jensen's Alpha
        """
        pf = self.panelframe
        tb = pd.read_hdf(self.datafile, 'treasury_bills')

        # get portfolio age to estimate durations
        start_date = pf['Open'].index.min()
        end_date = pf['Open'].index.max()
        portfolio_age = end_date - start_date
        portfolio_age = portfolio_age.days

        # get total portfolio return
        # stock_return = self.get_stock_return()
        total_return = self.calc_total_return()
        market_return = self.calc_market_return()

        # get daily betas
        # TODO - account for daily changes of beta
        ptf_beta = self.calc_beta_by_covar()

        if portfolio_age > 365:
            # the code below will give the date when 1yr investement will end
            pos = tb.index.get_loc((start_date + pd.DateOffset(years=1)) +
                                   pd.offsets.MonthBegin(0))
            # now get mean of TB1YR for 1yr+
            treasury_return = tb.iloc[pos:, tb.columns.get_loc('TB1YR')].mean()
        else:
            print("Portfolio age less than 1yr is not yet implemented")
            return -99

        # calculate Jensen's alpha
        # TODO - account for daily changes of beta
        ptf_alpha = total_return - treasury_return - ptf_beta * \
            (market_return - treasury_return)

        self.jensen_alpha = ptf_alpha
        return ptf_alpha, ptf_beta

    # get the portfolio return from stock price increase and dividends
    def calc_total_return(self):
        pf = self.panelframe
        # get total portfolio return
        stock_return = pf['current_total_return'].iloc[-1].sum() /\
            pf['cost_basis'].iloc[-1].sum()
        return stock_return * 100

    # get the market return
    def calc_market_return(self):
        pf = self.panelframe
        start_price = pf['Close', :, 'market'][0]
        end_price = pf['Close', :, 'market'][-1]
        market_return = (end_price - start_price) / start_price
        return market_return * 100

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

    def alpha_and_beta(self, period='monthly'):
        """
        Alpha and beta for all stocks, market and portfolio overall.
        Based on monthly returns
        -------------
        Parameters:
        - period: monthly, quarterly or annual
        Returns:
        - dataframe with alpha, beta
        """
        return None

    def monthly_capital_returns(self):
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

        # # monthly changes in portfolio value
        # # using indirect calculation with mean ratios
        # # TODO - implement a more accurate method
        # ptf_monthly_ratio = pf['current_ratio'].groupby([
        #     lambda x: x.year,
        #     lambda x: x.month]).mean()
        # ptf_monthly_returns = (stock_monthly_return * ptf_monthly_ratio).sum(1)

        # # merge the two
        # stock_monthly_return['portfolio'] = ptf_monthly_returns.values

        return stock_monthly_return

    def monthly_div_returns(self):
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

        # ptf_monthly_ratio = pf['current_ratio'].groupby([
        #     lambda x: x.year,
        #     lambda x: x.month]).mean()
        # ptf_monthly_div_yield = (ptf_monthly_ratio * stock_monthly_div_yield).sum(1)

        # stock_monthly_div_yield['portfolio'] = ptf_monthly_div_yield.values
        return stock_monthly_div_yield

    def monthly_returns(self):
        """
        Monthly returns = capital gain + dividend yields for all symbols
        -------------
        Parameters:
        - none
        Returns:
        - dataframe with monthly returns in % by symbol
        """
        return self.monthly_capital_returns() + self.monthly_div_returns()

    def annual_returns(self):
        """
        Annualized returns for all stocks
        -------------
        Parameters:
        - none
        Returns:
        - merge with other returns
        """
        return emp.annual_return(self.monthly_returns()/100, period='monthly')


if __name__ == '__main__':
    start_date = pd.to_datetime('07/07/2016')
    end_date = pd.to_datetime('07/03/2017')

    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    start_date = df_ord.date.min()
    end_date = df_ord.date.max()

    ptf = PortfolioModels('../data/data.h5')
    pf = ptf.daily_portfolio_changes(start_date, end_date).panelframe
    df1, df2 = ptf.stock_risk_analysis()
