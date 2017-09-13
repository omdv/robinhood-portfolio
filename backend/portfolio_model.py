# TODO:
# 1. take into account the RB fee when selling position
import numpy as np
import pandas as pd
import empyrical as emp
import portfolioopt as pfopt
from scipy import stats


# calculating portfolio performance
class PortfolioModels():
    def __init__(self, datafile):
        self.datafile = datafile
        self.panelframe = None
        self.APPROX_BDAYS_PER_YEAR = 252
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

        '''
        replace null stock prices using backfill to avoid issues with
        daily_change and beta calculations
        '''
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

        '''
        cumulative current value of the position for the given security
        at the start and end of the day
        '''
        pf['cum_value_close'] = pf['cum_size'] * pf['Close']
        pf['cum_value_open'] = pf['cum_size'] * pf['Open']

        # current weight of the given security in the portfolio - matrix
        # based on the close price
        pf['current_weight'] =\
            (pf['cum_value_close'].T / pf['cum_value_close'].sum(axis=1)).T

        # unrealized gain on open positions at the end of day
        pf['cum_unrealized_gain'] =\
            pf['cum_value_close'] - pf['cum_cost_basis']

        # total return
        pf['cum_total_return'] = pf['cum_unrealized_gain'] +\
            pf['cum_dividends'] + pf['cum_realized_gain']

        # return from price change only
        pf['cum_price_return'] = pf['cum_unrealized_gain']

        # calculate ROI
        pf['current_return_rate'] =\
            (pf['cum_total_return'] / pf['cum_cost_basis'] * 100).\
            where(pf['cum_size'] != 0).fillna(method='ffill')

        # assign to panelframe
        self.panelframe = pf
        return self

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
        stock_returns = self._stock_monthly_returns()
        stock_returns['portfolio'] = self._ptf_monthly_returns_indirect()

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

        df_covar = pd.DataFrame(
            returns_covar,
            columns=returns_mean.keys(),
            index=returns_mean.keys())
        df_covar = df_covar.iloc[:-2, :-2]

        df_corr = pd.DataFrame(
            returns_corr,
            columns=returns_mean.keys(),
            index=returns_mean.keys())

        return df_corr, df_covar

    def _observed_period_portfolio_return(self, _):
        """
        Calculate actual portfolio return over observed period
        """
        pf = self.panelframe
        ptf_return = pf['cum_total_return'].sum(1).iloc[-1] /\
            pf['cum_cost_basis'].sum(1).iloc[-1]
        return ptf_return

    def _observed_period_market_return(self, _):
        """
        Calculate actual market return over observed period
        """
        pf = self.panelframe
        market_prices = pf['Close', :, 'market']
        market_return = (market_prices[-1] - market_prices[0]) /\
            market_prices[0]
        return market_return

    def actual_portfolio_stats(self):
        """
        Calculate actual portfolio stats based on panelframe with daily changes
        -------------
        Parameters:
        - None
        - Uses daily panelframe
        Return:
        - Series with portfolio stats
        """
        pf = self.panelframe

        # can choose either a total return or capital gain only
        return_to_use = 'cum_total_return'

        cum_return_D1 = pf[return_to_use].sum(1).shift(1)
        cum_return_D2 = pf[return_to_use].sum(1)
        cost_basis = pf['cum_cost_basis'].sum(1)
        returns = (cum_return_D2 - cum_return_D1) / cost_basis
        returns.fillna(0, inplace=True)

        m_D1 = pf['Close', :, 'market'].shift(1)
        m_D2 = pf['Close', :, 'market']
        market = (m_D2 - m_D1) / pf['Close', :, 'market'].iloc[0]
        market.fillna(0, inplace=True)

        """
        Using empyrical functions
        and re-using code from pyfolio
        """

        SIMPLE_STAT_FUNCS = [
            self._observed_period_portfolio_return,
            self._observed_period_market_return,
            emp.annual_return,
            emp.annual_volatility,
            emp.sharpe_ratio,
            emp.calmar_ratio,
            emp.stability_of_timeseries,
            emp.max_drawdown,
            emp.omega_ratio,
            emp.sortino_ratio,
            stats.skew,
            stats.kurtosis,
            emp.tail_ratio,
            emp.value_at_risk,
        ]

        FACTOR_STAT_FUNCS = [
            emp.alpha,
            emp.beta,
        ]

        STAT_FUNC_NAMES = {
            '_observed_period_portfolio_return': 'Total return',
            '_observed_period_market_return': 'Market return',
            'annual_return': 'Annual return',
            'cum_returns_final': 'Cumulative returns',
            'annual_volatility': 'Annual volatility',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'sharpe_ratio': 'Sharpe ratio',
            'calmar_ratio': 'Calmar ratio',
            'stability_of_timeseries': 'Stability',
            'max_drawdown': 'Max drawdown',
            'omega_ratio': 'Omega ratio',
            'sortino_ratio': 'Sortino ratio',
            'tail_ratio': 'Tail ratio',
            'value_at_risk': 'Daily value at risk',
            'skew': 'Skew',
            'kurtosis': 'Kurtosis'
        }

        ptf_stats = pd.Series()
        for stat_func in SIMPLE_STAT_FUNCS:
            ptf_stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

        for stat_func in FACTOR_STAT_FUNCS:
            res = stat_func(returns, market)
            ptf_stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

        return ptf_stats

    def stock_risk_analysis(self, if_risk_free_return=False):
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

        pf = self.panelframe
        returns = (pf['Close'] - pf['Close'].shift(1))/pf['Close'].iloc[0]
        returns.fillna(0, inplace=True)

        # construct resulting dataframe
        df = pd.DataFrame({
            'means': returns.mean(axis=0),
        })

        SIMPLE_STAT_FUNCS = [
            emp.annual_return,
            emp.annual_volatility,
            emp.sharpe_ratio,
            emp.calmar_ratio,
            emp.stability_of_timeseries,
            emp.max_drawdown,
            emp.omega_ratio,
            emp.sortino_ratio,
            stats.skew,
            stats.kurtosis,
            emp.tail_ratio,
            emp.value_at_risk,
        ]

        FACTOR_STAT_FUNCS = [
            emp.alpha,
            emp.beta,
        ]

        STAT_FUNC_NAMES = {
            'annual_return': 'Annual return',
            'cum_returns_final': 'Cumulative returns',
            'annual_volatility': 'Annual volatility',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'sharpe_ratio': 'Sharpe ratio',
            'calmar_ratio': 'Calmar ratio',
            'stability_of_timeseries': 'Stability',
            'max_drawdown': 'Max drawdown',
            'omega_ratio': 'Omega ratio',
            'sortino_ratio': 'Sortino ratio',
            'tail_ratio': 'Tail ratio',
            'value_at_risk': 'Daily value at risk',
            'skew': 'Skew',
            'kurtosis': 'Kurtosis'
        }

        for stat_func in SIMPLE_STAT_FUNCS:
            df[STAT_FUNC_NAMES[stat_func.__name__]] =\
                returns.apply(lambda x: stat_func(x)).apply(pd.Series)

        for stat_func in FACTOR_STAT_FUNCS:
            df[STAT_FUNC_NAMES[stat_func.__name__]] =\
                returns.apply(lambda x: stat_func(
                    x, returns['market'])).apply(pd.Series)

        del df['means']

        return df

    def _risk_free_return(self, period='monthly'):
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

    def _stock_monthly_returns(self):
        """
        Monthly returns = capital gain + dividend yields for all symbols
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

        stock_monthly_div_yield = pf['dividend_yield'].groupby([
            lambda x: x.year,
            lambda x: x.month]).mean()
        stock_monthly_div_yield.fillna(0, inplace=True)

        return stock_monthly_return + stock_monthly_div_yield

    def _ptf_monthly_returns_indirect(self):
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
        stock_monthly_change = self._stock_monthly_returns()
        ptf_monthly_ratio = self.panelframe['current_weight'].groupby([
            lambda x: x.year,
            lambda x: x.month]).mean()
        ptf_monthly_returns = (
            stock_monthly_change * ptf_monthly_ratio).sum(1)
        return ptf_monthly_returns

    def _one_pfopt_case(self, cov_mat, stock_returns, market, weights, name):
        case = {}
        case['name'] = name
        case['weights'] = weights

        returns = np.dot(stock_returns, weights.values.reshape(-1, 1))
        returns = pd.Series(returns.flatten(), index=market.index)

        SIMPLE_STAT_FUNCS = [
            emp.annual_return,
            emp.annual_volatility,
            emp.sharpe_ratio,
            emp.stability_of_timeseries,
            emp.max_drawdown,
            emp.omega_ratio,
            emp.calmar_ratio,
            emp.sortino_ratio,
            emp.value_at_risk,
        ]

        FACTOR_STAT_FUNCS = [
            emp.alpha,
            emp.beta,
        ]

        STAT_FUNC_NAMES = {
            'annual_return': 'Annual return',
            'annual_volatility': 'Annual volatility',
            'alpha': 'Alpha',
            'beta': 'Beta',
            'sharpe_ratio': 'Sharpe ratio',
            'calmar_ratio': 'Calmar ratio',
            'stability_of_timeseries': 'Stability',
            'max_drawdown': 'Max drawdown',
            'omega_ratio': 'Omega ratio',
            'sortino_ratio': 'Sortino ratio',
            'value_at_risk': 'Daily value at risk',
        }

        ptf_stats = pd.Series()
        for stat_func in SIMPLE_STAT_FUNCS:
            ptf_stats[STAT_FUNC_NAMES[stat_func.__name__]] = stat_func(returns)

        for stat_func in FACTOR_STAT_FUNCS:
            res = stat_func(returns, market)
            ptf_stats[STAT_FUNC_NAMES[stat_func.__name__]] = res

        case['stats'] = ptf_stats

        return case

    def markowitz_portfolios(self):
        pf = self.panelframe
        returns = (pf['Close'] - pf['Close'].shift(1))/pf['Close'].shift(1)
        returns.fillna(0, inplace=True)
        market = returns['market']
        returns = returns.iloc[:, :-1]

        cov_mat = np.cov(returns, rowvar=False, ddof=1)
        cov_mat = pd.DataFrame(
            cov_mat,
            columns=returns.keys(),
            index=returns.keys())

        avg_rets = returns.mean(0).astype(np.float64)

        mrk = []

        weights = pfopt.min_var_portfolio(cov_mat)
        case = self._one_pfopt_case(
            cov_mat, returns, market, weights, 'Minimum variance portfolio')
        mrk.append(case)

        for t in [0.50, 0.75, 0.90]:
            target = avg_rets.quantile(t)
            weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target)
            case = self._one_pfopt_case(
                cov_mat, returns, market, weights,
                'Target: more than {:.0f}% of stock returns'.format(t*100))
            mrk.append(case)

        weights = pfopt.tangency_portfolio(cov_mat, avg_rets)
        case = self._one_pfopt_case(
            cov_mat, returns, market, weights, 'Tangency portfolio')
        mrk.append(case)

        return mrk


if __name__ == '__main__':
    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    df_div = pd.read_hdf('../data/data.h5', 'dividends')
    df_open = pd.read_hdf('../data/data.h5', 'open')
    df_closed = pd.read_hdf('../data/data.h5', 'closed')
    pf = pd.read_hdf('../data/data.h5', 'market')

    ptf = PortfolioModels('../data/data.h5')
    pf = ptf.daily_portfolio_changes().panelframe

    # this section uses only stock prices, div yields and weights
    df_risk = ptf.stock_risk_analysis(False)
    df_corr, df_cov = ptf.stock_correlation_matrix()
    pf_stats = ptf.actual_portfolio_stats()
    mrk = ptf.markowitz_portfolios()
