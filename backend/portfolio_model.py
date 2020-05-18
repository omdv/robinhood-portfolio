"""
Porfolio models and calculations
"""
from collections import OrderedDict
from scipy import stats
import numpy as np
import pandas as pd
import empyrical as emp
import portfolioopt as pfopt


# calculating portfolio performance
class PortfolioModels():
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self._daily = None
        self._calculate_daily()
        return None

    def _merge_market_with_orders(self, df_ord, mkt):
        """
        Helper for merging orders with panel frame with market data
        """

        # initialize columns
        mkt['cum_size'] = 0
        mkt['cum_cost_basis'] = 0
        mkt['cum_realized_gain'] = 0

        # loop over tickers, except the last one, which is market
        for _symbol in mkt.index.get_level_values(0).unique()[:-1]:
            df1 = mkt.loc[_symbol]
            df2 = df_ord[df_ord['symbol'] == _symbol].copy()
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

            # propagate values from last observed
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)

            mkt.loc[_symbol] = df.values

        return mkt

    def _merge_market_with_dividends(self, df_div, mkt):
        """
        Helper to merge the market frame with dividends
        """
        # initialize columns
        mkt['cum_dividends'] = 0
        mkt['dividend_rate'] = 0

        # loop over tickers, except the last one, which is market
        for _symbol in mkt.index.get_level_values(0).unique()[:-1]:
            df1 = mkt.loc[_symbol]
            df2 = df_div[df_div['symbol'] == _symbol].copy()
            df2.set_index('date', inplace=True)
            df = pd.merge(
                df1, df2[['cum_dividends', 'rate']],
                left_index=True, right_index=True, how='left')

            df.drop('cum_dividends_x', axis=1, inplace=True)
            df.drop('dividend_rate', axis=1, inplace=True)

            df.rename(columns={
                'cum_dividends_y': 'cum_dividends',
                'rate': 'dividend_rate'}, inplace=True)

            # propagate values from last observed
            df['cum_dividends'].fillna(method='ffill', inplace=True)
            df['cum_dividends'].fillna(0, inplace=True)

            mkt.loc[_symbol] = df.values

        return mkt

    def _calculate_daily(self):
        """
        Calculate daily prices, cost-basis, ratios, returns, etc.
        Used for plotting and also showing the final snapshot of
        the portfolio
        -------------
        Parameters:
        - None
        Return:
        - Multiindex dataframe with daily values
        """
        # read frames for internal use
        market = pd.read_pickle(self.datafolder + "/market.pkl")

        # recreate orders from open and closed pos
        df = pd.concat([
            pd.read_pickle(self.datafolder + "/open.pkl"),
            pd.read_pickle(self.datafolder + "/closed.pkl")]).sort_index()

        # calculate cumulative size and cost basis
        df['cum_size'] =\
            df.groupby('symbol').signed_size.cumsum()

        # cost basis for closed orders is equal to the one for original open
        # position, so cumulative does not include any gains or losses from
        # closing orders
        df['cum_cost_basis'] =\
            df.groupby('symbol').current_cost_basis.cumsum()

        # aggregate orders on the same day
        func = {
            'average_price': np.mean,
            'current_cost_basis': np.sum,
            'current_size': np.sum,
            'fees': np.sum,
            'final_cost_basis': np.sum,
            'final_size': np.sum,
            'signed_size': np.sum,
            'cum_size': np.sum,
            'cum_cost_basis': np.sum,
            'realized_gains': np.sum}
        df = df.groupby(['date', 'symbol'], as_index=False).agg(func)
        # df = pd.pivot_table(df,
        #     values=func.keys(),
        #     index=['symbol', 'date'],
        #     aggfunc=func).reset_index()

        # calculate cumulative size and cost basis
        df['cum_size'] =\
            df.groupby('symbol').signed_size.cumsum()
        df['cum_cost_basis'] =\
            df.groupby('symbol').current_cost_basis.cumsum()
        df['cum_realized_gain'] =\
            df.groupby('symbol').realized_gains.cumsum()

        # fix the average price, so it is weighted mean
        df['average_price'] =\
            df['cum_cost_basis'] / df['cum_size']

        # merge orders with market
        pf = self._merge_market_with_orders(df, market)


        df = pd.read_pickle(self.datafolder + "/dividends.pkl")

        # calculate cumulative dividends
        df['cum_dividends'] = df.groupby('symbol').amount.cumsum()

        # merge orders with market
        pf = self._merge_market_with_dividends(df, pf)

        
        #replace null stock prices using backfill to avoid issues with
        #daily_change and beta calculations
        close_price = pf['close']
        close_price.values[close_price.values == 0] = np.nan
        close_price.fillna(method='bfill', inplace=True)
        pf['close'] = close_price

        # Main daily portfolio properties
        # dividend yield
        pf['dividend_yield'] = pf['dividend_rate'] / pf['close'] * 100

        # cumulative current value of the position for the given security
        # at the start and end of the day
        pf['cum_value_close'] = pf['cum_size'] * pf['close']
        pf['cum_value_open'] = pf['cum_size'] * pf['open']

        # current weight of the given security in the portfolio - matrix
        # based on the close price
        pf['current_weight'] =\
            (pf['cum_value_close'].T /
                pf.groupby(level='date')['cum_value_close'].sum()).T

        # unrealized gain on open positions at the end of day
        pf['cum_unrealized_gain'] =\
            pf['cum_value_close'] - pf['cum_cost_basis']

        # investment return without dividends
        pf['cum_investment_return'] = pf['cum_unrealized_gain'] + \
            pf['cum_realized_gain']

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
        self._daily = pf
        return self

    def _observed_period_portfolio_return(self, _):
        """
        Calculate actual portfolio return over observed period
        """
        pf = self._daily
        res = pf.reset_index().pivot(
                    index='date',
                    columns='symbol',
                    values='cum_total_return').sum(axis=1) / \
            pf.reset_index().pivot(
                        index='date',
                        columns='symbol',
                        values='cum_cost_basis').sum(axis=1)
        return res[-1]

    def _observed_period_market_return(self, _):
        """
        Calculate actual market return over observed period
        """
        pf = self._daily
        return (pf.loc['SPY']['close'][-1] - pf.loc['SPY']['close'][0]) / \
            pf.loc['SPY']['close'][0]

    def _stock_daily_returns(self):
        """
        Estimate daily noncumulative returns for empyrical
        """
        pf = self._daily
        daily = pf.groupby(level='symbol')['close'].\
            transform(lambda x: (x-x.shift(1))/abs(x))
        daily.fillna(0, inplace=True)
        daily = daily.reset_index().pivot(
            index='date',
            columns='symbol',
            values='close')
        return daily

    def _stock_monthly_returns(self):
        """
        Monthly returns = capital gain + dividend yields for all symbols
        -------------
        Parameters:
        - none
        Returns:
        - dataframe with monthly returns in % by symbol
        """
        pf = self._daily

        # monthly changes in stock_prices prices
        # stock_prices = pf['close']
        stock_prices = pf.reset_index().pivot(
            index='date',
            columns='symbol',
            values='close')
        stock_month_start = stock_prices.groupby([
            lambda x: x.year,
            lambda x: x.month]).first()
        stock_month_end = stock_prices.groupby([
            lambda x: x.year,
            lambda x: x.month]).last()
        stock_monthly_return = (stock_month_end - stock_month_start) /\
            stock_month_start * 100

        stock_monthly_div_yield = pf.reset_index().pivot(
            index='date',
            columns='symbol',
            values='dividend_yield').groupby([
                lambda x: x.year,
                lambda x: x.month]).mean()
        stock_monthly_div_yield.fillna(0, inplace=True)

        return stock_monthly_return + stock_monthly_div_yield

    def _ptf_monthly_returns(self):
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
        ptf_monthly_ratio = self._daily.reset_index().pivot(
            index='date',
            columns='symbol',
            values='current_weight').groupby([
                lambda x: x.year,
                lambda x: x.month]).mean()
        ptf_monthly_returns = (
            stock_monthly_change * ptf_monthly_ratio).sum(1)
        return ptf_monthly_returns

    def _one_pfopt_case(self, stock_returns, market, weights, name):
        case = {}
        case['name'] = name
        case['weights'] = weights

        returns = np.dot(stock_returns, weights.values.reshape(-1, 1))
        returns = pd.Series(returns.flatten(), index=market.index)

        simple_stat_funcs = [
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

        factor_stat_funcs = [
            emp.alpha,
            emp.beta,
        ]

        stat_func_names = {
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
        for stat_func in simple_stat_funcs:
            ptf_stats[stat_func_names[stat_func.__name__]] = stat_func(returns)

        for stat_func in factor_stat_funcs:
            res = stat_func(returns, market)
            ptf_stats[stat_func_names[stat_func.__name__]] = res

        case['stats'] = ptf_stats

        return case

    def stocks_risk(self):
        """
        Calculate risk properties for every security in the portfolio
        using `empyrical` library.
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

        daily = self._stock_daily_returns()

        # # construct resulting dataframe
        df = pd.DataFrame({
            'means': daily.mean(axis=0),
        })

        simple_stat_funcs = [
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

        factor_stat_funcs = [
            emp.alpha,
            emp.beta,
        ]

        stat_func_names = {
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

        for stat_func in simple_stat_funcs:
            df[stat_func_names[stat_func.__name__]] =\
                daily.apply(lambda x: stat_func(x)).apply(pd.Series)

        for stat_func in factor_stat_funcs:
            df[stat_func_names[stat_func.__name__]] =\
                daily.apply(lambda x: stat_func(
                    x, daily['SPY'])).apply(pd.Series)

        del df['means']

        # assign for markowitz use
        self.stocks_daily = daily

        return df

    def stocks_correlation(self):
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
        stock_returns['portfolio'] = self._ptf_monthly_returns()

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
        df_covar = df_covar.iloc[:-1, :-1]

        df_corr = pd.DataFrame(
            returns_corr,
            columns=returns_mean.keys(),
            index=returns_mean.keys())

        # assign for markowitz use
        self.stocks_covar = df_covar

        return df_corr, df_covar

    def portfolio_returns(self):
        """
        Calculate portfolio evolution
        total stocks value
        investment returns
        dividend returns
        total returns
        """
        pf = self._daily
        cum_investment_returns = pf.reset_index().pivot(
            index='date',
            columns='symbol',
            values='cum_investment_return').sum(axis=1)

        cum_dividends = pf.reset_index().pivot(
            index='date',
            columns='symbol',
            values='cum_dividends').sum(axis=1)
        return cum_investment_returns, cum_dividends

    def portfolio_summary(self):
        """
        Calculate portfolio composition and summary by stock
        """
        df = self._daily
        df = df.groupby(level='symbol').last()

        columns_to_names = OrderedDict([
            ('cum_size', ['Shares', '{:,.0f}']),
            ('current_weight', ['Portfolio weight', '{:.2f}%']),
            ('cum_cost_basis', ['Current cost basis', '{:,.2f}']),
            ('cum_value_close', ['Current value', '{:,.2f}']),
            ('cum_realized_gain', ['Realized P/L', '{:,.2f}']),
            ('cum_dividends', ['Dividends', '{:,.2f}']),
            ('cum_unrealized_gain', ['Unrealized P/L', '{:,.2f}']),
            ('cum_total_return', ['Total return', '{:,.2f}']),
            ('current_return_rate', ['Total return rate', '{:,.2f}%'])
        ])

        # convert ratios to percent
        df['current_weight'] = df['current_weight'] * 100

        # add total row
        df = df.copy()  # avoid chained assignment warning
        df.loc['Portfolio', :] = df.sum(axis=0)
        df.loc['Portfolio', 'current_return_rate'] =\
            df.loc['Portfolio', 'cum_total_return'] /\
            df.loc['Portfolio', 'cum_cost_basis'] * 100

        # re-order
        df = df[list(columns_to_names.keys())]

        # format
        df = df.apply(
            lambda x: x.map(columns_to_names[x.name][1].format), axis=0)

        # rename columns
        df.columns =\
            df.columns.to_series().apply(lambda x: columns_to_names[x][0])
        return df

    def portfolio_stats(self):
        """
        Calculate actual portfolio stats based on panelframe with daily changes
        -------------
        Parameters:
        - None
        - Uses daily panelframe
        Return:
        - Series with portfolio stats
        TODO: daily returns or returns over cost_basis?
        """
        pf = self._daily

        # capital gains `cum_investment_return` or
        # total return `cum_total_return`
        return_to_use = 'cum_investment_return'

        # cum_return = pf.reset_index().pivot(
        #     index='date',
        #     columns='symbol',
        #     values='cum_investment_return').sum(axis=1)

        # cum_return_D1 = pf[return_to_use].sum(1).shift(1)
        # cum_return_D2 = pf[return_to_use].sum(1)
        # cost_basis = pf['cum_cost_basis'].sum(1)
        # returns = (cum_return_D2 - cum_return_D1) / cost_basis
        # returns.fillna(0, inplace=True)

        # portfolio return over cost_basis
        returns = pf.reset_index().pivot(
            index='date',
            columns='symbol',
            values=return_to_use).sum(axis=1)\
            .transform(lambda x: x-x.shift(1))/pf.reset_index().pivot(
                index='date',
                columns='symbol',
                values='cum_cost_basis').sum(axis=1)
        returns.fillna(0, inplace=True)

        # return of just 100% SPY portfolio
        # m_D1 = pf['close', :, 'market'].shift(1)
        # m_D2 = pf['close', :, 'market']
        # market = (m_D2 - m_D1) / pf['close', :, 'market'].iloc[0]
        market = pf.reset_index().pivot(
            index='date',
            columns='symbol',
            values='close')['SPY'].transform(lambda x: (x-x.shift(1))/x[0])
        market.fillna(0, inplace=True)

        """
        Using empyrical functions
        and re-using code from pyfolio
        """

        simple_stat_funcs = [
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

        factor_stat_funcs = [
            emp.alpha,
            emp.beta,
        ]

        stat_func_names = {
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
        for stat_func in simple_stat_funcs:
            ptf_stats[stat_func_names[stat_func.__name__]] = stat_func(returns)

        for stat_func in factor_stat_funcs:
            res = stat_func(returns, market)
            ptf_stats[stat_func_names[stat_func.__name__]] = res

        return ptf_stats

    def markowitz_portfolios(self):
        """
        Estimate Markowitz portfolios
        Inputs: daily returns by stock, avg returns by stock, cov_matrix
        """
        # pf = self._daily

        # returns = (pf['close'] - pf['close'].shift(1))/pf['close'].shift(1)
        # returns.fillna(0, inplace=True)
        # market = returns['market']
        # returns = returns.iloc[:, :-1]

        # cov_mat = np.cov(returns, rowvar=False, ddof=1)
        # cov_mat = pd.DataFrame(
        #     cov_mat,
        #     columns=returns.keys(),
        #     index=returns.keys())

        # avg_rets = returns.mean(0).astype(np.float64)

        # prepare inputs
        returns = self.stocks_daily
        market = returns['SPY']
        returns.drop('SPY', axis=1, inplace=True)

        avg_rets = returns.mean(0).astype(np.float64)

        cov_mat = self.stocks_covar
        cov_mat.drop('SPY', axis=0, inplace=True)
        cov_mat.drop('SPY', axis=1, inplace=True)

        mrk = []

        weights = pfopt.min_var_portfolio(cov_mat)
        case = self._one_pfopt_case(
            returns, market, weights,
            'Minimum variance portfolio')
        mrk.append(case)

        for t in [0.50, 0.75, 0.90]:
            target = avg_rets.quantile(t)
            weights = pfopt.markowitz_portfolio(cov_mat, avg_rets, target)
            case = self._one_pfopt_case(
                returns, market, weights,
                'Target: more than {:.0f}% of stock returns'.format(t*100))
            mrk.append(case)

        weights = pfopt.tangency_portfolio(cov_mat, avg_rets)
        case = self._one_pfopt_case(
            returns, market, weights,
            'Tangency portfolio')
        mrk.append(case)

        return mrk