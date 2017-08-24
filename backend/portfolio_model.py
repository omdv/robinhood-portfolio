# TODO:
# 1. take into account the RB fee when selling position
import numpy as np
import pandas as pd


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
            pf.loc[:, :, key] = df
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
        df_ord['total_cost_basis'] =\
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

    def calc_daily_returns(self, start_date, end_date):
        """
        Calculate daily prices, cost-basis, ratios, returns, etc.
        Used for plotting and also showing the final snapshot of
        the portfolio
        -------------
        Parameters:
        - None
        Return:
        - Panelframe with daily return values. Can be used for plot
        and for html output
        """

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

        # zero out cumulative total_cost_basis
        pf['total_cost_basis'] = pf['total_cost_basis'].\
            where(pf['total_quantity'] != 0).fillna(0)

        pf['current_roi_raw'] =\
            (pf['current_return_raw'] / pf['total_cost_basis']).\
            where(pf['total_quantity'] != 0).fillna(method='ffill')

        # Note that ROI calculation is not representative with dividends
        # because dividends payout is cumulative and may be based on a larger
        # total quantity than current
        pf['current_roi_div'] =\
            (pf['current_return_div'] / pf['total_cost_basis']).\
            where(pf['total_quantity'] != 0)

        # assign to panelframe
        self.panelframe = pf
        return self

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
        betas_dict = dict()
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

    def calc_monthly_properties(self):
        """
        Calculate monthly returns and run all main calculations,
        such as mean returns, std, portfolio mean and standard dev, beta, etc
        References: 
        1. p. 137 of Modern Portfolio Theory and Investment Analysis
        edition 9
        2. https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf
        -------------
        Parameters:
        - None
        Return:
        - Dataframe of properties for each security in portfolio with
        summary row for portfolio
        """
        pf = self.panelframe
        close = pf['Close']

        # get month start
        month_start = close.groupby([lambda x: x.year,\
            lambda x: x.month]).first()
        month_end = close.groupby([lambda x: x.year,\
            lambda x: x.month]).last()
        monthly_change = (month_end - month_start) / month_start * 100

        # get mean values and std by security
        returns_mean = monthly_change.mean(axis=0)
        returns_std = monthly_change.std(axis=0)

        # get covariances
        returns_covar = np.cov(monthly_change.values, rowvar=False, ddof=1)

        # get betas for each stock
        stock_betas = returns_covar[-1] / returns_std['market']**2

        # get alphas for each stock
        # ref [1]
        stock_alphas = returns_mean - 

        # get correlation coefficients
        std_products = np.dot(returns_std.values.reshape(-1,1),
            returns_std.values.reshape(1,-1))
        returns_corr = returns_covar / std_products

        # calculate portofolio values using the last know ratio
        ptf_ratio = pf['current_ratio'].iloc[-1].values
        ptf_std = np.sqrt(np.dot(
            np.dot(ptf_ratio.reshape(1,-1),returns_covar),
            ptf_ratio.reshape(-1,1)))
        ptf_beta = np.dot(stock_betas, ptf_ratio)
        return None

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

    def calc_assets_performance(self):
        """
        Calculate performance of individual assets, i.e. mean return,
        standard deviation, corelations, etc.
        -------------
        Parameters:
        - None
        Returns:
        - None
        """
        return None

    # get the portfolio return from stock price increase
    def calc_stock_return(self):
        pf = self.panelframe
        # get total portfolio return
        stock_return = pf['current_return_raw'].sum(1)[-1] /\
            pf['total_cost_basis'].sum(1)[-1]
        return stock_return * 100

    # get the portfolio return from stock price increase and dividends
    def calc_total_return(self):
        pf = self.panelframe
        # get total portfolio return
        stock_return = pf['current_return_div'].sum(1)[-1] /\
            pf['total_cost_basis'].sum(1)[-1]
        return stock_return * 100

    # get the market return
    def calc_market_return(self):
        pf = self.panelframe
        start_price = pf['Close', :, 'market'][0]
        end_price = pf['Close', :, 'market'][-1]
        market_return = (end_price - start_price) / start_price
        return market_return * 100


if __name__ == '__main__':
    start_date = pd.to_datetime('07/07/2016')
    end_date = pd.to_datetime('07/03/2017')

    df_ord = pd.read_hdf('../data/data.h5', 'orders')
    start_date = df_ord.date.min()
    end_date = df_ord.date.max()

    ptf = PortfolioModels('../data/data.h5')
    pf = ptf.calc_daily_returns(start_date, end_date).panelframe
