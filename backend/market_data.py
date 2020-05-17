"""
Market reader module
"""
import pandas_datareader.data as web

def download_save_market_data(api_key, symbols, start_date, end_date):
    """
    Return market data

    Keyword arguments:
        api_key: API for market data source
        symbols: list of symbols, index will be appended
        start_date: pd.Timestamp
        end_date: pd.Timestamp
    """
    # Downloading prices for all symbols and SPY for market baseline
    # will append if SPY is in symbols list to have index in last column
    symbols = list(symbols)
    symbols.append("SPY")

    market = web.get_data_tiingo(
        api_key=api_key,
        symbols=symbols,
        start=start_date,
        end=end_date)

    market.to_pickle("data/market.pkl")
    return market
