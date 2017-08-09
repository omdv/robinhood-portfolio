import os
import pandas as pd
from backend.backend import BackendClass
from bokeh.charts import TimeSeries
from bokeh.embed import components


# Create the returns figure
def create_figure(data):
    print(data)
    p = TimeSeries(data, x='Date', title='Test', ylabel='Stock Prices')
    return p

if __name__ == '__main__':
    bc = BackendClass(os.path.abspath("data/data.h5"))
    # bc.download_save_data()
    ptf = bc.calculate_portfolio_performance().ptf_daily
    # create figure
    returns = ptf['current_return_raw'].sum(axis=1)
    # returns.reset_index(inplace=True)
    # returns.rename(columns={0: 'Value'}, inplace=True)
    plot = create_figure(dict(Value=returns.values,Date=returns.index.values))
