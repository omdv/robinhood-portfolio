import pandas as pd
import backtrader as bt
from datetime import datetime

class RobinhoodHistory(bt.Strategy):
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.df_ord = pd.read_hdf('../data/data.h5', 'orders')
        self.pf = pd.read_hdf('../data/data.h5', 'market')
        self.symbols = list(self.pf.minor_axis.values)
        self.df_ord['dt'] = self.df_ord['date'].apply(lambda x: x.to_pydatetime().date())

    def next(self):
        # Simply log the closing price of the series from the reference
        dt = self.datas[0].datetime.date(0)
        if dt in self.df_ord['dt'].values:
            orders = self.df_ord[self.df_ord['dt'] == dt]
            order = orders.iloc[0]
            symbol = order.symbol
            datanum = self.symbols.index(order.symbol)
            price = order.average_price
            size = order.cumulative_quantity
            self.buy(
                data=self.datas[datanum],
                price=price,
                size=size)
            self.log('BUY CREATE {}/{}: {} @ {}'.format(symbol, datanum, size, price))


cerebro = bt.Cerebro()
cerebro.addstrategy(RobinhoodHistory)

pf = pd.read_hdf('../data/data.h5','market')
pf["OpenInterest"] = 0

is_first = True

for key in pf.minor_axis:
    data = bt.feeds.PandasData(dataname=pf.minor_xs(key), name=key)
    cerebro.adddata(data)

cerebro.broker.setcash(100000.0)

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
cerebro.run()

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
