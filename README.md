# Robinhood
Python client to access and analyze the Robinhood portfolio.
Based on [robinhood-api](https://github.com/Jamonek/Robinhood)

##Current Features 

###How To Install:
    pip install -r requirements.txt

###How to Use (see [example.py](https://github.com/MeheLLC/Robinhood/blob/master/example.py))

    from Robinhood import Robinhood
    my_trader = Robinhood()
    logged_in = my_trader.login(username="USERNAME HERE", password="PASSWORD HERE")
    stock_instrument = my_trader.instruments("GEVO")[0]
    quote_info = my_trader.quote_data("GEVO")
    buy_order = my_trader.place_buy_order(stock_instrument, 1)
    sell_order = my_trader.place_sell_order(stock_instrument, 1)

------------------

# Related

* [robinhood-ruby](https://github.com/rememberlenny/robinhood-ruby) - RubyGem for interacting with Robinhood API
* [robinhood-node](https://github.com/aurbano/robinhood-node) - NodeJS module to make trades with Robinhood Private API
