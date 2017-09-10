# Robinhood
Python client to access and analyze the Robinhood portfolio.
Based on unofficial [robinhood-api](https://github.com/Jamonek/Robinhood) and several python libraries for financial analysis, such as:
- [empyrical](https://github.com/quantopian/empyrical)
- [portfolioopt](https://github.com/czielinski/portfolioopt)

## Current Features 
- Creates a Flask web server with lightweight page
- Downloads orders and dividends from Robinhood account.
- Downloads market data from google API and market index from open source. Supports incremental download for additional dates to reduce a number of requests to open APIs.
- Calculates the total return of the portfolio, including dividend payouts and risk metric for the portfolio
- Calculates the risk metric for individual securities and correlations between securities
- Calculates Markowitz portfolios

## Screenshots
![Image1](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_1.png)
![Image2](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_2.png)
![Image3](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_3.png)
![Image4](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_4.png)
![Image5](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_5.png)


## Future Possible Features
- Backtesting using one of existing libraries. Enabled, but not implemented because none of the existing libraries support dividend payouts and do not provide any significant advantages vs what is implemented already.
- Automatic trading for simple portfolio allocations

### How To Install:
    pip install -r requirements.txt

### How to Use
	python3 app.py

### Docker container
Coming soon...

### Jupyter notebook
Coming soon...


------------------

# Related

* [empyrical](https://github.com/quantopian/empyrical)
* [portfolioopt](https://github.com/czielinski/portfolioopt)
* [robinhood-api](https://github.com/Jamonek/Robinhood)
* [backtrader](https://github.com/mementum/backtrader)
* [pyfolio](https://github.com/quantopian/pyfolio)
