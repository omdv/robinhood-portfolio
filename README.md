# Note
This app is heavily relying on [pandas-datareader](https://pydata.github.io/pandas-datareader/stable/remote_data.html#) for financial quotes. Over the last couple years multiple APIs were obsoleted by their providers (Google, Morningstar) and as I am no longer a RH client I have no time to keep up with those changes. If you encounter the "Bad Gateway" error or similar it is likely that the current market data source is no longer valid. You are welcome to fork and try different sources of quotes - I will try to fix it, when/if I have time.

**Current API is [TIINGO](https://api.tiingo.com/account/token) for stock and market index.**

# Robinhood Portfolio
Python client to access and analyze the Robinhood portfolio.
Based on unofficial [robinhood-api](https://github.com/Jamonek/Robinhood) and several python libraries for financial analysis, such as:
- [empyrical](https://github.com/quantopian/empyrical)
- [portfolioopt](https://github.com/czielinski/portfolioopt)

## Current Features 
- ~~Creates a Flask web server with lightweight page~~ Replaced with Jupyter notebook
- Downloads orders and dividends from Robinhood account.
- Downloads market data from google API and market index from open source. Supports incremental download for additional dates to reduce a number of requests to open APIs.
- Calculates the total return of the portfolio, including dividend payouts and risk metric for the portfolio
- Calculates the risk metric for individual securities and correlations between securities
- Calculates Markowitz portfolios

## Screenshots (OLD FLASK APP)
![Image1](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_1.png)
![Image2](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_2.png)
![Image3](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_3.png)
![Image4](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_4.png)
![Image5](https://github.com/omdv/robinhood-portfolio/blob/master/docs/image_5.png)


### Non-Docker way
Install dependencies, it is recommended to use virtualenv:
```
git clone https://github.com/omdv/robinhood-portfolio && cd robinhood-portfolio
virtualenv robinhood && source robinhood/bin/activate && pip3 install -r requirements.txt
```

To run:
```
jupyter notebook
```

Open `main.ipynb`, enter TIINGO API KEY, Robinhood credentials, set DEMO_RUN variable to `False` to run with your data, execute all cells.


<!-- ### Docker way
Docker container based on Ubuntu is [available](https://hub.docker.com/r/omdv/robinhood-portfolio/). To launch it in a background mode you need to get TIINGO API key and provide it to docker.
```
docker run -e TIINGO_API_KEY=<API-KEY> -d -p 8080:8080 --name robinhood omdv/robinhood-portfolio:ubuntu
```

Once up and running connect to [http://localhost:8080](http://localhost:8080). If using the older versions of docker you will need to use the ip of the docker-machine.

To specify a different port run:
```
docker run -e TIINGO_API_KEY=<API-KEY> -d -e PORT=$PORT -p $PORT:$PORT --name robinhood omdv/robinhood-portfolio:ubuntu
``` -->


### Disclaimer
This tool uses the unofficial Robinhood API to access your account. This code and the corresponding tools are provided on "as is" basis and the user is responsible for the safety of his/her own account.

------------------

# Related
* [empyrical](https://github.com/quantopian/empyrical)
* [portfolioopt](https://github.com/czielinski/portfolioopt)
* [robinhood-api](https://github.com/Jamonek/Robinhood)
* [backtrader](https://github.com/mementum/backtrader)
* [pyfolio](https://github.com/quantopian/pyfolio)
