# Note
This app is heavily relying on [pandas-datareader](https://pydata.github.io/pandas-datareader/stable/remote_data.html#) for financial quotes. Over the last couple years multiple APIs were obsoleted by their providers and as I am no longer a RH client I have no time to keep up with those changes. You are welcome to fork and try different sources of quotes, but this repository is _no longer functional as is_.

# Robinhood Portfolio
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
It is recommended to use virtualenv:
```
git clone https://github.com/omdv/robinhood-portfolio && cd robinhood-portfolio
virtualenv robinhood && source robinhood/bin/activate && pip3 install -r requirements.txt
```

### How to Use
```
python3 app.py
```

### Docker container
Docker container based on Ubuntu is [available](https://hub.docker.com/r/omdv/robinhood-portfolio/). To launch it in a background mode:
```
docker run -d -p 8080:8080 --name robinhood omdv/robinhood-portfolio:ubuntu
```

Once up and running connect to [http://localhost:8080](http://localhost:8080). If using the older versions of docker you will need to use the ip of the docker-machine.

To specify a different port run:
```
docker run -d -e PORT=$PORT -p $PORT:$PORT --name robinhood omdv/robinhood-portfolio:ubuntu
```


### Jupyter notebook (WORK IN PROGRESS)
You can find the Jupyter notebook using the backtrader library with pyfolio in "notebooks" folder.


### Disclaimer
This tool uses the unofficial Robinhood API to access your account. This code and the corresponding tools are provided on "as is" basis and the user is responsible for the safety of his/her own account.

------------------

# Related
* [empyrical](https://github.com/quantopian/empyrical)
* [portfolioopt](https://github.com/czielinski/portfolioopt)
* [robinhood-api](https://github.com/Jamonek/Robinhood)
* [backtrader](https://github.com/mementum/backtrader)
* [pyfolio](https://github.com/quantopian/pyfolio)
