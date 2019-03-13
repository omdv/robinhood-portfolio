FROM phusion/baseimage:latest
MAINTAINER Oleg Medvedev <ole.bjorne@gmail.com>
ARG VERSION=unknown

RUN apt-get update && \
    apt-get install -y \
    python3 python3-numpy python3-nose python3-pandas \
    pep8 python3-pip python3-matplotlib git && \
    pip3 install --upgrade setuptools

RUN mkdir -p /root/.config/matplotlib && \
	echo backend:Agg > /root/.config/matplotlib/matplotlibrc

RUN git clone https://github.com/Jamonek/Robinhood && \
	pip3 install Robinhood/

RUN VERSION=${VERSION} git clone https://github.com/omdv/robinhood-portfolio && \
	pip3 install --upgrade --force-reinstall -r robinhood-portfolio/requirements.txt

CMD cd robinhood-portfolio && python3 app.py