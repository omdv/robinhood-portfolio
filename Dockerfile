FROM jupyter/base-notebook:latest
MAINTAINER Oleg Medvedev <omdv.public@gmail.com>

USER root

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/omdv/robinhood-portfolio ./work && \
	pip3 install --upgrade --force-reinstall -r work/requirements.txt

USER $NB_USER