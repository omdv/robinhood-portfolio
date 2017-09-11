FROM blitznote/debootstrap-amd64:16.04

RUN apt-get update && \
    apt-get install -y \
    python3 python3-numpy python3-nose python3-pandas \
    pep8 python3-pip python3-matplotlib git openssl && \
    pip3 install --upgrade setuptools

RUN git clone https://github.com/omdv/robinhood-portfolio && \
	pip3 install --upgrade --force-reinstall -r robinhood-portfolio/requirements.txt

RUN openssl s_client -showcerts -connect stooq.com:443 \
	</dev/null 2>/dev/null|openssl x509 -outform PEM > stooq.pem && \
	mkdir -p /root/.config/matplotlib && \
	echo backend:Agg > /root/.config/matplotlib/matplotlibrc

CMD cd robinhood-portfolio && python3 app.py