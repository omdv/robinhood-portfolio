import os
import pandas as pd
from flask import Flask, render_template, Response
from backend.backend import BackendClass


# Initialize
app = Flask(__name__)
app.config['SECRET_KEY'] = 'My_l0ng_very_secure_secret_k3y'
app.debug = True


# generate html dataframe with portfolio summary
def get_portfolio_returns():
    # get current returns
    bc = BackendClass('data/data.h5')
    ptf = bc.calculate_portfolio_performance().ptf_daily
    df = ptf[:, -1, :-1]
    df = df[
        ['total_quantity', 'total_cost_basis', 'total_dividends',
         'current_price', 'current_ratio', 'current_return_raw',
         'current_return_div', 'current_roi']]

    html = df.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)
    return html


# default route
@app.route('/')
def portfolio():
    # get current returns
    return_html = get_portfolio_returns()

    return render_template(
        'pages/portfolio.html',
        orders=None,
        returns=return_html)


# orders API
@app.route('/api/orders/')
def orders():
    df_ord = pd.read_hdf('data/data.h5', 'orders')
    orders = df_ord.to_json(orient='records')
    return Response(orders, mimetype='application/json')


# orders API
@app.route('/api/returns/')
def returns():
    # Run calculations
    bc = BackendClass('data/data.h5')
    ptf = bc.calculate_portfolio_performance().ptf_daily
    ptf = ptf['current_return_div'].to_json(orient='records', date_format='iso')
    return Response(ptf, mimetype='application/json')


if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    app.run(debug=True, host=HOST, port=PORT)
