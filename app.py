import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response
from backend.backend import BackendClass
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import Span

# Initialize
app = Flask(__name__)
app.config['SECRET_KEY'] = 'My_l0ng_very_secure_secret_k3y'
app.debug = True


# Create the returns figure
def create_figure(data):
    plot = figure(
        x_axis_type='datetime',
        plot_width=800, plot_height=400,
        title='Portfolio returns')
    plot.line(x=data["Date"], y=data["Returns"])
    plot.yaxis.axis_label = 'Returns, $'

    # highlight 0
    zero_returns = Span(
        location=0,
        dimension='width', line_color='red',
        line_dash='dashed', line_width=3)
    plot.add_layout(zero_returns)
    return plot


# generate html dataframe with portfolio summary
def get_portfolio_returns():
    # get current returns
    bc = BackendClass('data/data.h5')
    ptf = bc.calculate_portfolio_performance().ptf_daily
    df = ptf[:, -1, :-1]
    df = df[
        ['total_quantity', 'total_cost_basis', 'total_dividends',
         'current_price', 'current_ratio', 'current_return_raw',
         'current_return_div', 'current_roi_raw', 'current_roi_div']]

    # convert quantities to integer
    df['total_quantity'] = df['total_quantity'].astype(np.int32)

    # convert ratios to percent
    df['current_ratio'] = df['current_ratio'] * 100
    df['current_roi_raw'] = df['current_roi_raw'] * 100
    df['current_roi_div'] = df['current_roi_div'] * 100

    # add Sumamry row
    df = df.copy()  # avoid chained assignment warning
    df.loc['Summary', :] = df.sum(axis=0)
    df.loc['Summary', 'current_roi_raw'] =\
        df.loc['Summary', 'current_return_raw'] /\
        df.loc['Summary', 'total_cost_basis'] * 100
    df.loc['Summary', 'current_roi_div'] =\
        df.loc['Summary', 'current_return_div'] /\
        df.loc['Summary', 'total_cost_basis'] * 100

    # rename for HTML
    df.rename(columns={
        'total_quantity': 'Total shares',
        'total_cost_basis': 'Total cost basis, $',
        'total_dividends': 'Cumulative dividends, $',
        'current_price': 'Current market price, $',
        'current_ratio': 'Portfolio percentage, %',
        'current_return_raw': 'Return, $',
        'current_return_div': 'Return with dividends, $',
        'current_roi_raw': 'ROI w/o dividends, %',
        'current_roi_div': 'ROI with dividends, %'}, inplace=True)

    # create figure
    returns = ptf['current_return_raw'].sum(axis=1)
    plot = create_figure(
        dict(Returns=returns.values, Date=returns.index.values))

    # show(plot)
    html = df.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)
    return html, plot


# default route
@app.route('/')
def portfolio():
    # get current returns
    df_returns_html, plot_returns = get_portfolio_returns()

    # get script and components
    plot_returns_script, plot_returns_div = components(plot_returns)

    return render_template(
        'pages/portfolio.html',
        dataframe_returns=df_returns_html,
        plot_returns_script=plot_returns_script,
        plot_returns_div=plot_returns_div)


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
    ptf = ptf['current_return_div'].to_json(orient='records')
    return Response(ptf, mimetype='application/json')


if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    app.run(debug=True, host=HOST, port=PORT)
