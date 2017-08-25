import os
import pandas as pd
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
def create_returns_figure(data):
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
# def get_portfolio_returns():
#     # get current returns
#     bc = BackendClass('data/data.h5')
#     bc.calc_all_values()
#     panel = bc.panel
#     df_returns = bc.df_returns

#     # create figure
#     returns = panel['current_return_div'].sum(axis=1)
#     plot = create_figure(
#         dict(Returns=returns.values, Date=returns.index.values))

#     # show returns
#     html = df_returns.to_html(
#         float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
#         index=True)
#     return html, plot


# default route
@app.route('/')
def portfolio():
    # initiate backend and get all values
    bc = BackendClass('data/data.h5')
    bc = bc.calc_all_values()
    panel = bc.panel

    # create figure
    returns = panel['current_return_div'].sum(axis=1)
    plot_returns = create_returns_figure(
        dict(Returns=returns.values, Date=returns.index.values))
    plot_returns_script, plot_returns_div = components(plot_returns)

    # returns dataframe
    df_returns_html = bc.df_returns.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)

    df_stocks_html = bc.df_stocks.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)

    return render_template(
        'pages/portfolio.html',
        df_returns=df_returns_html,
        df_stocks=df_stocks_html,
        plot_returns_script=plot_returns_script,
        plot_returns_div=plot_returns_div,
        ptf_jensen_alpha='{:.4f}'.format(bc.jensen_alpha),
        ptf_beta='{:.4f}'.format(bc.beta))


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
