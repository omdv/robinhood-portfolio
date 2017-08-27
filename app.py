import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, Response
from backend.backend import BackendClass
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.palettes import brewer
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    Span
)
from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import urllib.parse

# Initialize
app = Flask(__name__)
app.config['SECRET_KEY'] = 'My_l0ng_very_secure_secret_k3y'
app.debug = True
MY_DPI = 96


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


# Create the correlation heatmap
def create_correlation_heatmap(data):
    data = data.copy()
    data.values[np.tril_indices_from(data)] = np.nan
    symbols = list(data.index)

    # reshape to 1D array or rates with a month and year for each row.
    df = pd.DataFrame(data.stack()).reset_index()
    df.columns = ['symbolX', 'symbolY', 'R']

    colors = brewer["RdYlBu"][10]
    mapper = LinearColorMapper(palette=colors, low=df.R.min(), high=df.R.max())

    source = ColumnDataSource(df)
    TOOLS = "hover,save"

    p = figure(
        title="Correlations between stocks",
        x_range=symbols, y_range=list(reversed(symbols)),
        x_axis_location="above", plot_width=900, plot_height=400,
        tools=TOOLS, toolbar_location='below')

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.outline_line_color = None

    p.rect(
        x="symbolX", y="symbolY", width=1, height=1,
        source=source,
        fill_color={'field': 'R', 'transform': mapper},
        line_color=None)

    # color bar
    color_bar = ColorBar(
        color_mapper=mapper, major_label_text_font_size="5pt",
        ticker=BasicTicker(desired_num_ticks=len(colors)),
        formatter=PrintfTickFormatter(format="%.2f"),
        label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    p.select_one(HoverTool).tooltips = [
         ('@symbolX vs @symbolY', '@R{%.4f}')]
    return p


def create_heatmap2(corr):
    corr = corr.copy()
    img = BytesIO()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(900/MY_DPI, 400/MY_DPI), dpi=MY_DPI)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5},
                annot=True, fmt=".3f")

    f.savefig(img, format='png', dpi=MY_DPI)
    img.seek(0)
    plot_url = urllib.parse.quote(base64.b64encode(img.read()).decode())
    return plot_url

# default route
@app.route('/')
def portfolio():
    # initiate backend and get all values
    bc = BackendClass('data/data.h5')
    bc = bc.calculate_all()

    # create daily returns figure
    returns = bc.daily_returns
    plot_returns = create_returns_figure(
        dict(Returns=returns.values, Date=returns.index.values))
    plot_returns_script, plot_returns_div = components(plot_returns)

    # create heatmap
    plot_corr = create_correlation_heatmap(bc.df_stock_correlations)
    plot_corr_script, plot_corr_div = components(plot_corr)

    plot_url2 = create_heatmap2(bc.df_stock_correlations)

    # convert dataframes to html
    df_returns_html = bc.df_returns.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)

    df_stock_risk_html = bc.df_stock_risk.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)

    df_stock_corr_html = bc.df_stock_correlations.to_html(
        float_format=lambda x: '{0:.2f}'.format(x) if pd.notnull(x) else 'NA',
        index=True)

    return render_template(
        'pages/portfolio.html',
        df_returns=df_returns_html,
        plot_returns_script=plot_returns_script,
        plot_returns_div=plot_returns_div,
        df_stock_risk=df_stock_risk_html,
        df_stock_corr=df_stock_corr_html,
        plot_corr_div=plot_corr_div,
        plot_corr_script=plot_corr_script,
        plot_url=plot_url2
    )


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
