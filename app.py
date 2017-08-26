import os
import pandas as pd
from flask import Flask, render_template, Response
from backend.backend import BackendClass
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    ColorBar,
    Span
)

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


# Create the correlation heatmap
def create_correlation_heatmap(data):
    symbols = list(data.index)

    # reshape to 1D array or rates with a month and year for each row.
    df = pd.DataFrame(data.stack()).reset_index()
    df.columns = ['symbolX', 'symbolY', 'R']

    # this is the colormap from the original NYTimes plot
    colors = [
        "#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1",
        "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=df.R.min(), high=df.R.max())

    source = ColumnDataSource(df)
    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom"

    plot = figure(
        title="TEST",
        x_range=symbols, y_range=list(reversed(symbols)),
        x_axis_location="above", plot_width=900, plot_height=400,
        tools=TOOLS, toolbar_location='below')

    plot.grid.grid_line_color = None
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    plot.axis.major_label_text_font_size = "5pt"
    plot.axis.major_label_standoff = 0
    # plot.xaxis.major_label_orientation = pi / 3

    plot.rect(
        x="symbolX", y="symbolY", width=1, height=1,
        source=source,
        fill_color={'field': 'R', 'transform': mapper},
        line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%d%%"),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    plot.add_layout(color_bar, 'right')

    plot.select_one(HoverTool).tooltips = [
         ('pair', '@symbolX @symbolY'),
         ('correlation', '@R'),
    ]
    return plot

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
        plot_corr_script=plot_corr_script
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
