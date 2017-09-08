import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from flask import Flask, render_template, request
from backend.backend import BackendClass
from io import BytesIO

# Initialize
MY_DPI = 96
sns.set_style("whitegrid")
DATAFILE = 'data/data.h5'
USERFILE = 'data/user.pkl'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'My_l0ng_very_secure_secret_k3y'
app.debug = True


def plot_returns(data):
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(900/MY_DPI, 400/MY_DPI), dpi=MY_DPI)
    x = data.index.values
    y = data.values

    ax.plot(x, y, linewidth=1.0, color='#2c7fb8')
    ax.axhline(y=0, color='#e34a33', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Portfolio returns")
    ax.get_yaxis().set_major_formatter(
        ticker.FuncFormatter(lambda x, p: '${:.0f}'.format(x)))
    ax.grid(False, axis='both', linestyle='-', linewidth=0.5, color="#deebf7")

    # saving and exporting the svg
    with BytesIO() as img_svg:
        f.savefig(img_svg, format='svg', dpi=MY_DPI, bbox_inches='tight')
        figdata_svg =\
            '<svg' + img_svg.getvalue().decode('utf-8').split('<svg')[1]
    plt.close(f)
    return figdata_svg


def plot_heatmap(corr):
    # prepare data
    corr = corr.copy()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(900/MY_DPI, 400/MY_DPI), dpi=MY_DPI)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 20, n=10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    ax = sns.heatmap(
        corr, mask=mask, cmap=cmap, center=0,
        square=False, linewidths=.5,
        cbar_kws={
            "shrink": .8,
            'format': '%.2f'},
        annot=True, fmt=".3f")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # saving and exporting the svg
    with BytesIO() as img_svg:
        f.savefig(img_svg, format='svg', dpi=MY_DPI, bbox_inches='tight')
        figdata_svg =\
            '<svg' + img_svg.getvalue().decode('utf-8').split('<svg')[1]
    plt.close(f)
    return figdata_svg


# default route
@app.route('/', methods=["GET", "POST"])
def portfolio():
    date_fmt = '{:%d-%b-%Y}'
    # initiate backend and get all values
    bc = BackendClass(DATAFILE, USERFILE)
    bc = bc.calculate_all()

    # create plots
    plot_corr_svg = plot_heatmap(bc.stock['corr'])
    plot_returns_svg = plot_returns(bc.portfolio['daily'])

    # handle update of market or robinhood data
    if request.method == 'POST':
        if request.form['refresh'] == 'market':
            bc.update_market_data()
        elif request.form['refresh'] == 'robinhood':
            user = request.form['inputUser']
            password = request.form['inputPassword']
            bc.update_robinhood_data(user, password)

    bc = bc.calculate_all()

    return render_template(
        'pages/portfolio.html',
        plot_returns_svg=plot_returns_svg,
        plot_corr_svg=plot_corr_svg,
        portfolio=bc.portfolio,
        trades=bc.trades,
        stock=bc.stock,
        markowitz=bc.markowitz,
        rb_dates=[date_fmt.format(x) for x in bc.user['rb_dates']],
        mkt_dates=[date_fmt.format(x) for x in bc.user['mkt_dates']],
        today=date_fmt.format(bc.user['today']),
    )


if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    app.run(debug=True, host=HOST, port=PORT)
