import os
import pandas as pd
from flask import Flask, render_template, Response


# Initialize
app = Flask(__name__)
app.config['SECRET_KEY'] = 'NOTSECURELOL'
app.debug = True


# default route
@app.route('/')
def form():
    return render_template('orders.html')


# positions API
@app.route('/positions/')
def positions():
    df_pos = pd.read_hdf('data/data.h5', 'positions')
    df_pos = df_pos[df_pos.pctTotal > 0]
    fields = [
        'symbol', 'value', 'absGain', 'relGain', 'quantity', 'pctTotal',
        'asset_type']
    positions = df_pos[fields].to_json(orient='records')
    return Response(positions, mimetype='application/json')


# orders API
@app.route('/orders/')
def orders():
    df_ord = pd.read_hdf('data/data.h5', 'orders')
    df_ord = df_ord[df_ord.side == "buy"]
    orders = df_ord.to_json(orient='records')
    return Response(orders, mimetype='application/json')


if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    app.run(debug=True, host=HOST, port=PORT)
