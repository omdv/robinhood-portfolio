import os
import pandas as pd
from flask import Flask, render_template, request, jsonify, Response


# Initialize
app = Flask(__name__)
app.config['SECRET_KEY'] = 'NOTSECURELOL'
app.debug = True


# Define a route for the default URL, which loads the login form
@app.route('/')
def form():
    return render_template('dashboard.html')


@app.route('/positions/')
def data():
    # df_div = pd.read_hdf('data/data.h5', 'dividends')
    df_pos = pd.read_hdf('data/data.h5', 'positions')
    # df_ord = pd.read_hdf('data/data.h5', 'orders')

    # get positions
    df_pos = df_pos[df_pos.ratio > 0]
    fields = ['symbol', 'balance', 'subtotal', 'ratio', 'return']
    positions = df_pos[fields].to_json(orient='records')

    return Response(positions, mimetype='application/json')


if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 8080))
    HOST = os.getenv('HOST', '0.0.0.0')
    app.run(debug=True, host=HOST, port=PORT)




# # Accepting: POST requests
# @app.route('/dashboard/', methods=['GET'])
# def account():
#     # username = None
#     # password = None

#     # if request.method == 'POST':
#  #      username=request.form['username']
#  #      password=request.form['password']

#     #   logged_in = robinhood.login(username=username, password=password)

#     #   positions = rb.positions()
#     #   orders = rb.order_history()
#     #   dividends = rb.dividends()

#  #    elif request.method == 'GET':
#  #      name=None
#  #      password=None
#     return render_template('dashboard.html')
