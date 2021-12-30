# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 20:53:27 2021

@author: saibh
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#Home - It renders index.html on root page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]
    #prediction_text is replaces the variable in index.html
    return render_template('index.html', prediction_text='1-Buy 0-Not buy {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)