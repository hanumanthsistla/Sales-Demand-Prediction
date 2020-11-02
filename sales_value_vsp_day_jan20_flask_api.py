#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:37:06 2020

@author: hduser
"""
import numpy as np

import pandas as pd

from datetime import date

from flask import Flask, request, redirect, url_for, flash, jsonify

from vsp_sales_day_feature_calculation_jan20 import doTheCalculation

import json, pickle


app = Flask(__name__)

@app.route('/makecalc/', methods=['POST'])
def makecalc():
    """
     Function run at each API call
    """
    jsonfile = request.get_json()

    data = pd.read_json(json.dumps(jsonfile),orient='index',convert_dates=['BILLING_DATE'])

    print(data)

    res = dict()

    ypred = model.predict(doTheCalculation(data))

    for i in range(len(ypred)):

        res[i] = ypred[i] 

    return jsonify(res) 

if __name__ == '__main__':

    modelfile = 'F:/ML_Project_April_2020/SD_Sales_Predict_ML_Projects/Model_Files/sales_vsp_day_jan20_value.pickle'

    model = pickle.load(open(modelfile, 'rb'))

    print("sales Prediction Pickle Predict Model is Loaded: Flask Test Server is Running on port:8002")

def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=8002, debug=True)  # nosec

if __name__ == '__main__':
    main()


