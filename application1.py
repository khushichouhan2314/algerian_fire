from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

application1 = Flask(__name__)
app = application1

# Load the Ridge Regressor model and Standard Scaler
ridge_model = pickle.load(open('model/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('model/scaler.pkl', 'rb'))

@app.route("/")
def welcome():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST']) # in get we are just retrieving and in post we are sending the data
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        BUI = float(request.form.get('BUI'))
        DC = float(request.form.get('DC'))

        # Transform the input data
        scaled_data = standard_scaler.transform([[Temperature, RH, WS, Rain, FFMC, DMC, ISI, Classes, Region , BUI , DC]])  # to add two omore features fwi bui 
        result = ridge_model.predict(scaled_data)
        return render_template('home.html', results=result[0])

        
        

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0"  , debug = True)  