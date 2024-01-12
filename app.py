import flask
import json
from flask import Flask, request, render_template,jsonify
import pickle
import os
import time
import pandas as pd
import numpy as np
from train_generic import train_predict


print("app is started")
# Create flask app
app = Flask(__name__)

app._static_folder = 'static'

@app.route("/")
def home():
    return render_template("index1.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    st=time.time()
    if request.method == 'POST':
        # Get the uploaded CSV file
        uploaded_file = request.files['file']
        num_months = int(request.form.get('NOM'))

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                combined_predictions=train_predict(df,num_months)
                print(combined_predictions)
                return render_template('index1.html', tables=[combined_predictions.to_html(classes='data1', index=False)], titles='Predicted Sales')

            except Exception as e:
                return render_template("index1.html", prediction_text=f"Error: {str(e)}")

        else:
            return render_template("index1.html", prediction_text="Please upload a CSV file.")
    end=time.time()
    print("predictions time",end-time)
    return jsonify(combined_predictions)

if __name__ == '__main__':
    app.run(debug=True)
    
