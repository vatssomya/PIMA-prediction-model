# Importing necessary libraries
import os

    
import numpy as np
import flask
import joblib
from flask import Flask, render_template, request

# Creating an instance of the Flask class
app = Flask(__name__)

# Routes for index and home
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/printmsg')
def msg():
    return flask.render_template('msg.html')

# Prediction function using the model
def ValuePredictor(to_predict_list):
    # Reshape for the model (expecting 8 features from the PIMA dataset)
    to_predict = np.array(to_predict_list).reshape(1, 8)
    
    # Loading the pre-trained model (new name: best_model.sav)
    loaded_model = joblib.load('best_model.sav')  # Ensure this file is in the same directory
    
    # Perform the prediction
    result = loaded_model.predict(to_predict)
    return result[0]  # Returning the result

# Route to process the form data and provide predictions
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        # Extracting the form data
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        
        # Convert input values to floats (since PIMA dataset values are numeric)
        to_predict_list = list(map(float, to_predict_list))
        
        # Predict the result using the predictor function
        result = ValuePredictor(to_predict_list)
        
        # Display 'Diabetes' or 'No Diabetes' based on prediction result (1 for positive, 0 for negative)
        result_text = "Diabetes" if result == 1 else "No Diabetes"
        
        # Render the result.html page and pass the prediction
        return render_template("result.html", prediction=result_text)

# Running the Flask app
if __name__ == '__main__':
    app.run(debug=True)
