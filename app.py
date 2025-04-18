# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import xgboost
# Load the trained model
 
 
 # Load the model
model = joblib.load("model.pkl")
 
app = Flask(__name__)
 
@app.route('/')
def home():
 return render_template('index.html')
 
@app.route('/predict', methods=['POST'])
def predict():
 int_features = [int(x) for x in request.form.values()]
 final_features = [np.array(int_features)]
 prediction = model.predict(final_features)
 output = 'Placed' if prediction[0] == 1 else 'Not Placed'
 output = 'Not Fake' if prediction[0] == 1 else 'Fake'
 return render_template('facebook.html', prediction_text='Prediction: {}'.format(output))
 
if __name__ == "__main__":
 app.run(debug=True)
       
