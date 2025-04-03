# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import joblib
import xgboost
from sklearn.ensemble import RandomForestClassifier


# Load the trained models
facebook_model = joblib.load("facebook_model.pkl")
#twitter_model = joblib.load("twitter_model.pkl")
#instagram_model = joblib.load("instagram_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/facebook', methods=['GET', 'POST'])
def facebook():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = facebook_model.predict(final_features)
        output = 'Not Fake' if prediction[0] == 1 else 'Fake'
        return render_template('facebook.html', prediction_text='Facebook Prediction: {}'.format(output))
    return render_template('facebook.html')

@app.route('/twitter', methods=['GET', 'POST'])
def twitter():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = twitter_model.predict(final_features)
        output = 'Not Fake' if prediction[0] == 1 else 'Fake'
        return render_template('twitter.html', prediction_text='Twitter Prediction: {}'.format(output))
    return render_template('twitter.html')

@app.route('/instagram', methods=['GET', 'POST'])
def instagram():
    if request.method == 'POST':
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = instagram_model.predict(final_features)
        output = 'Not Fake' if prediction[0] == 1 else 'Fake'
        return render_template('instagram.html', prediction_text='Instagram Prediction: {}'.format(output))
    return render_template('instagram.html')

if __name__ == "__main__":
    app.run(debug=True)
