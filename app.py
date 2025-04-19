import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
pickled_model = pickle.load(open('regmodel.pkl', 'rb'))
model = pickled_model['model']
scaler = pickled_model['scaler']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():
    data = request.json['data']
    input_array = np.array(list(data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    output = model.predict(input_scaled)
    return jsonify({'prediction': output[0]})


if __name__ =="__main__":
    app.run(debug=True)