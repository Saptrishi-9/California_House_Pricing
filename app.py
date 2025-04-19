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


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform([data])
    output = model.predict(final_input)[0]
    return render_template('home.html', prediction_text=f"Predicted House Price: {output}")


if __name__ =="__main__":
    app.run(debug=True)