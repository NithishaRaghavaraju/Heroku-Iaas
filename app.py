import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

encoding = {'New York': 0, 'California': 1, 'Florida': 2}

app = Flask(__name__)
model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")
@app.route("/predict",methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    int_features[3] = encoding[int_features[3]]
    int_features = [float(x) for x in int_features]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template("index.html",prediction_text="Profit is {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
