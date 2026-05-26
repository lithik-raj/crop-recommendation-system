from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv("data.csv")

# Features and target
X = data[['Nitrogen', 'Phosphorous', 'Potassium',
          'Temperature', 'Humidity', 'Rainfall']]

y = data['Crop']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

# Load model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    nitrogen = float(request.form['Nitrogen'])
    phosphorous = float(request.form['Phosphorous'])
    potassium = float(request.form['Potassium'])
    temperature = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    rainfall = float(request.form['Rainfall'])

    prediction = model.predict([[nitrogen, phosphorous,
                                 potassium, temperature,
                                 humidity, rainfall]])

    result = prediction[0]

    return render_template('result.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)