from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and preprocessor
try:
    model, preprocessor = joblib.load('random_forest_model.pkl')  # Adjust the path as needed
except Exception as e:
    print(f"Error loading model: {e}")

def convert_range_to_numeric(value):
    try:
        if '-' in value:
            parts = value.split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        return float(value)
    except ValueError as e:
        print(f"Value conversion error: {e}")
        return np.nan

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        
        # Ensure numeric columns are correctly formatted
        df['Car_Year'] = pd.to_numeric(df['Car_Year'], errors='coerce')
        # df['Distance'] = df['Distance']
        
        # Debugging: check the dataframe before transformation
        print("DataFrame before transformation:", df)
        
        # Check for NaN values and handle them (e.g., by filling or dropping)
        if df.isnull().values.any():
            print("Data contains NaN values. Filling with 0.")
            df = df.fillna(0)
        
        # Preprocess the input data
        X = preprocessor.transform(df)
        
        # Debugging: check the transformed data
        print("Transformed Data:", X)
        
        # Make prediction
        prediction = model.predict(X)
        
        # Debugging: check the prediction
        print("Prediction:", prediction)
        
        predicted_price = int(round(prediction[0] / 10) * 10)


        
        return render_template('result.html', price=predicted_price)
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('result.html', price=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)
