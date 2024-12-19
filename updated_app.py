from flask import Flask, render_template, request, redirect, url_for
import os
import json
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('./model', exist_ok=True)

# Load the pre-trained LSTM model
try:
    model = load_model('./model/lstm_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")



# Route: Feedback Page
@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    feedback_file = app.config['FEEDBACK_FILE']

    if request.method == 'POST':
        # Get user feedback
        user_name = request.form.get('name', 'Anonymous')
        user_feedback = request.form.get('feedback', '')

        # Save feedback to a file
        feedback_data = []
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r') as f:
                feedback_data = json.load(f)

        feedback_data.append({"name": user_name, "feedback": user_feedback})

        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=4)

        return redirect(url_for('feedback'))

    # Load feedback for display
    feedback_data = []
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            feedback_data = json.load(f)

    return render_template('feedback.html', feedbacks=feedback_data)

# Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['dataset']
        if file:
            filepath = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
            file.save(filepath)
            print(f"File uploaded successfully: {filepath}")
            return redirect(url_for('predict', filename=file.filename))
    return render_template('upload.html')


@app.route('/predict/<filename>', methods=['GET', 'POST'])
def predict(filename):
    if model is None:
        return "Error: Model not loaded. Ensure the model file is in './model/lstm_model.h5'."

    # Load dataset
    filepath = f"{app.config['UPLOAD_FOLDER']}/{filename}"
    try:
        data = pd.read_csv(filepath)
        print(f"Dataset loaded successfully from {filepath}.")
    except Exception as e:
        return f"Error loading dataset: {e}"

    # Define features
    features = ['Burgers', 'Pizzas', 'Salads'] + [
        f'{item}_lag_{i}' for item in ['Burgers', 'Pizzas', 'Salads'] for i in range(1, 8)
    ]

    # Validate dataset columns
    if not all(feature in data.columns for feature in features):
        return f"Error: Dataset is missing required features. Expected columns: {features}"

    # Validate sufficient data for prediction
    if len(data) < 7:
        return "Error: Insufficient data. At least 7 rows of data are required for prediction."

    # Prepare recent data for prediction
    recent_data = data[features].iloc[-7:].values  # Last 7 days of data
    recent_data = np.expand_dims(recent_data, axis=0)  # Reshape to (1, 7, num_features)
    print(f"Prepared recent data shape: {recent_data.shape}")

    # Fit the scaler on the dataset
    scaler = MinMaxScaler()
    scaled_columns = ['Burgers', 'Pizzas', 'Salads'] + [
        f'{item}_lag_{i}' for item in ['Burgers', 'Pizzas', 'Salads'] for i in range(1, 8)
    ]
    data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

    # Scale recent data for prediction
    scaled_recent_data = scaler.transform(data[features].iloc[-7:].values)
    scaled_recent_data = np.expand_dims(scaled_recent_data, axis=0)  # Reshape to (1, 7, num_features)

    # Predict future sales
    try:
        scaled_predictions = model.predict(scaled_recent_data).flatten()
        print(f"Scaled predictions: {scaled_predictions}")
    except Exception as e:
        return f"Prediction error: {e}"

    # Inverse transform predictions to original scale
    temp_array = np.zeros((1, len(scaled_columns)))
    temp_array[0, :3] = scaled_predictions  # Map predictions to first 3 columns (Burgers, Pizzas, Salads)
    unscaled_predictions = scaler.inverse_transform(temp_array)[0, :3]
    print(f"Unscaled predictions: {unscaled_predictions}")

    # Replace negative predictions with 0
    unscaled_predictions = np.maximum(unscaled_predictions, 0)

    # Format results
    results = {
        "Burgers": round(unscaled_predictions[0], 2),
        "Pizzas": round(unscaled_predictions[1], 2),
        "Salads": round(unscaled_predictions[2], 2)
    }
    print(f"Final results: {results}")
    return render_template('results.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
