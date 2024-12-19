# Restaurant Sales Prediction Project

This project focuses on predicting restaurant sales for three main food categories: Burgers, Pizzas, and Salads. It employs an LSTM (Long Short-Term Memory) neural network for time-series forecasting, with a Flask-based web interface for uploading datasets and visualizing predictions.

## Features

- **Time-Series Sales Prediction**: Predict future sales for Burgers, Pizzas, and Salads using past sales data.
- **LSTM Model**: A trained deep learning model is used for accurate forecasting.
- **Dynamic Web Interface**: Upload datasets, view predictions, and provide feedback via a Flask app.
- **Feedback Mechanism**: Users can submit feedback about the application.
- **Interactive Visualizations**: Compare actual sales with predictions using detailed plots.

## File Structure

- **`updated_lstm.py`**: Contains the code for training the LSTM model on time-series sales data.
- **`updated_app.py`**: Flask application for uploading datasets, running predictions, and displaying results.
- **`feedback.json`**: Stores user feedback collected from the web interface.
- **`restaurant_sales.csv`**: Main dataset for training and evaluation (not included for privacy).
- **`synthetic_restaurant_sales.csv`**: Synthetic dataset for testing purposes.
- **`/model/lstm_model.h5`**: Pre-trained LSTM model file.

## Setup

### Prerequisites

- Python 3.8 or higher
- Required Python libraries:
  - Flask
  - pandas
  - numpy
  - tensorflow
  - scikit-learn
  - matplotlib

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/restaurant-sales-prediction.git
   cd restaurant-sales-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Preparing the Environment

- Place the pre-trained model (`lstm_model.h5`) in the `./model` directory.
- Ensure datasets are placed in the `./uploads` directory or uploaded via the web interface.

## Running the Application

1. Start the Flask server:
   ```bash
   python updated_app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`.

### Features in the Web Interface

- **Upload Dataset**: Upload a CSV file containing sales data.
- **View Predictions**: See predicted sales for the next time step based on the uploaded dataset.
- **Submit Feedback**: Provide your thoughts or suggestions on the app.

## Model Training and Evaluation

The LSTM model is trained using the `updated_lstm.py` script, which performs the following:

1. **Data Preprocessing**:
   - Scaling sales data using MinMaxScaler.
   - Generating lag features for input sequences.
2. **Model Architecture**:
   - Two LSTM layers with dropout for regularization.
   - A dense output layer predicting sales for Burgers, Pizzas, and Salads.
3. **Evaluation**:
   - Calculates RMSE (Root Mean Squared Error) to measure prediction accuracy.
   - Visualizes actual vs predicted sales using Matplotlib.

### Training

To train the model on new data, run:
```bash
python updated_lstm.py
```
The trained model will be saved in the `./model` directory.

## Feedback Storage

Feedback is stored in `feedback.json` in the following format:
```json
[
    {
        "name": "Adam",
        "feedback": "I like this webpage"
    },
    {
        "name": "Douglas",
        "feedback": "I want a more complex prediction model!"
    }
]
```

## Future Enhancements

- Add multi-step forecasting to predict sales for several days ahead.
- Implement user authentication for personalized feedback.
- Enhance the web interface with richer visualizations and dynamic charts.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- Developed by Adam Abdulmajid
- Leveraged TensorFlow for building and training the LSTM model.

