import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math

# Load dataset
data = pd.read_csv('synthetic_restaurant_sales.csv', parse_dates=['Date'], index_col='Date')

# Preprocess data (Fill NaNs caused by lagging)
data.fillna(method='ffill', inplace=True)

# Scale numerical columns
scaler = MinMaxScaler()
scaled_columns = ['Burgers', 'Pizzas', 'Salads'] + [f'{item}_lag_{i}' for item in ['Burgers', 'Pizzas', 'Salads'] for i in range(1, 8)]
data[scaled_columns] = scaler.fit_transform(data[scaled_columns])

# Drop rows with NaN values caused by lagging
data.dropna(inplace=True)

# Define features and targets
seq_length = 7
features = ['Burgers', 'Pizzas', 'Salads'] + [f'{item}_lag_{i}' for item in ['Burgers', 'Pizzas', 'Salads'] for i in range(1, 8)]
X, y = [], []

for i in range(len(data) - seq_length):
    X.append(data[features].iloc[i:i + seq_length].values)  # Input sequences
    y.append(data[['Burgers', 'Pizzas', 'Salads']].iloc[i + seq_length].values)  # Targets

X, y = np.array(X), np.array(y)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(3)  # 3 outputs: Burgers, Pizzas, Salads
])

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('./model/lstm_model.h5')

# Predictions and inverse scaling for validation
predictions = model.predict(X_test)

# Create a temporary array with the same number of features used during scaling
temp_array = np.zeros((predictions.shape[0], len(scaled_columns)))

# Place the predictions into the correct positions for Burgers, Pizzas, and Salads
temp_array[:, :3] = predictions  # Assuming Burgers, Pizzas, and Salads are the first three columns

# Inverse transform only using the relevant columns
scaled_predictions = scaler.inverse_transform(temp_array)[:, :3]

# Similarly, inverse transform the test target values
temp_test_array = np.zeros((y_test.shape[0], len(scaled_columns)))
temp_test_array[:, :3] = y_test
scaled_y_test = scaler.inverse_transform(temp_test_array)[:, :3]

# Calculate RMSE
rmse = math.sqrt(mean_squared_error(scaled_y_test, scaled_predictions))
print(f'RMSE: {rmse}')

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(scaled_y_test[:, 0], label='Actual Burgers')  # Actual sales for Burgers
plt.plot(scaled_predictions[:, 0], label='Predicted Burgers')  # Predicted sales for Burgers
plt.legend()
plt.show()

