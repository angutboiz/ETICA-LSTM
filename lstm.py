import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import math
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define file path
file_path = "Dữ liệu Lịch sử VN 30.csv"

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path, encoding='utf-8')

# Display basic information
print(f"Data shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# Convert numeric columns - handle the comma format in Vietnamese numbers
for col in ['Lần cuối', 'Mở', 'Cao', 'Thấp']:
    if col in df.columns:
        df[col] = df[col].str.replace(',', '').astype(float)

# Convert KL (volume) - handle the K suffix
if 'KL' in df.columns:
    df['KL'] = df['KL'].str.replace('K', '').str.replace(',', '').astype(float) * 1000

# Convert date column to datetime
if 'Ngày' in df.columns:
    df['Ngày'] = pd.to_datetime(df['Ngày'], format='%d/%m/%Y')
    df.set_index('Ngày', inplace=True)
    print("Set Ngày as index")

# Use Lần cuối (Last/Closing price) for prediction
target_col = 'Lần cuối'
if target_col in df.columns:
    print(f"Using {target_col} for prediction")
else:
    print(f"Could not find {target_col} column. Available columns:", df.columns.tolist())
    exit(1)

# Sort the data by date (ascending)
df = df.sort_index()
print("\nSorted data by date (oldest first):")
print(df.head())

# Prepare the data
data = df[target_col].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length (number of time steps)
seq_length = 60

# Create sequences
X, y = create_sequences(scaled_data, seq_length)

# Split into training and test sets (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nTraining data shape: X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing data shape: X: {X_test.shape}, y: {y_test.shape}")

# Build LSTM model
print("\nBuilding LSTM model...")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('lstm/training_history.png')

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Inverse transform to get actual values
y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate performance metrics
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)

print(f"\nModel Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Create a DataFrame with actual dates for better visualization
test_dates = df.index[train_size+seq_length:].tolist()
results_df = pd.DataFrame({
    'Date': test_dates,
    'Actual': y_test_inv.flatten(),
    'Predicted': y_pred_inv.flatten()
})
results_df.set_index('Date', inplace=True)

# Visualize predictions with dates
plt.figure(figsize=(16, 8))
plt.plot(results_df.index, results_df['Actual'], label='Actual Prices')
plt.plot(results_df.index, results_df['Predicted'], label='Predicted Prices')
plt.title(f'VN30 Stock Price Prediction - RMSE: {rmse:.4f}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('lstm/prediction_results.png')

# Predict future values
last_sequence = scaled_data[-seq_length:]
future_days = 30
future_predictions = []

current_sequence = last_sequence.reshape(1, seq_length, 1)
for _ in range(future_days):
    # Get prediction for next day
    next_pred = model.predict(current_sequence, verbose=0)[0]
    # Add to future predictions
    future_predictions.append(next_pred)
    # Update sequence
    current_sequence = np.append(current_sequence[:, 1:, :], 
                                [[next_pred]], axis=1)

# Convert future predictions to actual values
future_predictions = scaler.inverse_transform(np.array(future_predictions))

# Generate future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# Create future DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted': future_predictions.flatten()
})
future_df.set_index('Date', inplace=True)

# Plot historical and future predictions with dates
plt.figure(figsize=(16, 8))
plt.plot(df.index[-365:], df[target_col][-365:], label='Historical Data')
plt.plot(future_df.index, future_df['Predicted'], 'r--', label='Future Predictions')
plt.title('VN30 Stock Price Prediction - Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('lstm/future_predictions.png')

print("\nPrediction for the next 30 days:")
for i, date in enumerate(future_dates):
    print(f"{date.strftime('%d/%m/%Y')}: {future_predictions[i][0]:.2f}")

# Save the model
model.save('lstm/vn30_lstm_model.h5')
print("\nModel saved as 'vn30_lstm_model.h5'")

print("\nAnalysis complete!")
