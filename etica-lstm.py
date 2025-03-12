import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Attention, concatenate, Layer, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import os

# Create directories for saving results
os.makedirs('etica-lstm', exist_ok=True)

# ETICA-LSTM Custom Layer Implementation
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        at = K.softmax(et)
        at = K.expand_dims(at, axis=-1)
        output = x * at
        return K.sum(output, axis=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class ErrorCorrectionLayer(Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        super(ErrorCorrectionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # Adjust to use the actual input shape's last dimension instead of self.units
        self.correction_kernel = self.add_weight(
            name='correction_kernel',
            shape=(input_shape[-1], input_shape[-1]),  # Use input_shape[-1] for both dimensions
            initializer='glorot_uniform',
            trainable=True
        )
        super(ErrorCorrectionLayer, self).build(input_shape)
        
    def call(self, x):
        # Adding error correction term
        correction = K.dot(x, self.correction_kernel)
        return x + 0.1 * correction
        
    def compute_output_shape(self, input_shape):
        return input_shape

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

# Create feature engineering for ETICA-LSTM
# Technical indicators
df['MA5'] = df[target_col].rolling(window=5).mean()  # 5-day moving average
df['MA20'] = df[target_col].rolling(window=20).mean()  # 20-day moving average
df['MA50'] = df[target_col].rolling(window=50).mean()  # 50-day moving average

# Relative Strength Index (RSI)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df[target_col])

# Bollinger Bands
df['20d_std'] = df[target_col].rolling(window=20).std()
df['upper_band'] = df['MA20'] + (df['20d_std'] * 2)
df['lower_band'] = df['MA20'] - (df['20d_std'] * 2)
df['BB_width'] = (df['upper_band'] - df['lower_band']) / df['MA20']

# MACD (Moving Average Convergence Divergence)
df['EMA12'] = df[target_col].ewm(span=12, adjust=False).mean()
df['EMA26'] = df[target_col].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA12'] - df['EMA26']
df['Signal_line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_hist'] = df['MACD'] - df['Signal_line']

# Add price momentum features
df['Price_Momentum'] = df[target_col].pct_change(periods=5)
df['Price_Acceleration'] = df['Price_Momentum'].diff()

# Add volume-based features
if 'KL' in df.columns:
    df['Volume_Momentum'] = df['KL'].pct_change(periods=5)
    df['Volume_MA15'] = df['KL'].rolling(window=15).mean()
    df['Volume_Ratio'] = df['KL'] / df['Volume_MA15']

# Drop NaN values after creating features
df = df.dropna()
print("\nData shape after feature engineering and removing NaN:", df.shape)

# Prepare features and target
feature_columns = ['Lần cuối', 'Mở', 'Cao', 'Thấp', 
                  'MA5', 'MA20', 'MA50', 'RSI', 'BB_width', 
                  'MACD', 'MACD_hist', 'Price_Momentum', 'Price_Acceleration']

if 'KL' in df.columns:
    feature_columns.extend(['Volume_Momentum', 'Volume_Ratio'])

# Get features and scale them
features = df[feature_columns].values
target = df[target_col].values.reshape(-1, 1)

# Scale the data
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features = feature_scaler.fit_transform(features)
scaled_target = target_scaler.fit_transform(target)

# Function to create sequences for LSTM
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length (number of time steps)
seq_length = 60

# Create sequences
X, y = create_sequences(scaled_features, scaled_target, seq_length)

# Split into training and test sets (80-20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"\nTraining data shape: X: {X_train.shape}, y: {y_train.shape}")
print(f"Testing data shape: X: {X_test.shape}, y: {y_test.shape}")

# Build ETICA-LSTM model
print("\nBuilding ETICA-LSTM model...")

# Input layer
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# First LSTM layer with error correction
lstm1 = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
# Apply error correction after the bidirectional layer processes the input
lstm1 = ErrorCorrectionLayer(64)(lstm1)  # The 64 parameter is not used anymore
lstm1 = Dropout(0.3)(lstm1)

# Second LSTM layer
lstm2 = Bidirectional(LSTM(32, return_sequences=True))(lstm1)
lstm2 = Dropout(0.3)(lstm2)

# Attention layer
attention = AttentionLayer()(lstm2)

# Dense layers
dense1 = Dense(32, activation='relu')(attention)
dense1 = Dropout(0.2)(dense1)
output = Dense(1)(dense1)

# Create model
model = Model(inputs=input_layer, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model
print("\nTraining the model...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ETICA-LSTM Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('etica-lstm/training_history.png')
plt.close()

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Inverse transform to get actual values
y_test_inv = target_scaler.inverse_transform(y_test)
y_pred_inv = target_scaler.inverse_transform(y_pred)

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
plt.title(f'VN30 Stock Price Prediction (ETICA-LSTM) - RMSE: {rmse:.4f}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('etica-lstm/prediction_results.png')
plt.close()

# Predict future values
print("\nPredicting future values...")
last_sequence = scaled_features[-seq_length:]
future_days = 30
future_predictions = []

current_sequence = last_sequence.reshape(1, seq_length, len(feature_columns))
for _ in range(future_days):
    # Get prediction for next day
    next_pred = model.predict(current_sequence, verbose=0)[0]
    future_predictions.append(next_pred)
    
    # We need to update the sequence with the new prediction
    # Create a new feature vector for the predicted day
    # For simplicity, we'll copy the last day's features and update the price
    new_features = np.copy(current_sequence[0, -1, :])
    
    # Update price related features (just a simple approach)
    new_features[0] = next_pred  # Update closing price
    
    # Update the sequence by removing the first day and adding the new day
    current_sequence = np.append(current_sequence[:, 1:, :], 
                                np.expand_dims(np.expand_dims(new_features, axis=0), axis=0),
                                axis=1)

# Convert future predictions to actual values
future_predictions = target_scaler.inverse_transform(np.array(future_predictions))

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
plt.title('VN30 Stock Price Prediction (ETICA-LSTM) - Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('etica-lstm/future_predictions.png')
plt.close()

print("\nPrediction for the next 30 days:")
for i, date in enumerate(future_dates):
    print(f"{date.strftime('%d/%m/%Y')}: {future_predictions[i][0]:.2f}")

# Save the model
model.save('etica-lstm/vn30_etica_lstm_model.h5')
print("\nModel saved as 'vn30_etica_lstm_model.h5'")

# Compare with basic LSTM model if available
try:
    from tensorflow.keras.models import load_model
    basic_lstm = load_model('lstm/vn30_lstm_model.h5')
    print("\nComparing ETICA-LSTM with basic LSTM model...")
    
    # Create comparison plot of test predictions
    plt.figure(figsize=(16, 8))
    plt.plot(results_df.index, results_df['Actual'], label='Actual Prices', color='blue')
    plt.plot(results_df.index, results_df['Predicted'], label='ETICA-LSTM', color='red')
    
    # Get basic LSTM predictions if possible
    try:
        basic_pred = basic_lstm.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1))
        basic_pred_inv = target_scaler.inverse_transform(basic_pred)
        
        basic_rmse = math.sqrt(mean_squared_error(y_test_inv, basic_pred_inv))
        
        plt.plot(results_df.index, basic_pred_inv, label=f'Basic LSTM (RMSE: {basic_rmse:.4f})', color='green')
        plt.title(f'Model Comparison - ETICA-LSTM (RMSE: {rmse:.4f}) vs Basic LSTM')
    except:
        plt.title(f'ETICA-LSTM Prediction - RMSE: {rmse:.4f}')
    
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('etica-lstm/model_comparison.png')
    plt.close()
    
except Exception as e:
    print(f"Could not compare with basic LSTM: {e}")

print("\nETICA-LSTM Analysis complete!")
