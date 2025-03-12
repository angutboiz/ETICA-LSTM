import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Layer, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import os
import time

# Create directory for comparison results
os.makedirs('comparison', exist_ok=True)

print("=" * 50)
print("COMPARING ETICA-LSTM AND STANDARD LSTM FOR STOCK PREDICTION")
print("=" * 50)

# Custom layers for ETICA-LSTM
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
        self.correction_kernel = self.add_weight(
            name='correction_kernel',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        super(ErrorCorrectionLayer, self).build(input_shape)
        
    def call(self, x):
        correction = K.dot(x, self.correction_kernel)
        return x + 0.1 * correction
        
    def compute_output_shape(self, input_shape):
        return input_shape

# Set random seed for reproducibility
np.random.seed(42)

# Define file path
file_path = "Dữ liệu Lịch sử VN 30.csv"

# 1. DATA LOADING AND PREPROCESSING
print("\n1. LOADING AND PREPROCESSING DATA")
print("-" * 40)

# Load the data
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path, encoding='utf-8')

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

# 2. FEATURE ENGINEERING
print("\n2. FEATURE ENGINEERING")
print("-" * 40)

# Create a copy of the original dataframe for standard LSTM (with minimal features)
df_simple = df.copy()

# Enhanced feature engineering for ETICA-LSTM
print("Creating technical indicators for ETICA-LSTM...")
# Technical indicators
df['MA5'] = df[target_col].rolling(window=5).mean()
df['MA20'] = df[target_col].rolling(window=20).mean()
df['MA50'] = df[target_col].rolling(window=50).mean()

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
print(f"ETICA-LSTM data shape after feature engineering: {df.shape}")

# 3. PREPARE TRAIN/TEST DATA FOR BOTH MODELS
print("\n3. PREPARING DATASETS")
print("-" * 40)

# Sequence length (time steps)
seq_length = 60

# DATASET FOR STANDARD LSTM
data_simple = df_simple[target_col].values.reshape(-1, 1)
scaler_simple = MinMaxScaler(feature_range=(0, 1))
scaled_data_simple = scaler_simple.fit_transform(data_simple)

# Create sequences for standard LSTM
def create_sequences_simple(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X_simple, y_simple = create_sequences_simple(scaled_data_simple, seq_length)

# DATASET FOR ETICA-LSTM
# Prepare features and target
feature_columns = ['Lần cuối', 'Mở', 'Cao', 'Thấp', 
                  'MA5', 'MA20', 'MA50', 'RSI', 'BB_width', 
                  'MACD', 'MACD_hist', 'Price_Momentum', 'Price_Acceleration']

if 'KL' in df.columns:
    feature_columns.extend(['Volume_Momentum', 'Volume_Ratio'])

# Get features and scale them
features_etica = df[feature_columns].values
target_etica = df[target_col].values.reshape(-1, 1)

# Scale the data
feature_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_features_etica = feature_scaler.fit_transform(features_etica)
scaled_target_etica = target_scaler.fit_transform(target_etica)

# Function to create sequences for ETICA-LSTM
def create_sequences_etica(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i + seq_length])
        y.append(target[i + seq_length])
    return np.array(X), np.array(y)

X_etica, y_etica = create_sequences_etica(scaled_features_etica, scaled_target_etica, seq_length)

# Split data for both models using the same proportion
train_size = int(len(X_simple) * 0.8)
X_train_simple, X_test_simple = X_simple[:train_size], X_simple[train_size:]
y_train_simple, y_test_simple = y_simple[:train_size], y_simple[train_size:]

# For ETICA, match the length of the simplified dataset
train_size_etica = min(train_size, len(X_etica) - len(X_test_simple))
X_train_etica, X_test_etica = X_etica[:train_size_etica], X_etica[-len(X_test_simple):]
y_train_etica, y_test_etica = y_etica[:train_size_etica], y_etica[-len(y_test_simple):]

print(f"Standard LSTM - Training data: {X_train_simple.shape}, Testing data: {X_test_simple.shape}")
print(f"ETICA-LSTM - Training data: {X_train_etica.shape}, Testing data: {X_test_etica.shape}")

# 4. BUILD AND TRAIN MODELS
print("\n4. BUILDING AND TRAINING MODELS")
print("-" * 40)

# Define a reusable function for model training
def train_model(model, X_train, y_train, X_test, y_test, name, epochs=100, batch_size=32, patience=10):
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    
    print(f"Training {name} model...")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"{name} training completed in {training_time:.2f} seconds")
    
    return history, training_time

# Try to load existing models first
try_load_models = True  # Set to False to force retraining

# Build and train Standard LSTM model
if try_load_models and os.path.exists('lstm/vn30_lstm_model.h5'):
    print("Loading pre-trained standard LSTM model...")
    lstm_model = load_model('lstm/vn30_lstm_model.h5')
    lstm_training_time = 0  # Not trained in this run
    lstm_history = None
else:
    print("Building and training standard LSTM model...")
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_simple.shape[1], 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=25))
    lstm_model.add(Dense(units=1))
    
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.summary()
    
    lstm_history, lstm_training_time = train_model(
        lstm_model, 
        X_train_simple, y_train_simple,
        X_test_simple, y_test_simple,
        "Standard LSTM"
    )

# Build and train ETICA-LSTM model
custom_objects = {
    'AttentionLayer': AttentionLayer,
    'ErrorCorrectionLayer': ErrorCorrectionLayer
}

if try_load_models and os.path.exists('etica-lstm/vn30_etica_lstm_model.h5'):
    print("Loading pre-trained ETICA-LSTM model...")
    tf.keras.utils.get_custom_objects().update(custom_objects)
    etica_model = load_model('etica-lstm/vn30_etica_lstm_model.h5')
    etica_training_time = 0  # Not trained in this run
    etica_history = None
else:
    print("Building and training ETICA-LSTM model...")
    # Input layer
    input_layer = Input(shape=(X_train_etica.shape[1], X_train_etica.shape[2]))
    
    # First LSTM layer with error correction
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    lstm1 = ErrorCorrectionLayer(64)(lstm1)
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
    etica_model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    etica_model.compile(optimizer='adam', loss='mean_squared_error')
    etica_model.summary()
    
    etica_history, etica_training_time = train_model(
        etica_model, 
        X_train_etica, y_train_etica,
        X_test_etica, y_test_etica,
        "ETICA-LSTM",
        epochs=150,
        patience=15
    )

# 5. EVALUATE AND COMPARE MODELS
print("\n5. MODEL EVALUATION AND COMPARISON")
print("-" * 40)

# Make predictions with both models
print("Generating predictions...")
lstm_predictions = lstm_model.predict(X_test_simple, verbose=0)
etica_predictions = etica_model.predict(X_test_etica, verbose=0)

# Check and ensure array lengths match
min_length = min(len(lstm_predictions), len(etica_predictions))
if len(lstm_predictions) != len(etica_predictions):
    print(f"Warning: Predictions have different lengths. LSTM: {len(lstm_predictions)}, ETICA: {len(etica_predictions)}")
    print(f"Truncating to minimum length: {min_length}")
    lstm_predictions = lstm_predictions[:min_length]
    etica_predictions = etica_predictions[:min_length]
    y_test_simple = y_test_simple[:min_length]
    y_test_etica = y_test_etica[:min_length]

# Inverse transform predictions to original scale
lstm_pred_inv = scaler_simple.inverse_transform(lstm_predictions)
lstm_test_inv = scaler_simple.inverse_transform(y_test_simple[:len(lstm_predictions)])
etica_pred_inv = target_scaler.inverse_transform(etica_predictions)
etica_test_inv = target_scaler.inverse_transform(y_test_etica[:len(etica_predictions)])

# Calculate performance metrics
lstm_mse = mean_squared_error(lstm_test_inv, lstm_pred_inv)
lstm_rmse = math.sqrt(lstm_mse)
lstm_mae = mean_absolute_error(lstm_test_inv, lstm_pred_inv)

etica_mse = mean_squared_error(etica_test_inv, etica_pred_inv)
etica_rmse = math.sqrt(etica_mse)
etica_mae = mean_absolute_error(etica_test_inv, etica_pred_inv)

# Calculate improvement percentages
rmse_improvement = ((lstm_rmse - etica_rmse) / lstm_rmse) * 100
mae_improvement = ((lstm_mae - etica_mae) / lstm_mae) * 100

# Print comparison table
print("\nPERFORMANCE COMPARISON:")
print("-" * 70)
print(f"{'Metric':<20} {'Standard LSTM':<20} {'ETICA-LSTM':<20} {'Improvement'}")
print("-" * 70)
print(f"{'MSE':<20} {lstm_mse:<20.4f} {etica_mse:<20.4f} {(lstm_mse - etica_mse):.4f}")
print(f"{'RMSE':<20} {lstm_rmse:<20.4f} {etica_rmse:<20.4f} {(lstm_rmse - etica_rmse):.4f} ({rmse_improvement:.2f}%)")
print(f"{'MAE':<20} {lstm_mae:<20.4f} {etica_mae:<20.4f} {(lstm_mae - etica_mae):.4f} ({mae_improvement:.2f}%)")
print(f"{'Training Time':<20} {lstm_training_time:<20.2f}s {etica_training_time:<20.2f}s")
print("-" * 70)

# Create visualizations to compare model performance

# 1. Create a DataFrame with actual dates for better visualization
# Make sure to account for possibly different lengths of test data
try:
    # Ensure we have enough dates by using the minimum length
    max_dates = min(len(lstm_test_inv), len(df.index) - (train_size + seq_length))
    test_dates = df.index[train_size+seq_length:train_size+seq_length+max_dates].tolist()
    
    # Ensure all arrays have the same length
    min_len = min(len(test_dates), len(lstm_test_inv), len(lstm_pred_inv), len(etica_pred_inv))
    test_dates = test_dates[:min_len]
    actual_vals = lstm_test_inv.flatten()[:min_len]
    lstm_preds = lstm_pred_inv.flatten()[:min_len]
    etica_preds = etica_pred_inv.flatten()[:min_len]
    
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actual_vals,
        'LSTM_Pred': lstm_preds,
        'ETICA_Pred': etica_preds
    })
    results_df.set_index('Date', inplace=True)
    
    # 2. Plot predictions comparison
    plt.figure(figsize=(16, 10))
    plt.plot(results_df.index, results_df['Actual'], label='Actual Prices', color='black', linewidth=2)
    plt.plot(results_df.index, results_df['LSTM_Pred'], label=f'Standard LSTM (RMSE: {lstm_rmse:.2f})', color='blue')
    plt.plot(results_df.index, results_df['ETICA_Pred'], label=f'ETICA-LSTM (RMSE: {etica_rmse:.2f})', color='red')

    plt.title('VN30 Stock Price Prediction: LSTM vs ETICA-LSTM', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comparison/prediction_comparison.png')
    plt.close()

    # 3. Plot error comparison
    plt.figure(figsize=(16, 10))
    plt.plot(results_df.index, np.abs(results_df['Actual'] - results_df['LSTM_Pred']), 
             label=f'LSTM Error (MAE: {lstm_mae:.2f})', color='blue')
    plt.plot(results_df.index, np.abs(results_df['Actual'] - results_df['ETICA_Pred']), 
             label=f'ETICA-LSTM Error (MAE: {etica_mae:.2f})', color='red')

    plt.title('Absolute Prediction Error: LSTM vs ETICA-LSTM', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Absolute Error', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('comparison/error_comparison.png')
    plt.close()

except Exception as e:
    print(f"Error creating comparison visualizations: {e}")
    print("Skipping visualization steps, but continuing with the rest of the analysis...")

# 4. Plot training history if we have it
if lstm_history and etica_history:
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history.history['loss'], label='LSTM Training Loss')
    plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
    plt.title('Standard LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(etica_history.history['loss'], label='ETICA Training Loss')
    plt.plot(etica_history.history['val_loss'], label='ETICA Validation Loss')
    plt.title('ETICA-LSTM Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison/training_history_comparison.png')
    plt.close()

# 6. FUTURE PREDICTIONS WITH BOTH MODELS
print("\n6. FUTURE PREDICTIONS")
print("-" * 40)

# Predict future values with both models
future_days = 30
print(f"\nPredicting next {future_days} days with both models...")

# Standard LSTM future prediction
last_sequence_simple = scaled_data_simple[-seq_length:]
lstm_future = []

current_sequence = last_sequence_simple.reshape(1, seq_length, 1)
for _ in range(future_days):
    next_pred = lstm_model.predict(current_sequence, verbose=0)[0]
    lstm_future.append(next_pred)
    current_sequence = np.append(current_sequence[:, 1:, :], 
                                [[next_pred]], axis=1)

lstm_future = scaler_simple.inverse_transform(np.array(lstm_future))

# ETICA-LSTM future prediction
last_sequence_etica = scaled_features_etica[-seq_length:]
etica_future = []

current_sequence = last_sequence_etica.reshape(1, seq_length, len(feature_columns))
for _ in range(future_days):
    next_pred = etica_model.predict(current_sequence, verbose=0)[0]
    etica_future.append(next_pred)
    
    # Update features for next prediction (simplified approach)
    new_features = np.copy(current_sequence[0, -1, :])
    new_features[0] = next_pred  # Update closing price
    
    current_sequence = np.append(current_sequence[:, 1:, :], 
                                np.expand_dims(np.expand_dims(new_features, axis=0), axis=0),
                                axis=1)

etica_future = target_scaler.inverse_transform(np.array(etica_future))

# Generate future dates
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# Create future DataFrame
future_df = pd.DataFrame({
    'Date': future_dates,
    'LSTM_Prediction': lstm_future.flatten(),
    'ETICA_Prediction': etica_future.flatten()
})
future_df.set_index('Date', inplace=True)

# Plot future predictions comparison
plt.figure(figsize=(16, 8))
plt.plot(df.index[-90:], df[target_col][-90:], label='Historical Data', color='black')
plt.plot(future_df.index, future_df['LSTM_Prediction'], 'b--', label='LSTM Future Prediction')
plt.plot(future_df.index, future_df['ETICA_Prediction'], 'r--', label='ETICA-LSTM Future Prediction')
plt.title('VN30 Stock Price Prediction - Next 30 Days: LSTM vs ETICA-LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('comparison/future_predictions_comparison.png')

# Print future prediction values
print("\nFuture prediction comparison:")
print("-" * 70)
print(f"{'Date':<15} {'LSTM Prediction':<20} {'ETICA-LSTM Prediction':<20} {'Difference'}")
print("-" * 70)
for i, date in enumerate(future_dates):
    lstm_val = lstm_future[i][0]
    etica_val = etica_future[i][0]
    diff = etica_val - lstm_val
    print(f"{date.strftime('%d/%m/%Y'):<15} {lstm_val:<20.2f} {etica_val:<20.2f} {diff:+.2f}")
print("-" * 70)

# 7. MODEL ARCHITECTURE COMPARISON
print("\n7. MODEL ARCHITECTURE COMPARISON")
print("-" * 40)
print("\nStandard LSTM Model:")
lstm_model.summary()

print("\nETICA-LSTM Model:")
etica_model.summary()

# 8. CONCLUSION
print("\n8. CONCLUSION")
print("-" * 40)
print(f"ETICA-LSTM demonstrated {rmse_improvement:.2f}% improvement in RMSE over the standard LSTM model.")
print(f"ETICA-LSTM demonstrated {mae_improvement:.2f}% improvement in MAE over the standard LSTM model.")

if rmse_improvement > 0:
    print("\nETICA-LSTM outperformed the standard LSTM model with:")
    print("  - Advanced feature engineering with technical indicators")
    print("  - Bidirectional LSTM layers for capturing temporal patterns")
    print("  - Attention mechanism for focusing on the most relevant time steps")
    print("  - Error correction layer for refining predictions")
else:
    print("\nIn this dataset, the standard LSTM model performed comparably or better than the more complex ETICA-LSTM model.")
    print("This may be due to the simplicity of the underlying patterns in the data or overfitting in the ETICA model.")

print("\nComparison completed. Results saved in the 'comparison' directory.")
