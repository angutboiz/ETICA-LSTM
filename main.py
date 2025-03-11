# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from math import sqrt
import warnings
import os
import threading
import time
from etica_algorithm import calculate_etica_components, calculate_ex_in_components
warnings.filterwarnings('ignore')

# Bước 1: Đọc dữ liệu và điền giá trị null
def load_and_preprocess_data(file_path):
    # Đọc dữ liệu
    df = pd.read_csv(file_path, thousands=',')
    
    # Đảm bảo dữ liệu được sắp xếp theo thời gian
    df = df.sort_values(by='Ngày')
    
    # Chuyển đổi các cột số từ string sang float
    numeric_columns = ['Lần cuối', 'Mở', 'Cao', 'Thấp']
    for col in numeric_columns:
        if col in df.columns:
            # Xử lý trường hợp giá trị vẫn là string
            if df[col].dtype == object:
                df[col] = df[col].str.replace(',', '').astype(float)
    
    # Điền giá trị null bằng phương pháp forward fill (sử dụng giá trị gần nhất trước đó)
    df = df.fillna(method='ffill')
    
    # Nếu vẫn còn giá trị null ở đầu chuỗi, sử dụng backward fill
    df = df.fillna(method='bfill')
    
    return df

# Bước 2: Chuyển đổi P_i(t) thành p_i(t) (tính phần trăm thay đổi)
def calculate_percentage_change(df, price_column='Lần cuối'):
    # Tạo một DataFrame mới để lưu trữ phần trăm thay đổi
    df_pct = df.copy()
    
    # Tính phần trăm thay đổi theo công thức: p_i(t) = (P_i(t) - P_i(t-1))/P_i(t-1) * 100
    df_pct[f'{price_column}_pct'] = df[price_column].pct_change() * 100
    
    # Loại bỏ hàng đầu tiên vì không có giá trị phần trăm thay đổi
    df_pct = df_pct.dropna()
    
    return df_pct

# Bước 3 & 4: Áp dụng ETICA để phân tách thành phần bên trong và bên ngoài
def apply_etica(df, price_pct_column, debug=False):
    # Sử dụng thuật toán ETICA từ thư viện với tùy chọn debug
    return calculate_etica_components(
        df, 
        price_column=price_pct_column.replace('_pct', ''), 
        window_size=5,
        debug=debug
    )

# Bước 5: Chuyển đổi ngược p_ext(t) và p_int(t) thành ex(t) và in(t)
def convert_to_ex_in(df, price_column='Lần cuối'):
    # Sử dụng hàm từ thư viện
    return calculate_ex_in_components(df, price_column)

# Bước 6: Chuẩn hóa P_i(t), ex(t) và in(t) bằng phương pháp Min-Max    
def normalize_data(df, columns_to_normalize):
    df_normalized = df.copy()
    scaler_dict = {}
    for column in columns_to_normalize:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_normalized[f'{column}_scaled'] = scaler.fit_transform(df_normalized[column].values.reshape(-1, 1)).flatten()
        scaler_dict[column] = scaler
    return df_normalized, scaler_dict

# Bước 7: Chuẩn bị dữ liệu đầu vào cho LSTM (tạo chuỗi time-series)
def create_sequences(df, target_column, feature_columns, seq_length=60):
    X, y = [], []
    for i in range(len(df) - seq_length):
        # Tạo chuỗi đầu vào X sử dụng các đặc trưng
        features_seq = df[feature_columns].iloc[i:i+seq_length].values
        X.append(features_seq)
        
        # Giá trị mục tiêu y (giá tại thời điểm tiếp theo)
        y.append(df[target_column].iloc[i+seq_length])
    
    return np.array(X), np.array(y)

# Bước 7: Chia dữ liệu thành tập huấn luyện và tập kiểm tra
def split_data(X, y, train_ratio=0.6):
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# Bước 8: Xây dựng và huấn luyện mô hình LSTM
def build_and_train_lstm_model(X_train, y_train, X_test, y_test, epochs=200, batch_size=32, model_path='model.h5'):
    # Lấy kích thước đầu vào
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Tạo checkpoint để lưu mô hình tốt nhất
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    
    return model, history

# Hàm để tạo luồng huấn luyện
def train_model_thread(X_train, y_train, X_test, y_test, epochs, batch_size, model_path, results_dict):
    model, history = build_and_train_lstm_model(X_train, y_train, X_test, y_test, epochs, batch_size, model_path)
    results_dict['model'] = model
    results_dict['history'] = history
    results_dict['completed'] = True
    print("\nHuấn luyện mô hình hoàn tất!")

# Bước 9: Dự đoán giá cổ phiếu và các thành phần
def predict_with_lstm(model, X):
    predictions = model.predict(X)
    # Loại bỏ các giá trị NaN nếu có
    return np.nan_to_num(predictions)

# Bước 10: Đánh giá mô hình
def evaluate_model(y_true, y_pred):
    # Ensure both arrays are 1-dimensional
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Loại bỏ các giá trị NaN từ dữ liệu trước khi đánh giá
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        print("Cảnh báo: Không có dữ liệu hợp lệ để đánh giá sau khi loại bỏ NaN")
        return {'RMSE': np.nan, 'MAE': np.nan}
    
    # Tính RMSE
    rmse = sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    
    # Tính MAE
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    
    return {
        'RMSE': rmse,
        'MAE': mae
    }

# Bước 11: Xây dựng mô hình LSTM truyền thống (không sử dụng ETICA)
def build_traditional_lstm(df, price_column='Lần cuối', seq_length=60, train_ratio=0.6, epochs=200, batch_size=32):
    print("Xây dựng mô hình LSTM truyền thống (không sử dụng ETICA)...")
    
    # Chuẩn hóa giá trực tiếp
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[price_column].values.reshape(-1, 1))
    
    # Tạo chuỗi X và y cho LSTM
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Xây dựng mô hình LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'traditional_lstm_model.h5')
    
    # Tạo checkpoint để lưu mô hình tốt nhất
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
    
    # Huấn luyện mô hình
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Dự đoán giá
    y_pred = model.predict(X_test)
    
    # Chuyển đổi dự đoán về giá thực
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    # Đánh giá mô hình
    metrics = evaluate_model(y_test, y_pred)
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'actual': y_test_inv,
        'predicted': y_pred_inv,
        'X_test': X_test,
        'y_test': y_test
    }

# Hàm so sánh các mô hình
def compare_models(models_results):
    """
    So sánh hiệu suất của các mô hình khác nhau
    
    Parameters:
    models_results (dict): Dictionary chứa kết quả của các mô hình
    
    Returns:
    dict: Kết quả so sánh
    """
    comparison = {}
    
    # So sánh các metrics
    for model_name, results in models_results.items():
        comparison[model_name] = {
            'RMSE': results['metrics']['RMSE'],
            'MAE': results['metrics']['MAE']
        }
    
    # Tạo bảng so sánh
    df_comparison = pd.DataFrame(comparison).T
    
    # Vẽ biểu đồ so sánh
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE
    df_comparison['RMSE'].plot(kind='bar', ax=ax[0], color='skyblue')
    ax[0].set_title('RMSE Comparison')
    ax[0].set_ylabel('RMSE Value')
    ax[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE
    df_comparison['MAE'].plot(kind='bar', ax=ax[1], color='lightgreen')
    ax[1].set_title('MAE Comparison')
    ax[1].set_ylabel('MAE Value')
    ax[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/model_comparison.png')
    plt.show()
    
    # Vẽ biểu đồ dự đoán của các mô hình
    plt.figure(figsize=(12, 6))
    
    # Giới hạn số lượng điểm dữ liệu để biểu đồ dễ đọc hơn
    last_points = 100
    
    for model_name, results in models_results.items():
        plt.plot(results['actual'][-last_points:], label=f'Actual ({model_name})')
        plt.plot(results['predicted'][-last_points:], label=f'Predicted ({model_name})', linestyle='--')
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/prediction_comparison.png')
    plt.show()
    
    return df_comparison

# Placeholder cho các mô hình hybrid khác
def build_emd_lstm_model(df, price_column='Lần cuối'):
    # Placeholder for EMD-LSTM implementation
    print("EMD-LSTM model not implemented yet")
    return {
        'metrics': {'RMSE': None, 'MAE': None},
        'actual': None,
        'predicted': None
    }

def build_ceemdan_lstm_model(df, price_column='Lần cuối'):
    # Placeholder for CEEMDAN-LSTM implementation
    print("CEEMDAN-LSTM model not implemented yet")
    return {
        'metrics': {'RMSE': None, 'MAE': None},
        'actual': None,
        'predicted': None
    }

# Hàm chính để chạy toàn bộ workflow
def run_etica_lstm_workflow(file_path, price_column='Lần cuối', seq_length=60, train_ratio=0.6, epochs=200, batch_size=32):
    # Tạo thư mục để lưu mô hình nếu chưa tồn tại
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'etica_lstm_model.h5')
    train_data_path = os.path.join(model_dir, 'train_data.npz')

    # Bước 1: Đọc và tiền xử lý dữ liệu
    print("Bước 1: Đọc và tiền xử lý dữ liệu...")
    df = load_and_preprocess_data(file_path)
    
    # Bước 2: Tính phần trăm thay đổi
    print("Bước 2: Tính phần trăm thay đổi p_i(t)...")
    df_pct = calculate_percentage_change(df, price_column)
    
    # Bước 3 & 4: Áp dụng ETICA
    print("Bước 3 & 4: Áp dụng ETICA để tính p_ext(t) và p_int(t)...")
    df_etica = apply_etica(df_pct, f'{price_column}_pct', debug=True)
    
    # Bước 5: Chuyển đổi ngược về ex(t) và in(t)
    print("Bước 5: Chuyển đổi ngược về ex(t) và in(t)...")
    df_ex_in = convert_to_ex_in(df_etica, price_column)
    
    # Thêm: Trực quan hóa các thành phần ETICA để kiểm tra
    from etica_algorithm import visualize_etica_components
    print("Đang hiển thị trực quan các thành phần ETICA để kiểm tra...")
    visualize_etica_components(df_ex_in, price_column, start_idx=-200, save_path='images/etica_verification.png')
    
    # Bước 6: Chuẩn hóa dữ liệu
    print("Bước 6: Chuẩn hóa dữ liệu...")
    columns_to_normalize = [price_column, 'ex_t', 'in_t']
    df_normalized, scalers = normalize_data(df_ex_in, columns_to_normalize)
    
    # Chuẩn bị đặc trưng cho mô hình LSTM
    feature_columns = [f'{price_column}_scaled', 'ex_t_scaled', 'in_t_scaled']
    target_column = f'{price_column}_scaled'
    
    # Bước 7: Tạo chuỗi time-series và chia dữ liệu
    print("Bước 7: Tạo chuỗi time-series và chia dữ liệu...")
    X, y = create_sequences(df_normalized, target_column, feature_columns, seq_length)
    X_train, X_test, y_train, y_test = split_data(X, y, train_ratio)
    
    # Lưu dữ liệu huấn luyện và kiểm tra để sử dụng lại
    np.savez(train_data_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    # Kiểm tra xem mô hình đã tồn tại chưa
    model = None
    history = None
    if os.path.exists(model_path):
        print(f"Tìm thấy mô hình đã lưu tại {model_path}. Đang tải mô hình...")
        try:
            model = load_model(model_path)
            print("Đã tải mô hình thành công!")
        except Exception as e:
            print(f"Không thể tải mô hình: {str(e)}. Huấn luyện mô hình mới...")
            model = None
    
    # Nếu không tìm thấy mô hình đã lưu hoặc không thể tải, huấn luyện mới
    if model is None:
        # Bước 8: Xây dựng và huấn luyện mô hình LSTM trong một luồng riêng
        print("Bước 8: Xây dựng và huấn luyện mô hình LSTM trong luồng riêng...")
        training_results = {'completed': False}
        training_thread = threading.Thread(
            target=train_model_thread,
            args=(X_train, y_train, X_test, y_test, epochs, batch_size, model_path, training_results)
        )
        training_thread.start()
        
        # Đợi cho đến khi quá trình huấn luyện hoàn tất
        while not training_results.get('completed', False):
            print("Mô hình đang được huấn luyện... (Bạn có thể nhấn Ctrl+C để dừng và sử dụng mô hình đã lưu)")
            time.sleep(10)  # Kiểm tra mỗi 10 giây
        
        model = training_results['model']
        history = training_results['history']
    
    # Bước 9: Dự đoán
    print("Bước 9: Dự đoán...")
    if model is None:
        print("Tải mô hình đã lưu để dự đoán...")
        model = load_model(model_path)
    
    y_pred = predict_with_lstm(model, X_test)
    
    # Bước 10: Đánh giá mô hình...")
    metrics = evaluate_model(y_test, y_pred)
    print(f"RMSE: {metrics['RMSE']}")
    print(f"MAE: {metrics['MAE']}")
    
    # Chuyển đổi dự đoán về giá thực
    y_test_inv = scalers[price_column].inverse_transform(y_test.reshape(-1, 1))
    y_pred_inv = scalers[price_column].inverse_transform(y_pred)
    
    # Vẽ đồ thị kết quả
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual Price')
    plt.plot(y_pred_inv, label='Predicted Price')
    plt.title('Stock Price Prediction with ETICA-LSTM')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Bước 11: So sánh với các mô hình khác
    print("\nBước 11: So sánh ETICA-LSTM với các mô hình khác...")
    
    # Đọc lại dữ liệu gốc cho mô hình LSTM truyền thống
    df_original = load_and_preprocess_data(file_path)
    
    # Xây dựng và huấn luyện mô hình LSTM truyền thống
    traditional_lstm_results = build_traditional_lstm(
        df=df_original,
        price_column=price_column,
        seq_length=seq_length,
        train_ratio=train_ratio,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Tạo dictionary để lưu kết quả của các mô hình
    models_results = {
        'ETICA-LSTM': {
            'metrics': metrics,
            'actual': y_test_inv,
            'predicted': y_pred_inv
        },
        'Traditional LSTM': traditional_lstm_results
    }
    
    # Thêm placeholder cho các mô hình hybrid khác
    # Có thể uncomment để triển khai trong tương lai
    # models_results['EMD-LSTM'] = build_emd_lstm_model(df_original, price_column)
    # models_results['CEEMDAN-LSTM'] = build_ceemdan_lstm_model(df_original, price_column)
    
    # So sánh các mô hình
    comparison_result = compare_models(models_results)
    print("\nKết quả so sánh các mô hình:")
    print(comparison_result)
    
    return {
        'model': model,
        'history': history,
        'metrics': metrics,
        'actual': y_test_inv,
        'predicted': y_pred_inv,
        'scalers': scalers,
        'df_final': df_normalized,
        'model_comparison': comparison_result,
        'models_results': models_results
    }

# Chạy workflow với dữ liệu cổ phiếu
if __name__ == "__main__":
    # Đường dẫn đến file dữ liệu cổ phiếu
    file_path = "Dữ liệu Lịch sử VN 30.csv"
    
    # Chạy workflow
    results = run_etica_lstm_workflow(
        file_path=file_path,
        price_column='Lần cuối',  # Cột giá đóng cửa
        seq_length=60,         # Độ dài chuỗi (60 ngày)
        train_ratio=0.6,       # Tỷ lệ chia tập huấn luyện (60%)
        epochs=200,            # Số epochs
        batch_size=32          # Kích thước batch
    )