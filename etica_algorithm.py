"""
Thư viện triển khai thuật toán ETICA cho phân tích chuỗi thời gian
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_etica_components(df, price_column, window_size=5, debug=False):
    """
    Tính toán các thành phần ETICA đầy đủ theo công thức đã cho
    
    Tham số:
    - df: DataFrame chứa dữ liệu giá cổ phiếu
    - price_column: Tên cột chứa giá cổ phiếu
    - window_size: Kích thước cửa sổ (không còn được sử dụng trong phương pháp mới)
    - debug: Bật chế độ gỡ lỗi với biểu đồ và log chi tiết
    
    Trả về:
    - DataFrame với các cột p_ext, p_int đã tính toán
    """
    logging.info("Bắt đầu tính toán thành phần ETICA theo công thức mới...")
    
    # Tạo một bản sao của DataFrame
    result = df.copy()
    
    # Tính phần trăm thay đổi giá theo công thức (1): p_i(t) = (P_i(t) - P_i(t-1)) / P_i(t-1) × 100
    pct_change_col = f'{price_column}_pct'
    if pct_change_col not in result.columns:
        result[pct_change_col] = result[price_column].pct_change() * 100
        result = result.dropna()  # Loại bỏ hàng đầu tiên vì NaN
    
    # Lấy dữ liệu phần trăm thay đổi
    p_t = result[pct_change_col].values
    
    # Tính toán hệ số a theo công thức (4): a = sum(p_i(t)) / sum(sum(p_i(t)))
    # Để tính sum(sum(p_i(t))), chúng ta sẽ tạo ma trận tích lũy
    cumsum_p_t = np.cumsum(p_t)
    sum_p_t = np.sum(p_t)
    sum_cumsum_p_t = np.sum(cumsum_p_t)
    
    # Tránh chia cho 0
    if abs(sum_cumsum_p_t) < 1e-10:
        a = 0.5  # Giá trị mặc định
        logging.warning("Mẫu số gần bằng 0 khi tính hệ số a. Sử dụng giá trị mặc định 0.5")
    else:
        a = sum_p_t / sum_cumsum_p_t
    
    logging.info(f"Hệ số a = {a}")
    
    # Tính toán thành phần bên ngoài theo công thức (3): p_ext(t) = a * sum(p_i(t))
    # Sử dụng cumsum để tính tổng tích lũy
    p_ext = a * cumsum_p_t
    
    # Tính toán thành phần bên trong theo công thức (5)
    p_int = p_t - p_ext
    
    # Thêm thành phần vào DataFrame
    result['p_ext'] = p_ext
    result['p_int'] = p_int
    
    # Debug: Kiểm tra các vấn đề trong thành phần đã tính
    if debug:
        logging.info(f"p_ext - mean: {np.mean(p_ext)}, std: {np.std(p_ext)}")
        logging.info(f"p_int - mean: {np.mean(p_int)}, std: {np.std(p_int)}")
        
        # Kiểm tra tính đúng đắn: p_ext + p_int = p_t
        diff = p_t - (p_ext + p_int)
        logging.info(f"Kiểm tra - Sai số trung bình giữa p_t và (p_ext + p_int): {np.mean(abs(diff))}")
        
        # Hiển thị biểu đồ các thành phần
        plt.figure(figsize=(12, 8))
        plt.subplot(311)
        plt.plot(p_t, 'b', label='Nguyên bản')
        plt.title('Phần trăm thay đổi giá nguyên bản')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(p_ext, 'r', label='Bên ngoài')
        plt.title('Thành phần bên ngoài')
        plt.legend()
        
        plt.subplot(313)
        plt.plot(p_int, 'y', label='Bên trong')  # Màu vàng thay vì xanh lá
        plt.title('Thành phần bên trong')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('images/etica_components.png')
        plt.close()
    
    logging.info("Hoàn thành tính toán thành phần ETICA.")
    return result

def calculate_ex_in_components(df, price_column):
    """
    Chuyển đổi các thành phần p_ext và p_int thành các thành phần ex_t và in_t
    theo công thức (31) và (32)
    
    Tham số:
    - df: DataFrame chứa các cột p_ext và p_int
    - price_column: Tên cột chứa giá cổ phiếu
    
    Trả về:
    - DataFrame với các cột ex_t và in_t đã tính toán
    """
    logging.info("Đang chuyển đổi thành phần tỷ lệ phần trăm sang thành phần tuyệt đối...")
    
    result = df.copy()
    
    # Kiểm tra các cột cần thiết
    required_cols = ['p_ext', 'p_int', price_column]
    for col in required_cols:
        if col not in result.columns:
            error_msg = f"Cột '{col}' không tồn tại trong DataFrame"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    # Tạo cột P_prev - giá tại thời điểm trước
    result['P_prev'] = result[price_column].shift(1)
    
    # Loại bỏ các hàng có NaN sau khi shift
    result = result.dropna(subset=['P_prev'])
    
    # Áp dụng công thức chuyển đổi theo công thức (31) và (32):
    # in_t = (p_int(t) / 100 × P_i(t-1)) + P_i(t-1) / 2
    # ex_t = (p_ext(t) / 100 × P_i(t-1)) + P_i(t-1) / 2
    result['in_t'] = (result['p_int'] / 100 * result['P_prev']) + (result['P_prev'] / 2)
    result['ex_t'] = (result['p_ext'] / 100 * result['P_prev']) + (result['P_prev'] / 2)
    
    # Kiểm tra và xử lý các giá trị NaN hoặc vô cùng
    for col in ['in_t', 'ex_t']:
        # Thay thế các giá trị vô cùng bằng NaN
        result[col] = result[col].replace([np.inf, -np.inf], np.nan)
        # Điền các giá trị NaN bằng phương pháp nội suy
        result[col] = result[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    logging.info("Hoàn thành chuyển đổi thành phần tỷ lệ phần trăm sang thành phần tuyệt đối.")
    return result

def visualize_etica_components(df, price_column='Lần cuối', start_idx=None, end_idx=None, save_path=None):
    """
    Hiển thị trực quan các thành phần ETICA và chuỗi giá gốc.
    
    Tham số:
    - df: DataFrame với các thành phần ETICA
    - price_column: Tên cột chứa giá cổ phiếu
    - start_idx: Chỉ số bắt đầu cho hiển thị
    - end_idx: Chỉ số kết thúc cho hiển thị
    - save_path: Đường dẫn để lưu biểu đồ
    
    Trả về:
    - None
    """
    required_cols = ['p_ext', 'p_int', 'ex_t', 'in_t', price_column]
    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Cột '{col}' không tồn tại trong DataFrame")
            raise ValueError(f"Cột '{col}' không tồn tại trong DataFrame")
    
    # Cắt dữ liệu nếu có chỉ số
    if start_idx is not None or end_idx is not None:
        start = 0 if start_idx is None else start_idx
        end = len(df) if end_idx is None else end_idx
        df_slice = df.iloc[start:end].copy()
    else:
        df_slice = df.copy()
    
    plt.figure(figsize=(15, 10))
    
    # Biểu đồ 1: Giá gốc
    plt.subplot(311)
    plt.plot(df_slice[price_column], 'k', label='Giá gốc')
    plt.title('Giá gốc và phân tách ETICA')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ 2: Thành phần bên ngoài và bên trong (tuyệt đối)
    plt.subplot(312)
    plt.plot(df_slice['ex_t'], 'r', label='Thành phần bên ngoài')
    plt.plot(df_slice['in_t'], 'y', label='Thành phần bên trong')  # Màu vàng
    plt.title('Thành phần bên ngoài và bên trong')
    plt.legend()
    plt.grid(True)
    
    # Biểu đồ 3: Thành phần phần trăm
    plt.subplot(313)
    pct_change_col = f'{price_column}_pct'
    if pct_change_col in df_slice.columns:
        plt.plot(df_slice[pct_change_col], 'b', label='Thay đổi %')
    plt.plot(df_slice['p_ext'], 'r', label='Bên ngoài %')
    plt.plot(df_slice['p_int'], 'y', label='Bên trong %')  # Màu vàng
    plt.title('Thành phần phần trăm thay đổi')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logging.info(f"Đã lưu biểu đồ tại {save_path}")
    
