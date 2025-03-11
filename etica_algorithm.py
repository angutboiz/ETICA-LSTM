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
    Tính toán các thành phần ETICA đầy đủ với phương pháp tự tương quan
    
    Tham số:
    - df: DataFrame chứa dữ liệu giá cổ phiếu
    - price_column: Tên cột chứa giá cổ phiếu
    - window_size: Kích thước cửa sổ cho tính toán tự tương quan
    - debug: Bật chế độ gỡ lỗi với biểu đồ và log chi tiết
    
    Trả về:
    - DataFrame với các cột p_ext, p_int đã tính toán
    """
    logging.info("Bắt đầu tính toán thành phần ETICA...")
    
    # Tạo một bản sao của DataFrame
    result = df.copy()
    
    # Tính phần trăm thay đổi giá
    pct_change_col = f'{price_column}_pct'
    if pct_change_col not in result.columns:
        result[pct_change_col] = result[price_column].pct_change() * 100
        result = result.dropna()  # Loại bỏ hàng đầu tiên vì NaN
    
    # Lấy dữ liệu phần trăm thay đổi
    p_t = result[pct_change_col].values
    
    # Tính toán hàm tự tương quan
    logging.info(f"Đang tính toán tự tương quan với window_size={window_size}")
    try:
        # Tính hàm tự tương quan cho độ trễ đến window_size
        r_k = acf(p_t, nlags=window_size, fft=False)
        
        # Kiểm tra nếu có giá trị NaN
        if np.isnan(r_k).any():
            logging.error("Hàm tự tương quan chứa giá trị NaN. Sử dụng phương pháp thay thế.")
            # Phương pháp thay thế: tính tự tương quan thủ công
            r_k = np.array([np.corrcoef(p_t[:-i], p_t[i:])[0, 1] if i > 0 else 1.0 
                          for i in range(window_size + 1)])
    except Exception as e:
        logging.error(f"Lỗi khi tính tự tương quan: {str(e)}. Sử dụng phương pháp thay thế.")
        # Phương pháp thay thế
        r_k = np.array([np.corrcoef(p_t[:-i], p_t[i:])[0, 1] if i > 0 else 1.0 
                      for i in range(window_size + 1)])
    
    # Debug: Hiển thị biểu đồ tự tương quan
    if debug:
        plt.figure(figsize=(10, 5))
        plt.stem(range(len(r_k)), r_k)
        plt.title('Hàm tự tương quan')
        plt.xlabel('Độ trễ')
        plt.ylabel('Tự tương quan')
        plt.axhline(y=0, linestyle='--', color='gray')
        plt.grid(True)
        plt.savefig('images/autocorrelation.png')
        plt.close()
        
        logging.info(f"Giá trị tự tương quan: {r_k}")
    
    # Khởi tạo mảng cho thành phần bên ngoài và bên trong
    p_ext = np.zeros_like(p_t)
    p_int = np.zeros_like(p_t)
    
    # Tính toán thành phần bên ngoài và bên trong
    logging.info("Đang tính toán thành phần bên ngoài và bên trong...")
    
    # Đảm bảo r_k[0] không bằng không để tránh chia cho 0
    if abs(r_k[0]) < 1e-10:
        r_k[0] = 1.0
        logging.warning("Giá trị tự tương quan đầu tiên gần bằng không. Đặt bằng 1.0 để tránh lỗi chia cho 0.")
    
    # Tính hệ số βe và βi
    beta_e = r_k[1] / r_k[0]
    beta_i = 1 - beta_e
    
    logging.info(f"Hệ số beta: beta_e={beta_e}, beta_i={beta_i}")
    
    # Nếu beta vượt ra ngoài [0,1], sử dụng giá trị mặc định
    if not (0 <= beta_e <= 1) or not (0 <= beta_i <= 1):
        logging.warning(f"Hệ số beta không hợp lệ: beta_e={beta_e}, beta_i={beta_i}. Sử dụng giá trị mặc định.")
        beta_e = 0.5
        beta_i = 0.5
    
    # Tính p_ext và p_int theo công thức ETICA
    for t in range(1, len(p_t)):
        p_ext[t] = beta_e * p_t[t-1] + beta_e * p_ext[t-1] if t > 1 else beta_e * p_t[t-1]
        p_int[t] = p_t[t] - p_ext[t]
    
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
    
    # Áp dụng công thức chuyển đổi đúng:
    # ex_t = P_prev * p_ext / 100
    # in_t = P_prev * p_int / 100
    result['ex_t'] = result['P_prev'] * result['p_ext'] / 100
    result['in_t'] = result['P_prev'] * result['p_int'] / 100
    
    # Kiểm tra tính đúng đắn: P(t) ≈ P(t-1) + ex_t + in_t
    expected_price = result['P_prev'] + result['ex_t'] + result['in_t']
    result['price_error'] = result[price_column] - expected_price
    
    logging.info(f"Sai số trung bình khi kiểm tra: {np.mean(np.abs(result['price_error'].dropna()))}")
    
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
    
    plt.show()
