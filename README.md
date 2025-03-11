# ETICA-LSTM Hybrid Model for Stock Market Prediction

## Tổng quan

Dự án này tập trung vào việc phát triển một mô hình lai kết hợp giữa phương pháp phân tích thành phần bên trong và xu hướng bên ngoài (ETICA - External Trend and Internal Component Analysis) với mạng trí nhớ dài hạn (LSTM - Long Short-Term Memory) nhằm nâng cao hiệu quả dự đoán thị trường chứng khoán.

Mô hình ETICA-LSTM đã được chứng minh vượt trội so với mô hình LSTM đơn lẻ trong việc dự đoán các chỉ số chứng khoán như S&P 500, NASDAQ và Dow Jones.

## Đặc điểm

-   Kết hợp phương pháp ETICA với mô hình LSTM
-   Phân tích cả thành phần bên trong và xu hướng bên ngoài của thị trường
-   Cải thiện độ chính xác dự đoán so với các mô hình truyền thống
-   Tối ưu hóa cho các chiến lược đầu tư và quản lý rủi ro

## Phương pháp luận

Mô hình ETICA-LSTM hoạt động theo các bước sau:

1. Phân tích thành phần bên trong (Internal Component Analysis): Xác định và phân tích các yếu tố nội tại của chuỗi thời gian chứng khoán
2. Phân tích xu hướng bên ngoài (External Trend Analysis): Xác định các xu hướng và ảnh hưởng từ bên ngoài
3. Kết hợp với LSTM: Sử dụng kết quả phân tích làm đầu vào cho mô hình LSTM
4. Dự đoán: Tạo dự báo chính xác về biến động thị trường

## Kết quả

Nghiên cứu cho thấy mô hình ETICA-LSTM:

-   Vượt trội hơn so với mô hình LSTM đơn lẻ
-   Giúp LSTM tập trung vào các mẫu rõ ràng hơn
-   Nâng cao độ chính xác dự đoán
-   Mở ra hướng tiếp cận mới cho việc dự báo tài chính

## Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## Cách sử dụng

```python
# Import thư viện
from etica_lstm import ETICA_LSTM_Model

# Khởi tạo mô hình
model = ETICA_LSTM_Model()

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Dự đoán
predictions = model.predict(X_test)
```

## Yêu cầu hệ thống

-   Python 3.6+
-   TensorFlow 2.0+
-   NumPy
-   Pandas
-   Matplotlib (cho visualizations)

## Tài liệu tham khảo

-   [Enhancing stock market predictions via hybrid external trend and internal components analysis and long short term memory model](https://www.sciencedirect.com/science/article/pii/S1319157824003410)
-   [LSTM](https://viblo.asia/p/tim-hieu-lstm-bi-quyet-giu-thong-tin-lau-dai-hieu-qua-MG24BaezVz3)
