# Kế hoạch dự án AI Pipeline nhận diện và phân loại số áo cầu thủ

## 1. Phân tích dữ liệu
- **Mục tiêu**: Hiểu cấu trúc, số lượng và chất lượng dữ liệu
- **Công việc**:
  - Phân tích cấu trúc file JSON annotation
  - Thống kê số lượng frame, số lượng cầu thủ mỗi frame
  - Phân tích phân bố của số áo cầu thủ
  - Kiểm tra chất lượng annotation, độ rõ nét của số áo

## 2. Xử lý và chuẩn bị dữ liệu cho Object Detection
- **Mục tiêu**: Chuyển đổi dữ liệu sang định dạng YOLO
- **Công việc**:
  - Viết script trích xuất từng frame từ video
  - Chuyển đổi annotation từ COCO sang định dạng YOLO
  - Tổ chức dữ liệu theo cấu trúc thư mục YOLO chuẩn
  - Chia dữ liệu thành tập train/val

## 3. Huấn luyện Object Detection Model
- **Mục tiêu**: Phát hiện chính xác vị trí cầu thủ (và bóng)
- **Công việc**:
  - Cài đặt Ultralytics (YOLOv11)
  - Cấu hình tham số huấn luyện phù hợp
  - Tiến hành huấn luyện và theo dõi quá trình
  - Đánh giá và cải thiện hiệu suất mô hình

## 4. Chuẩn bị dữ liệu cho Image Classification
- **Mục tiêu**: Tạo bộ dữ liệu cho phân loại số áo
- **Công việc**:
  - Cắt hình ảnh cầu thủ dựa trên bounding box
  - Gán nhãn số áo cho từng hình ảnh
  - Phân tích và cân bằng phân bố các lớp số áo
  - Tổ chức thành tập train/val

## 5. Huấn luyện Image Classification Model
- **Mục tiêu**: Nhận diện chính xác số áo cầu thủ
- **Công việc**:
  - Thiết kế kiến trúc CNN hoặc sử dụng mô hình pretrained
  - Cấu hình quá trình huấn luyện (learning rate, batch size, augmentation)
  - Huấn luyện mô hình và theo dõi quá trình
  - Đánh giá và cải thiện hiệu suất

## 6. Tích hợp pipeline
- **Mục tiêu**: Xây dựng pipeline hoàn chỉnh từ video đến kết quả nhận diện số áo
- **Công việc**:
  - Viết script liên kết hai mô hình
  - Tối ưu hóa tốc độ xử lý
  - Hiển thị kết quả trực quan

## 7. Đánh giá và cải thiện
- **Mục tiêu**: Đảm bảo độ chính xác và hiệu suất
- **Công việc**:
  - Đánh giá toàn bộ pipeline trên tập test
  - Xác định điểm yếu và cải thiện
  - Thử nghiệm thêm các phương pháp nâng cao

## 8. Giải quyết Bonus task (nếu có thời gian)
- **Mục tiêu**: Xây dựng mô hình đa nhiệm vụ
- **Công việc**:
  - Thiết kế mô hình vừa phân loại số áo vừa phân loại màu áo
  - Điều chỉnh dữ liệu và label phù hợp
  - Huấn luyện và đánh giá hiệu suất