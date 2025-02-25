# Tìm hiểu về Ultralytics/YOLO

## Giới thiệu về YOLO và Ultralytics

YOLO (You Only Look Once) là một trong những thuật toán phát hiện đối tượng (object detection) phổ biến nhất trong lĩnh vực thị giác máy tính (Computer Vision). Thuật toán này nổi bật vì tốc độ nhanh và hiệu suất tốt, đặc biệt trong các ứng dụng thời gian thực.

Ultralytics là công ty đã phát triển và duy trì các phiên bản YOLO gần đây, đặc biệt là từ YOLOv5 trở đi. Họ đã xây dựng một hệ sinh thái rất mạnh mẽ xung quanh các mô hình YOLO, bao gồm công cụ đào tạo, tối ưu hóa, và triển khai.

## So sánh các phiên bản YOLOv5, YOLOv8 và YOLOv11

### YOLOv5 (2020)

- **Đặc điểm**: Là phiên bản YOLO đầu tiên được Ultralytics phát triển, viết bằng PyTorch
- **Kiến trúc**: Backbone CSP (Cross Stage Partial Network), neck là PANet, head là YOLO
- **Hiệu suất**: Cải thiện đáng kể về tốc độ và độ chính xác so với các phiên bản trước (YOLOv3, YOLOv4)
- **Mô hình**: Có 5 biến thể chính - YOLOv5n (nano), YOLOv5s (small), YOLOv5m (medium), YOLOv5l (large), và YOLOv5x (xlarge)
- **Ưu điểm**: Dễ sử dụng, có cộng đồng lớn, tài liệu rõ ràng, tương đối nhẹ
- **Nhược điểm**: Kiến trúc cũ hơn so với các phiên bản mới

### YOLOv8 (2023)

- **Đặc điểm**: Hoàn toàn được viết lại so với YOLOv5, với nhiều cải tiến
- **Kiến trúc**: Backbone CSPDarknet mới, với C2f module thay thế C3, sử dụng anchor-free
- **Nhiệm vụ hỗ trợ**: Không chỉ object detection mà còn hỗ trợ instance segmentation, pose estimation, classification, và tracking
- **Hiệu suất**: Cải thiện cả về tốc độ và độ chính xác so với YOLOv5
- **Mô hình**: Cũng có 5 biến thể tương tự YOLOv5
- **Ưu điểm**: API thân thiện hơn, hỗ trợ nhiều nhiệm vụ, architecture hiện đại hơn
- **Nhược điểm**: Cần tài nguyên nhiều hơn YOLOv5 cho các mô hình cùng kích thước

### YOLOv11 (2024)

- **Đặc điểm**: Phiên bản mới nhất từ Ultralytics, giới thiệu vào năm 2024
- **Kiến trúc**: Nâng cấp lớn từ YOLOv8, tích hợp các kỹ thuật state-of-the-art mới nhất
- **Hiệu năng**: Cải thiện đáng kể về độ chính xác, đặc biệt với các đối tượng nhỏ và trong điều kiện ánh sáng khó
- **Cải tiến chính**:
  - Tích hợp kỹ thuật làm giàu đặc trưng (feature enrichment) mạnh mẽ hơn
  - Cơ chế attention cải tiến
  - Tối ưu hóa cho việc phát hiện đối tượng ở các tỷ lệ khác nhau
  - Giảm thiểu số lượng tham số trong khi vẫn duy trì hoặc tăng hiệu suất
- **Ưu điểm**: Hiệu suất cao nhất trong các phiên bản YOLO, hỗ trợ nhiều tác vụ mới
- **Nhược điểm**: Là phiên bản mới nhất nên tài liệu và cộng đồng hỗ trợ có thể chưa nhiều như các phiên bản trước

## So sánh hiệu suất

| Phiên bản | mAP (COCO val) | Tốc độ (FPS) | Kích thước mô hình | Đặc điểm nổi bật |
|-----------|----------------|--------------|---------------------|------------------|
| YOLOv5s   | ~37%           | ~155         | ~14MB               | Cân bằng giữa tốc độ và độ chính xác |
| YOLOv8s   | ~44%           | ~160         | ~22MB               | Cải thiện đáng kể độ chính xác |
| YOLOv11s  | ~49%           | ~155         | ~25MB               | Hiệu suất cao nhất, đặc biệt với đối tượng khó |

## Ý nghĩa thông số so sánh

### mAP (Mean Average Precision)

**mAP (Mean Average Precision)** là chỉ số quan trọng nhất để đánh giá độ chính xác của mô hình object detection như YOLO:

- **Định nghĩa**: mAP là giá trị trung bình của Average Precision (AP) trên tất cả các lớp đối tượng
- **COCO val**: Chỉ số này được tính trên tập dữ liệu kiểm định COCO (Common Objects in Context), một tập dữ liệu chuẩn trong computer vision với 80 lớp đối tượng phổ biến
- **Thang đo**: Được thể hiện bằng phần trăm (%), càng cao càng tốt

**Chi tiết về cách tính mAP**:
1. **Precision**: Tỷ lệ dự đoán đúng trên tổng số dự đoán (TP/(TP+FP))
2. **Recall**: Tỷ lệ dự đoán đúng trên tổng số đối tượng thực (TP/(TP+FN))
3. **AP (Average Precision)**: Diện tích dưới đường cong Precision-Recall cho mỗi lớp
4. **mAP**: Trung bình của AP trên tất cả các lớp

**Ý nghĩa của các giá trị trong bảng**:
- YOLOv5s: ~37% - Mô hình có thể phát hiện chính xác khoảng 37% đối tượng trong tập COCO
- YOLOv8s: ~44% - Cải thiện 7% so với YOLOv5s
- YOLOv11s: ~49% - Cải thiện thêm 5% so với YOLOv8s

Mức cải thiện 5-7% là khá đáng kể trong lĩnh vực object detection, thể hiện sự tiến bộ thực sự giữa các phiên bản.

### Tốc độ (FPS - Frames Per Second)

**FPS** là số khung hình mà mô hình có thể xử lý trong một giây:

- **Định nghĩa**: Số lượng ảnh/frame mà mô hình có thể phân tích và phát hiện đối tượng trong 1 giây
- **Môi trường đo**: Thường được đo trên phần cứng chuẩn (như NVIDIA GPU cụ thể) và kích thước đầu vào chuẩn (thường là 640x640 pixels)
- **Ý nghĩa**: Chỉ số này cực kỳ quan trọng cho ứng dụng thời gian thực

**Giá trị FPS trong bảng**:
- YOLOv5s: ~155 FPS - Có thể xử lý 155 frame/giây
- YOLOv8s: ~160 FPS - Hơi nhanh hơn YOLOv5s
- YOLOv11s: ~155 FPS - Tương đương YOLOv5s

**Ý nghĩa thực tế**:
- Video tiêu chuẩn thường có 24-30 FPS
- Video HD/4K thường có 30-60 FPS
- Tất cả các phiên bản YOLO đều có thể xử lý video thời gian thực với tốc độ cao hơn nhiều so với tốc độ khung hình tiêu chuẩn

### Mối quan hệ giữa mAP và FPS

Thường có sự đánh đổi (trade-off) giữa độ chính xác (mAP) và tốc độ (FPS):
- Mô hình có mAP cao hơn thường chậm hơn
- Mô hình nhanh hơn thường có mAP thấp hơn

**Điểm đáng chú ý từ bảng so sánh**:
- YOLOv8 và YOLOv11 đã phá vỡ quy luật này bằng cách cải thiện mAP mà không ảnh hưởng nhiều đến tốc độ
- YOLOv11 đặc biệt ấn tượng vì cải thiện mAP lên 49% trong khi vẫn duy trì tốc độ tương đương YOLOv5

### Ý nghĩa trong dự án

Trong dự án phát hiện số áo cầu thủ, các thông số này có ý nghĩa như sau:

- **mAP cao**: Sẽ giúp phát hiện chính xác hơn vị trí của cầu thủ và bóng, đặc biệt trong các tình huống phức tạp (cầu thủ chồng lên nhau, bị che khuất một phần)
- **FPS cao**: Cho phép xử lý video với độ phân giải cao (3840x1200) mà không bị giật lag

Vì video dữ liệu có độ phân giải rất cao (3840x1200), nên hiệu suất thực tế có thể thấp hơn các con số trong bảng (thường đo ở 640x640). Do đó, YOLOv8 hoặc YOLOv11 với mAP cao hơn có thể sẽ mang lại kết quả tốt hơn.
