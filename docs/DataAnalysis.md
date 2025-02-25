# Phân tích và khám phá dữ liệu

## Cấu trúc dữ liệu

### Tổ chức
```
FootballDnC
└── data
    ├── football_test
    │   ├── Match_1864_1_0_subclip
    │   │   ├── Match_1864_1_0_subclip.json
    │   │   └── Match_1864_1_0_subclip.mp4
    │   └── Match_1953_2_0_subclip
    │       ├── Match_1953_2_0_subclip.json
    │       └── Match_1953_2_0_subclip.mp4
    └── football_train
        ├── Match_1824_1_0_subclip_3
        │   ├── Match_1824_1_0_subclip_3.json
        │   └── Match_1824_1_0_subclip_3.mp4
        ├── Match_1951_1_0_subclip
        │   ├── Match_1951_1_0_subclip.json
        │   └── Match_1951_1_0_subclip.mp4
        ├── Match_2022_3_0_subclip
        │   ├── Match_2022_3_0_subclip.json
        │   └── Match_2022_3_0_subclip.mp4
        ├── Match_2023_3_0_subclip
        │   ├── Match_2023_3_0_subclip.json
        │   └── Match_2023_3_0_subclip.mp4
        └── Match_2031_5_0_test
            └── Match_2031_5_0_test.mp4
```

### Thống kê tổng quan
- **Số lượng video train**: 5
- **Số lượng video test**: 2
- **Tổng số video**: 7
- **Số video có file annotation**: 6 (1 video train không có file JSON)
- **Kích thước video**: 3840 x 1200 pixels
- **Tốc độ khung hình**: 25 fps
- **Thời lượng mỗi video**: 1 phút (~1500 frames)

## Phân tích cấu trúc file JSON

### Các thành phần chính
1. **Licenses**: Thông tin về bản quyền
2. **Info**: Thông tin mô tả về bộ dữ liệu
3. **Categories**: Danh sách các loại đối tượng
   - id: 1, name: field (sân)
   - id: 2, name: bystander (người ngoài)
   - id: 3, name: ball (bóng)
   - id: 4, name: player (cầu thủ)
4. **Images**: Thông tin về các frames
   - id: ID của frame
   - width: Chiều rộng (3840)
   - height: Chiều cao (1200)
   - file_name: Tên file (frame_XXXXXX.PNG)
5. **Annotations**: Thông tin về các đối tượng trong mỗi frame
   - id: ID của annotation
   - image_id: ID của frame chứa đối tượng
   - category_id: Loại đối tượng (1, 2, 3, 4)
   - area: Diện tích của bounding box
   - bbox: Tọa độ bounding box [x, y, width, height]
   - attributes: (chỉ có với category_id=4 - player)
     + object_ID: ID định danh của cầu thủ
     + facing_side: Hướng nhìn của cầu thủ
     + jersey_number: Số áo của cầu thủ
     + team_jersey_color: Màu áo đội (white/dark)
     + occluded: Có bị che khuất không (occluded/no_occluded)
     + number_visible: Số áo có nhìn thấy được không (visible/invisible)

## Phân tích đặc điểm cầu thủ

### Phân tích số áo cầu thủ
Theo đề bài, phân bố số áo cầu thủ:
- Các số từ 1-10 xuất hiện nhiều hơn các số khác
- Có trường hợp số áo không nhìn thấy rõ ('invisible')
- Đề xuất phân loại số áo thành:
  + Class 0: Không nhìn rõ số áo
  + Class 1-10: Dành cho các số từ 1 đến 10
  + Class 11: Dành cho các số từ 11 trở lên

### Phân tích màu áo
- Chỉ có 2 loại màu áo: trắng (white) và đen (dark)
- Có thể sử dụng để phân biệt đội bóng

## Đánh giá thách thức và dự kiến

### Thách thức
1. Kích thước video lớn (3840x1200), cần xử lý hiệu quả
2. Có thể có nhiều đối tượng player trong một frame
3. Số áo có thể không nhìn thấy rõ trong nhiều trường hợp
4. Phân bố số áo có thể không đồng đều

### Dự kiến cho Object Detection
1. Trích xuất từng frame từ video
2. Lọc ra chỉ giữ lại annotation của cầu thủ (category_id=4) và bóng (category_id=3)
3. Chuyển đổi từ định dạng COCO sang định dạng YOLO
4. Có thể cần resize ảnh xuống kích thước nhỏ hơn để tăng tốc huấn luyện

### Dự kiến cho Image Classification
1. Cắt hình ảnh cầu thủ dựa trên bounding box
2. Lọc bỏ các trường hợp số áo không nhìn thấy khi huấn luyện
3. Cân bằng dữ liệu để đảm bảo mỗi lớp có đủ mẫu
4. Cần xây dựng thêm lớp 'không nhìn thấy số áo'
5. Có thể sử dụng data augmentation để tăng cường dữ liệu

### Dự kiến cho Bonus task
1. Thiết kế mô hình với 2 heads: một cho số áo, một cho màu áo
2. Hoặc thiết kế mô hình multi-label với (n+2) classes:
   - n classes cho số áo (0-10, 11+)
   - 2 classes cho màu áo (white, dark)

## Công việc tiếp theo cần thực hiện

1. Viết script để trích xuất frames từ video
2. Viết script để chuyển đổi annotation từ COCO sang YOLO
3. Chuẩn bị dữ liệu cho huấn luyện Object Detection