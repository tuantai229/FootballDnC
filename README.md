# AI Pipeline nhận diện và phân loại số áo cầu thủ

Dự án xây dựng pipeline nhận diện và phân loại số áo cầu thủ bóng đá từ video.

## Mô tả dự án

Pipeline bao gồm 2 models:
1. **Object Detection**: Phát hiện vị trí cầu thủ và bóng trong video
2. **Image Classification**: Phân loại số áo của cầu thủ

## Cấu trúc dự án
```
FootballDnC
├── data/                # Thư mục chứa dữ liệu
│   ├── football_test/   # Dữ liệu kiểm thử
│   └── football_train/  # Dữ liệu huấn luyện
├── docs/                # Tài liệu dự án
├── models/              # Lưu trữ model đã huấn luyện
├── notebooks/           # Jupyter notebooks cho phân tích
├── scripts/             # Scripts xử lý dữ liệu
└── src/                 # Mã nguồn chính
    ├── detection/       # Mã nguồn cho Object Detection
    ├── classification/  # Mã nguồn cho Image Classification
    └── pipeline/        # Mã nguồn kết hợp 2 models
```

## Công nghệ sử dụng

- YOLOv11 cho Object Detection
- PyTorch cho Image Classification
- OpenCV cho xử lý ảnh và video