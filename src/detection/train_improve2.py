#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script huấn luyện YOLO với các tham số điều chỉnh để cải thiện khả năng phát hiện bóng.
Tập trung vào việc điều chỉnh loss và augmentation để xử lý mất cân bằng lớp.
"""

from ultralytics import YOLO

# Đường dẫn tới file cấu hình dataset
data_path = "../../dataset/data.yaml"

# Tải mô hình - tiếp tục sử dụng YOLO11n
model = YOLO("yolo11n.pt")

# Huấn luyện mô hình với các cải tiến
results = model.train(
    data=data_path,            # File cấu hình dataset
    epochs=80,                 # Tăng số epochs lên 80
    imgsz=1280,                # Giữ nguyên kích thước ảnh lớn
    batch=8,                   # Giữ nguyên batch size
    project="../../models/detection",
    name="balanced_train",     # Tên thử nghiệm mới
    device="mps",
    
    # Điều chỉnh tham số loss
    box=10.0,                  # Tăng trọng số cho box loss (mặc định: 7.5)
    cls=2.0,                   # Tăng mạnh trọng số cho class loss (mặc định: 0.5)
    
    # Điều chỉnh tham số phát hiện đối tượng nhỏ
    nbs=32,                    # Giảm nominal batch size (mặc định: 64)
    
    # Augmentation mạnh hơn
    fliplr=0.5,                # Lật ngang ảnh
    scale=0.7,                 # Tăng tỷ lệ scale
    mosaic=1.0,                # Tiếp tục sử dụng mosaic
    mixup=0.3,                 # Tăng mixup augmentation
    
    # Tham số cho quá trình huấn luyện
    patience=30,               # Tăng patience
    save_period=20,            # Lưu model mỗi 20 epochs
    
    # Tối ưu hóa cho Mac M2
    workers=2,
    half=True,
)

print("Huấn luyện cân bằng lớp hoàn thành!")