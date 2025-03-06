#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script cơ bản để huấn luyện mô hình YOLO cho phát hiện cầu thủ và bóng.
"""

from ultralytics import YOLO

# Đường dẫn tới file cấu hình dataset
data_path = "../../dataset/data.yaml"

# Tải mô hình - sử dụng YOLO11n (nhẹ, nhanh)
model = YOLO("yolo11n.pt")

# Huấn luyện mô hình với các tham số cơ bản
results = model.train(
    data=data_path,        # File cấu hình dataset
    epochs=50,             # Số lượng epochs
    imgsz=640,             # Kích thước ảnh đầu vào
    batch=16,              # Kích thước batch
    project="../../models/detection",  # Thư mục lưu kết quả
    name="basic_train",    # Tên thử nghiệm
    device="mps"           # Sử dụng GPU trên Mac (hoặc "0" cho NVIDIA GPU, "cpu" cho CPU)
)

print("Huấn luyện hoàn thành!")