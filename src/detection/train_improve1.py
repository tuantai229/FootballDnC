#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script cải tiến để huấn luyện mô hình YOLO cho phát hiện cầu thủ và bóng.
Cải tiến: kích thước ảnh lớn hơn, điều chỉnh box/cls loss để phát hiện bóng tốt hơn.
"""

from ultralytics import YOLO

# Đường dẫn tới file cấu hình dataset
data_path = "../../dataset/data.yaml"

# Tải mô hình - sử dụng YOLO11n (nhẹ, nhanh)
model = YOLO("yolo11n.pt")

# Huấn luyện mô hình với các cải tiến
results = model.train(
    data=data_path,            # File cấu hình dataset
    epochs=50,                 # Số lượng epochs
    imgsz=1280,                # Tăng kích thước ảnh lên 1280 để phát hiện bóng tốt hơn
    batch=8,                   # Giảm batch size do tăng kích thước ảnh
    project="../../models/detection",  # Thư mục lưu kết quả
    name="improved_train",     # Tên thử nghiệm
    device="mps",              # Sử dụng GPU trên Mac
    
    # Tham số cải tiến
    box=7.5,                   # Tăng trọng số cho box loss (mặc định: 7.5)
    cls=0.5,                   # Điều chỉnh trọng số cho class loss (mặc định: 0.5)
    
    # Bật một số augmentation cơ bản
    fliplr=0.5,                # Lật ngang ảnh với xác suất 50%
    scale=0.5,                 # Tỷ lệ scale ngẫu nhiên ảnh
    mosaic=1.0,                # Bật augmentation mosaic (ghép 4 ảnh)
    
    # Tham số cho quá trình huấn luyện
    patience=15,               # Dừng sớm nếu không cải thiện sau 15 epochs
    save_period=10,            # Lưu model mỗi 10 epochs
    
    # Tối ưu hóa cho Mac M2
    workers=2,                 # Giảm số worker threads
    half=True,                 # Sử dụng half precision (FP16)
)

print("Huấn luyện cải tiến hoàn thành!")