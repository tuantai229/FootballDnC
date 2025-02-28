#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
from pathlib import Path

# Thư mục gốc của dữ liệu YOLO
yolo_dir = 'data/yolo'

# Kiểm tra tất cả các thư mục train trong YOLO
train_img_dirs = glob.glob(os.path.join(yolo_dir, '**/images/train'), recursive=True)
train_label_dirs = glob.glob(os.path.join(yolo_dir, '**/labels/train'), recursive=True)

print(f"Tìm thấy {len(train_img_dirs)} thư mục images/train")
print(f"Tìm thấy {len(train_label_dirs)} thư mục labels/train")

# Kiểm tra từng thư mục
for img_dir in train_img_dirs:
    match_name = Path(img_dir).parents[1].name
    print(f"\nKiểm tra trận đấu: {match_name}")
    
    # Tìm thư mục labels tương ứng
    label_dir = img_dir.replace('images', 'labels')
    
    if not os.path.exists(label_dir):
        print(f"  ERROR: Không tìm thấy thư mục labels tương ứng: {label_dir}")
        continue
    
    # Đếm số lượng file
    img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    print(f"  Số lượng file ảnh: {len(img_files)}")
    print(f"  Số lượng file label: {len(label_files)}")
    
    # Kiểm tra sự khớp nhau
    img_names = [Path(f).stem for f in img_files]
    label_names = [Path(f).stem for f in label_files]
    
    # Tìm những file ảnh không có label
    img_without_label = [name for name in img_names if name not in label_names]
    if img_without_label:
        print(f"  WARNING: Có {len(img_without_label)} file ảnh không có label tương ứng")
        for name in img_without_label[:5]:  # Chỉ hiển thị 5 file đầu tiên
            print(f"    - {name}")
        if len(img_without_label) > 5:
            print(f"    - ... và {len(img_without_label) - 5} file khác")
    
    # Tìm những file label không có ảnh
    label_without_img = [name for name in label_names if name not in img_names]
    if label_without_img:
        print(f"  WARNING: Có {len(label_without_img)} file label không có ảnh tương ứng")
        for name in label_without_img[:5]:  # Chỉ hiển thị 5 file đầu tiên
            print(f"    - {name}")
        if len(label_without_img) > 5:
            print(f"    - ... và {len(label_without_img) - 5} file khác")
    
    # Kết luận
    if not img_without_label and not label_without_img:
        print("  SUCCESS: Tất cả các file ảnh và label đều khớp nhau hoàn toàn!")