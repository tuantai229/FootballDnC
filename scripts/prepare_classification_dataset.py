#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để chuẩn bị dataset cho bài toán phân loại số áo cầu thủ.
Trích xuất crop ảnh cầu thủ từ các frames và annotation.
"""

import os
import cv2
import json
import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def extract_player_crops(data_dir, annotation_dir, frame_dir, output_dir, 
                         min_samples=20, augment=False, resize_dim=None):
    """
    Trích xuất crop ảnh cầu thủ từ các frames và annotation
    
    Args:
        data_dir (str): Thư mục gốc chứa dữ liệu
        annotation_dir (str): Thư mục chứa file annotation
        frame_dir (str): Thư mục chứa frames đã trích xuất
        output_dir (str): Thư mục đầu ra cho dataset
        min_samples (int): Số lượng mẫu tối thiểu cho mỗi class
        augment (bool): Có áp dụng data augmentation không
        resize_dim (tuple): Kích thước để resize ảnh crop (width, height)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo thư mục tạm để lưu tất cả crops theo số áo
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Dict để lưu thống kê số lượng mẫu cho mỗi số áo
    jersey_counts = defaultdict(int)
    # Dict để lưu đường dẫn ảnh cho mỗi số áo
    jersey_images = defaultdict(list)
    # Dict để lưu số mẫu theo team_jersey_color
    team_color_counts = defaultdict(int)
    
    # Đường dẫn tới thư mục dataset
    print(f"Đang xử lý dữ liệu từ: {data_dir}")
    
    # Duyệt qua tất cả các file annotation
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, f"football_{split}")
        
        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(split_dir):
            print(f"Thư mục {split_dir} không tồn tại, bỏ qua...")
            continue
            
        print(f"\nĐang xử lý split: {split}")
        
        # Lấy danh sách các thư mục match
        match_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        
        # Xử lý từng match
        for match_dir in match_dirs:
            print(f"\nXử lý match: {match_dir}")
            
            # Đường dẫn đến file annotation và video
            anno_path = os.path.join(split_dir, match_dir, f"{match_dir}.json")
            
            # Kiểm tra file annotation có tồn tại không
            if not os.path.exists(anno_path):
                print(f"File annotation {anno_path} không tồn tại, bỏ qua...")
                continue
                
            print(f"  Annotation: {anno_path}")
            
            # Đọc file annotation
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            
            # Tạo mapping từ image_id sang frame_id
            image_map = {}
            filename_map = {}
            for img in annotations['images']:
                # Frame ID thường được lưu trong "file_name" dưới dạng "frame_XXXXXX.PNG"
                frame_id = int(img['file_name'].split('_')[-1].split('.')[0])
                image_map[img['id']] = frame_id
                filename_map[frame_id] = img['file_name']
            
            # Tìm tất cả các frames đã trích xuất cho match này
            match_frames = []
            for frame_name in os.listdir(frame_dir):
                if frame_name.startswith(f"{match_dir}_frame_") and frame_name.endswith((".jpg", ".png")):
                    frame_id = int(frame_name.split('_')[-1].split('.')[0])
                    match_frames.append((frame_id, os.path.join(frame_dir, frame_name)))
            
            if not match_frames:
                print(f"  Không tìm thấy frames cho match {match_dir}, bỏ qua...")
                continue
                
            print(f"  Tìm thấy {len(match_frames)} frames")
            
            # Lập danh sách các annotation của cầu thủ (category_id=4) có số áo
            player_annotations = []
            for ann in annotations['annotations']:
                if ann['category_id'] == 4:  # Cầu thủ
                    if 'attributes' in ann and 'jersey_number' in ann['attributes']:
                        jersey_number = ann['attributes']['jersey_number']
                        if jersey_number == "invisible" or not jersey_number.isdigit():
                            jersey_number = "0"  # Gán class 0 cho những số áo không nhìn thấy
                        
                        # Lưu thêm thông tin về màu áo
                        team_color = "unknown"
                        if 'team_jersey_color' in ann['attributes']:
                            team_color = ann['attributes']['team_jersey_color']
                        
                        # Lưu thông tin về độ rõ của số áo
                        number_visible = "invisible"
                        if 'number_visible' in ann['attributes']:
                            number_visible = ann['attributes']['number_visible']
                        
                        # Thêm vào danh sách
                        player_annotations.append({
                            'image_id': ann['image_id'],
                            'frame_id': image_map[ann['image_id']],
                            'bbox': ann['bbox'],
                            'jersey_number': jersey_number,
                            'team_color': team_color,
                            'number_visible': number_visible
                        })
            
            print(f"  Tìm thấy {len(player_annotations)} annotations cầu thủ có thông tin số áo")
            
            # Tạo ánh xạ từ frame_id sang đường dẫn frame
            frame_path_map = {frame_id: path for frame_id, path in match_frames}
            
            # Xử lý từng annotation
            for ann in tqdm(player_annotations, desc=f"Trích xuất crop cho {match_dir}"):
                frame_id = ann['frame_id']
                
                # Bỏ qua nếu không tìm thấy frame tương ứng
                if frame_id not in frame_path_map:
                    continue
                
                # Lấy đường dẫn đến frame
                frame_path = frame_path_map[frame_id]
                
                # Đọc frame
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                # Lấy bounding box
                x, y, w, h = ann['bbox']
                x, y, w, h = int(x), int(y), int(w), int(h)
                
                # Tránh các bbox nằm ngoài kích thước ảnh
                img_h, img_w = frame.shape[:2]
                x = max(0, x)
                y = max(0, y)
                w = min(w, img_w - x)
                h = min(h, img_h - y)
                
                # Bỏ qua nếu bbox quá nhỏ
                if w <= 5 or h <= 5:
                    continue
                
                # Cắt crop từ frame
                crop = frame[y:y+h, x:x+w]
                
                # Resize nếu cần
                if resize_dim is not None:
                    crop = cv2.resize(crop, resize_dim)
                
                # Lấy số áo và màu áo
                jersey_number = ann['jersey_number']
                team_color = ann['team_color']
                number_visible = ann['number_visible']
                
                # Chỉ lưu những ảnh có số áo rõ ràng
                if number_visible == "visible" or jersey_number == "0":
                    # Xác định class name theo đề bài
                    class_name = jersey_number
                    if jersey_number.isdigit():
                        jersey_num = int(jersey_number)
                        if jersey_num == 0:
                            class_name = "0"  # Không nhìn thấy số áo
                        elif 1 <= jersey_num <= 10:
                            class_name = str(jersey_num)  # Số từ 1-10 giữ nguyên
                        else:
                            class_name = "11"  # Số từ 11 trở lên gộp vào class 11
                    
                    # Tạo thư mục cho class nếu chưa có
                    class_dir = os.path.join(temp_dir, class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # Tạo tên file cho crop
                    crop_name = f"{match_dir}_frame_{frame_id:06d}_x{x}_y{y}.jpg"
                    crop_path = os.path.join(class_dir, crop_name)
                    
                    # Lưu crop
                    cv2.imwrite(crop_path, crop)
                    
                    # Cập nhật thống kê
                    jersey_counts[class_name] += 1
                    jersey_images[class_name].append(crop_path)
                    team_color_counts[team_color] += 1
                    
                    # Thực hiện data augmentation nếu cần
                    if augment and int(class_name) > 0:  # Chỉ augment cho các lớp số áo thật
                        # Lật ngang
                        if random.random() < 0.5:
                            flip_crop = cv2.flip(crop, 1)
                            flip_name = f"{match_dir}_frame_{frame_id:06d}_x{x}_y{y}_flip.jpg"
                            flip_path = os.path.join(class_dir, flip_name)
                            cv2.imwrite(flip_path, flip_crop)
                            jersey_counts[class_name] += 1
                            jersey_images[class_name].append(flip_path)
                        
                        # Xoay nhẹ
                        if random.random() < 0.3:
                            angle = random.uniform(-15, 15)
                            h, w = crop.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, angle, 1.0)
                            rotated_crop = cv2.warpAffine(crop, M, (w, h))
                            rot_name = f"{match_dir}_frame_{frame_id:06d}_x{x}_y{y}_rot.jpg"
                            rot_path = os.path.join(class_dir, rot_name)
                            cv2.imwrite(rot_path, rotated_crop)
                            jersey_counts[class_name] += 1
                            jersey_images[class_name].append(rot_path)
                        
                        # Điều chỉnh độ sáng
                        if random.random() < 0.3:
                            brightness = 0.5 + random.random() * 1.0  # 0.5 to 1.5
                            bright_crop = cv2.convertScaleAbs(crop, alpha=brightness, beta=0)
                            bright_name = f"{match_dir}_frame_{frame_id:06d}_x{x}_y{y}_bright.jpg"
                            bright_path = os.path.join(class_dir, bright_name)
                            cv2.imwrite(bright_path, bright_crop)
                            jersey_counts[class_name] += 1
                            jersey_images[class_name].append(bright_path)
    
    # Hiển thị thống kê
    print("\n=== THỐNG KÊ SỐ LƯỢNG MẪU ===")
    for jersey_num in sorted(jersey_counts.keys(), key=lambda x: int(x)):
        print(f"Số áo {jersey_num}: {jersey_counts[jersey_num]} mẫu")
    
    print("\n=== THỐNG KÊ MÀU ÁO ===")
    for color, count in team_color_counts.items():
        print(f"Màu {color}: {count} mẫu")
    
    # Hiển thị biểu đồ phân bố số lượng mẫu
    sorted_jerseys = sorted(jersey_counts.keys(), key=lambda x: int(x))
    counts = [jersey_counts[j] for j in sorted_jerseys]
    
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_jerseys, counts)
    plt.xlabel('Số áo')
    plt.ylabel('Số lượng mẫu')
    plt.title('Phân bố số lượng mẫu theo số áo')
    plt.savefig(os.path.join(output_dir, 'jersey_distribution.png'))
    
    # Tạo thư mục train và val
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Tỷ lệ chia train/val
    val_ratio = 0.2
    
    # Chia tập train/val cho mỗi class
    print("\n=== CHIA TẬP TRAIN/VAL ===")
    for jersey_num in sorted_jerseys:
        images = jersey_images[jersey_num]
        count = len(images)
        
        if count < min_samples:
            print(f"Bỏ qua lớp {jersey_num} vì chỉ có {count} mẫu (yêu cầu tối thiểu {min_samples})")
            continue
        
        # Shuffle danh sách ảnh
        random.shuffle(images)
        
        # Chia tập train/val
        val_size = int(count * val_ratio)
        val_images = images[:val_size]
        train_images = images[val_size:]
        
        # Tạo thư mục cho class trong train và val
        train_class_dir = os.path.join(train_dir, jersey_num)
        val_class_dir = os.path.join(val_dir, jersey_num)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy ảnh vào thư mục tương ứng
        for img_path in train_images:
            dst_path = os.path.join(train_class_dir, os.path.basename(img_path))
            shutil.copy(img_path, dst_path)
        
        for img_path in val_images:
            dst_path = os.path.join(val_class_dir, os.path.basename(img_path))
            shutil.copy(img_path, dst_path)
        
        print(f"Lớp {jersey_num}: {len(train_images)} mẫu train, {len(val_images)} mẫu val")
    
    # Xóa thư mục tạm
    shutil.rmtree(temp_dir)
    
    # Tạo các file classes.txt và đếm số lượng ảnh trong mỗi split
    classes = [c for c in sorted_jerseys if len(os.listdir(os.path.join(train_dir, c))) > 0]
    
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for c in classes:
            f.write(f"{c}\n")
    
    # Đếm tổng số ảnh
    train_count = sum([len(os.listdir(os.path.join(train_dir, c))) for c in classes])
    val_count = sum([len(os.listdir(os.path.join(val_dir, c))) for c in classes])
    
    print("\n=== TỔNG KẾT ===")
    print(f"Số lượng classes: {len(classes)}")
    print(f"Tổng số ảnh train: {train_count}")
    print(f"Tổng số ảnh val: {val_count}")
    print(f"\nDataset đã được tạo tại: {output_dir}")
    print("Cấu trúc dataset:")
    print(f"{output_dir}/")
    print("├── train/")
    for c in classes[:5]:
        print(f"│   ├── {c}/")
    if len(classes) > 5:
        print("│   ├── ...")
    print("├── val/")
    for c in classes[:5]:
        print(f"│   ├── {c}/")
    if len(classes) > 5:
        print("│   ├── ...")
    print("├── classes.txt")
    print("└── jersey_distribution.png")

def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị dataset cho phân loại số áo cầu thủ')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Thư mục gốc chứa dữ liệu')
    parser.add_argument('--frame-dir', type=str, default='./dataset/train/images',
                        help='Thư mục chứa frames đã trích xuất')
    parser.add_argument('--output-dir', type=str, default='./classification_dataset',
                        help='Thư mục đầu ra cho dataset phân loại')
    parser.add_argument('--min-samples', type=int, default=20,
                        help='Số lượng mẫu tối thiểu cho mỗi class')
    parser.add_argument('--augment', action='store_true',
                        help='Có áp dụng data augmentation không')
    parser.add_argument('--resize-width', type=int, default=None,
                        help='Chiều rộng để resize crop (mặc định: giữ nguyên kích thước)')
    parser.add_argument('--resize-height', type=int, default=None,
                        help='Chiều cao để resize crop (mặc định: giữ nguyên kích thước)')
    
    args = parser.parse_args()
    
    # Xác định kích thước resize
    resize_dim = None
    if args.resize_width is not None and args.resize_height is not None:
        resize_dim = (args.resize_width, args.resize_height)
    
    # Trích xuất crops
    extract_player_crops(
        data_dir=args.data_dir,
        annotation_dir=args.data_dir,
        frame_dir=args.frame_dir,
        output_dir=args.output_dir,
        min_samples=args.min_samples,
        augment=args.augment,
        resize_dim=resize_dim
    )

if __name__ == "__main__":
    main()