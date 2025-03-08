#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để chuẩn bị dataset YOLO từ dữ liệu bóng đá
Bao gồm:
- Trích xuất frames từ video
- Chuyển đổi annotation từ COCO sang YOLO
- Chia tập train/val
- Tạo cấu trúc thư mục phù hợp với Ultralytics YOLO
"""

import os
import cv2
import argparse
import json
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

def extract_frames(video_path, output_dir, frame_interval=1, resize_dim=None):
    """
    Trích xuất frame từ video và lưu vào thư mục đầu ra
    
    Args:
        video_path (str): Đường dẫn đến file video
        output_dir (str): Thư mục đích để lưu các frame
        frame_interval (int): Chỉ lưu 1 frame sau mỗi frame_interval frames
        resize_dim (tuple, optional): Kích thước để resize frame (width, height)
    
    Returns:
        list: Danh sách các đường dẫn đến frame đã trích xuất
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy tên video để tạo prefix cho tên file
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Mở video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return []
    
    # Đọc thông tin về video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Danh sách lưu đường dẫn đến các frame
    frame_paths = []
    extracted_frame_indices = []
    
    # Đọc và lưu các frame
    with tqdm(total=frame_count, desc=f"Đang trích xuất frames") as pbar:
        frame_idx = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Chỉ lưu frame nếu đạt đến interval
            if frame_idx % frame_interval == 0:
                # Resize frame nếu cần
                if resize_dim is not None:
                    frame = cv2.resize(frame, resize_dim)
                
                # Tên file với prefix từ tên video để tránh trùng lặp
                frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Lưu frame
                cv2.imwrite(frame_path, frame)
                
                frame_paths.append(frame_path)
                extracted_frame_indices.append(frame_idx)
            
            frame_idx += 1
            pbar.update(1)
    
    # Giải phóng tài nguyên
    video.release()
    
    return frame_paths, extracted_frame_indices

def convert_coco_to_yolo(annotation_path, frame_indices, output_dir, image_width, image_height):
    """
    Chuyển đổi annotation từ định dạng COCO sang định dạng YOLO
    
    Args:
        annotation_path (str): Đường dẫn đến file annotation COCO
        frame_indices (list): Danh sách các frame index đã trích xuất
        output_dir (str): Thư mục đầu ra cho label YOLO
        image_width (int): Chiều rộng của ảnh
        image_height (int): Chiều cao của ảnh
    
    Returns:
        list: Danh sách các đường dẫn đến file label đã tạo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc file annotation
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    # Lấy tên video để tạo prefix cho tên file label
    video_name = os.path.basename(annotation_path).split('.')[0]
    
    # Tạo mapping từ image_id sang frame_id
    image_map = {}
    for img in annotations['images']:
        # Frame ID thường được lưu trong "file_name" dưới dạng "frame_XXXXXX.PNG"
        frame_id = int(img['file_name'].split('_')[-1].split('.')[0])
        image_map[img['id']] = frame_id
    
    # Tạo map từ category_id sang index mới (chỉ quan tâm ball và player)
    # category_id=3: ball, category_id=4: player
    category_map = {3: 0, 4: 1}  # Chuyển sang index 0: ball, 1: player
    
    # Danh sách lưu đường dẫn đến các file label
    label_paths = []
    
    # Tạo một dict để lưu annotations theo frame_id
    frame_annotations = {}
    for ann in annotations['annotations']:
        if ann['category_id'] not in [3, 4]:  # Chỉ lấy ball và player
            continue
        
        image_id = ann['image_id']
        frame_id = image_map[image_id]
        
        if frame_id not in frame_annotations:
            frame_annotations[frame_id] = []
        
        # Lấy thông tin bounding box [x, y, width, height] (pixel)
        x, y, width, height = ann['bbox']
        
        # Chuyển sang định dạng YOLO: [class_id, x_center, y_center, width, height] (normalized)
        x_center = (x + width / 2) / image_width
        y_center = (y + height / 2) / image_height
        width_norm = width / image_width
        height_norm = height / image_height
        
        # Giới hạn giá trị trong khoảng [0, 1]
        x_center = min(max(x_center, 0.0), 1.0)
        y_center = min(max(y_center, 0.0), 1.0)
        width_norm = min(max(width_norm, 0.0), 1.0)
        height_norm = min(max(height_norm, 0.0), 1.0)
        
        # Lấy class mới
        class_id = category_map[ann['category_id']]
        
        # Thêm vào danh sách
        frame_annotations[frame_id].append([class_id, x_center, y_center, width_norm, height_norm])
    
    # Lưu annotation cho các frame đã trích xuất
    for frame_idx in frame_indices:
        if frame_idx in frame_annotations:
            # Tạo tên file label tương ứng với frame
            label_filename = f"{video_name}_frame_{frame_idx:06d}.txt"
            label_path = os.path.join(output_dir, label_filename)
            
            # Lưu annotations vào file
            with open(label_path, 'w') as f:
                for ann in frame_annotations[frame_idx]:
                    f.write(' '.join(map(str, ann)) + '\n')
            
            label_paths.append(label_path)
    
    return label_paths

def prepare_dataset(args):
    """
    Chuẩn bị dataset cho YOLO
    
    Args:
        args: Các tham số từ command line
    """
    # Tạo thư mục output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Tạo thư mục train và val
    train_img_dir = os.path.join(args.output_dir, 'train', 'images')
    train_label_dir = os.path.join(args.output_dir, 'train', 'labels')
    val_img_dir = os.path.join(args.output_dir, 'val', 'images')
    val_label_dir = os.path.join(args.output_dir, 'val', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    # Tạo thư mục tạm để lưu trữ dữ liệu trung gian
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Danh sách tất cả các cặp (frame, label)
    all_data = []
    
    # Xử lý từng split (train/test)
    for split in args.splits:
        print(f"Đang xử lý split: {split}")
        split_dir = os.path.join(args.data_dir, f"football_{split}")
        
        # Lấy danh sách các thư mục match
        match_dirs = [d for d in os.listdir(split_dir) 
                    if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        
        # Xử lý từng match
        for match_dir in match_dirs:
            print(f"Xử lý match: {match_dir}")
            
            # Đường dẫn đến video và annotation
            video_path = os.path.join(split_dir, match_dir, f"{match_dir}.mp4")
            anno_path = os.path.join(split_dir, match_dir, f"{match_dir}.json")
            
            if not os.path.exists(video_path) or not os.path.exists(anno_path):
                print(f"Bỏ qua: Không tìm thấy cả video và annotation cho {match_dir}")
                continue
            
            print(f"Video: {video_path}")
            print(f"Annotation: {anno_path}")
            
            # Tạo thư mục tạm cho match này
            match_temp_dir = os.path.join(temp_dir, match_dir)
            image_temp_dir = os.path.join(match_temp_dir, 'images')
            label_temp_dir = os.path.join(match_temp_dir, 'labels')
            os.makedirs(image_temp_dir, exist_ok=True)
            os.makedirs(label_temp_dir, exist_ok=True)
            
            # Trích xuất frames
            resize_dim = None
            if args.resize_width and args.resize_height:
                resize_dim = (args.resize_width, args.resize_height)
            
            frame_paths, frame_indices = extract_frames(
                video_path, 
                image_temp_dir, 
                frame_interval=args.frame_interval,
                resize_dim=resize_dim
            )
            
            print(f"Đã trích xuất {len(frame_paths)} frames")
            
            if not frame_paths:
                continue
            
            # Xác định kích thước ảnh để chuẩn hóa annotation
            img_width = args.resize_width if args.resize_width else 3840
            img_height = args.resize_height if args.resize_height else 1200
            
            # Chuyển đổi annotation sang định dạng YOLO
            label_paths = convert_coco_to_yolo(
                anno_path, 
                frame_indices, 
                label_temp_dir,
                img_width, 
                img_height
            )
            
            # Tạo danh sách cặp (frame, label)
            for frame_path in frame_paths:
                frame_filename = os.path.basename(frame_path)
                label_filename = os.path.splitext(frame_filename)[0] + '.txt'
                label_path = os.path.join(label_temp_dir, label_filename)
                
                # Chỉ thêm vào danh sách nếu có label tương ứng
                if os.path.exists(label_path):
                    all_data.append((frame_path, label_path))
    
    # Shuffle dữ liệu
    random.shuffle(all_data)
    
    # Chia tập train/val
    val_size = int(len(all_data) * args.val_split)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    
    print(f"Tập train: {len(train_data)} frames")
    print(f"Tập val: {len(val_data)} frames")
    
    # Copy dữ liệu vào thư mục cuối cùng
    for src_img, src_label in train_data:
        dst_img = os.path.join(train_img_dir, os.path.basename(src_img))
        dst_label = os.path.join(train_label_dir, os.path.basename(src_label))
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
    
    for src_img, src_label in val_data:
        dst_img = os.path.join(val_img_dir, os.path.basename(src_img))
        dst_label = os.path.join(val_label_dir, os.path.basename(src_label))
        shutil.copy(src_img, dst_img)
        shutil.copy(src_label, dst_label)
    
    # Tạo file data.yaml
    classes = ['ball', 'player']
    yaml_content = f"""# YOLOv11 Dataset đã được tạo bởi prepare_yolo_dataset.py
# Đường dẫn gốc
path: {os.path.abspath(args.output_dir)}
train: train/images
val: val/images

# Classes
nc: {len(classes)}  # số lượng class
names: {classes}  # tên các class
"""
    
    with open(os.path.join(args.output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    
    # Xóa thư mục tạm
    shutil.rmtree(temp_dir)
    
    print(f"\nHoàn thành! Dataset đã được tạo tại: {args.output_dir}")
    print("Cấu trúc dataset:")
    print(f"{args.output_dir}/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── val/")
    print("│   ├── images/")
    print("│   └── labels/")
    print("├── classes.txt")
    print("└── data.yaml")
    print("Sử dụng dataset với lệnh:")
    print(f"yolo detect train data={os.path.join(args.output_dir, 'data.yaml')} model=yolo11n.pt")

def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị dataset YOLO từ dữ liệu bóng đá')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Thư mục gốc chứa dữ liệu')
    parser.add_argument('--output-dir', type=str, default='./dataset',
                        help='Thư mục đầu ra cho dataset YOLO')
    parser.add_argument('--splits', type=str, nargs='+', default=['train'],
                        help='Loại split cần xử lý (train/test)')
    parser.add_argument('--frame-interval', type=int, default=10,
                        help='Chỉ lưu 1 frame sau mỗi N frames')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Tỷ lệ chia tập validation')
    parser.add_argument('--resize-width', type=int, default=None,
                        help='Chiều rộng để resize frame (mặc định: giữ nguyên kích thước)')
    parser.add_argument('--resize-height', type=int, default=None,
                        help='Chiều cao để resize frame (mặc định: giữ nguyên kích thước)')
    
    args = parser.parse_args()
    prepare_dataset(args)

if __name__ == "__main__":
    main()