#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để trích xuất frames từ video và chuyển đổi annotation COCO sang định dạng YOLO theo cấu trúc chuẩn.
"""

import os
import cv2
import json
import random
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, annotation_path, frame_interval=10, val_split=0.2):
    """
    Trích xuất frames từ video và chuyển đổi annotation sang định dạng YOLO
    
    Args:
        video_path (str): Đường dẫn tới file video
        output_dir (str): Thư mục gốc để lưu dataset (dataset/)
        annotation_path (str): Đường dẫn tới file annotation json
        frame_interval (int): Khoảng cách giữa các frame được trích xuất
        val_split (float): Tỷ lệ dữ liệu dành cho validation
    """
    # Đảm bảo thư mục output tồn tại
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_lbl_dir = os.path.join(output_dir, "train", "labels")
    val_img_dir = os.path.join(output_dir, "val", "images")
    val_lbl_dir = os.path.join(output_dir, "val", "labels")
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # Tạo đối tượng VideoCapture
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return
    
    # Đọc thông tin video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Đọc file annotation
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    # Tạo map từ image_id sang filename và ngược lại
    image_id_to_filename = {}
    filename_to_image_id = {}
    for img in annotations['images']:
        image_id_to_filename[img['id']] = img['file_name']
        filename_to_image_id[img['file_name']] = img['id']
    
    # Tạo map từ image_id sang danh sách annotations
    image_annotations = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Danh sách lưu các frames đã trích xuất
    extracted_frames = []
    
    # Trích xuất frames
    print(f"Đang trích xuất frames từ {video_path}...")
    frame_idx = 0
    with tqdm(total=frame_count) as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Chỉ xử lý frame nếu phù hợp với interval
            if frame_idx % frame_interval == 0:
                # Tạo tên file dựa trên định dạng trong file annotation
                frame_filename = f"frame_{frame_idx:06d}.PNG"
                
                # Kiểm tra xem frame này có trong annotation không
                if frame_filename in filename_to_image_id:
                    image_id = filename_to_image_id[frame_filename]
                    
                    # Lưu thông tin frame
                    extracted_frames.append({
                        'frame_idx': frame_idx,
                        'filename': frame_filename.replace('.PNG', '.jpg'),
                        'image_id': image_id
                    })
                    
                    # Lưu frame dưới dạng jpg (tiêu chuẩn hơn cho ML)
                    output_filename = frame_filename.replace('.PNG', '.jpg')
                    
                    # Viết file ảnh (tạm thời lưu vào thư mục train)
                    cv2.imwrite(os.path.join(train_img_dir, output_filename), frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Giải phóng video
    video.release()
    
    # Phân chia ngẫu nhiên thành tập train và val
    random.shuffle(extracted_frames)
    val_count = int(len(extracted_frames) * val_split)
    
    val_frames = extracted_frames[:val_count]
    train_frames = extracted_frames[val_count:]
    
    print(f"Đã trích xuất {len(extracted_frames)} frames")
    print(f"Tập train: {len(train_frames)} frames")
    print(f"Tập val: {len(val_frames)} frames")
    
    # Di chuyển các file ảnh từ train sang val theo phân chia
    for frame_info in val_frames:
        src_img = os.path.join(train_img_dir, frame_info['filename'])
        dst_img = os.path.join(val_img_dir, frame_info['filename'])
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
    
    # Tạo map các category_id
    category_map = {}
    for cat in annotations['categories']:
        category_map[cat['id']] = cat['name']
    
    # Tạo file classes.txt
    with open(os.path.join(output_dir, "classes.txt"), 'w') as f:
        classes = []
        # Chỉ lấy category ball và player
        for cat_id, cat_name in category_map.items():
            if cat_name in ['ball', 'player']:
                classes.append(cat_name)
        f.write('\n'.join(classes))
    
    # Dictionary ánh xạ tên category sang index cho YOLO
    # Chỉ quan tâm đến ball và player (0: ball, 1: player)
    yolo_class_map = {'ball': 0, 'player': 1}
    
    # Tạo file annotation YOLO cho tập train
    for frame_info in train_frames:
        image_id = frame_info['image_id']
        if image_id in image_annotations:
            annotations_list = image_annotations[image_id]
            yolo_annotations = []
            
            for ann in annotations_list:
                category_id = ann['category_id']
                category_name = category_map.get(category_id)
                
                # Chỉ xử lý ball và player
                if category_name in ['ball', 'player']:
                    # Lấy thông tin bounding box
                    x, y, w, h = ann['bbox']
                    
                    # Chuẩn hóa giá trị cho YOLO (x_center, y_center, width, height) từ 0-1
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_normalized = w / width
                    h_normalized = h / height
                    
                    # Chuyển sang định dạng YOLO (class x_center y_center width height)
                    yolo_class = yolo_class_map[category_name]
                    yolo_line = f"{yolo_class} {x_center} {y_center} {w_normalized} {h_normalized}"
                    yolo_annotations.append(yolo_line)
            
            # Ghi ra file txt
            if yolo_annotations:
                txt_filename = frame_info['filename'].replace('.jpg', '.txt')
                with open(os.path.join(train_lbl_dir, txt_filename), 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    # Tạo file annotation YOLO cho tập val
    for frame_info in val_frames:
        image_id = frame_info['image_id']
        if image_id in image_annotations:
            annotations_list = image_annotations[image_id]
            yolo_annotations = []
            
            for ann in annotations_list:
                category_id = ann['category_id']
                category_name = category_map.get(category_id)
                
                # Chỉ xử lý ball và player
                if category_name in ['ball', 'player']:
                    # Lấy thông tin bounding box
                    x, y, w, h = ann['bbox']
                    
                    # Chuẩn hóa giá trị cho YOLO (x_center, y_center, width, height) từ 0-1
                    x_center = (x + w / 2) / width
                    y_center = (y + h / 2) / height
                    w_normalized = w / width
                    h_normalized = h / height
                    
                    # Chuyển sang định dạng YOLO (class x_center y_center width height)
                    yolo_class = yolo_class_map[category_name]
                    yolo_line = f"{yolo_class} {x_center} {y_center} {w_normalized} {h_normalized}"
                    yolo_annotations.append(yolo_line)
            
            # Ghi ra file txt
            if yolo_annotations:
                txt_filename = frame_info['filename'].replace('.jpg', '.txt')
                with open(os.path.join(val_lbl_dir, txt_filename), 'w') as f:
                    f.write('\n'.join(yolo_annotations))
    
    # Tạo file data.yaml
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 2,
        'names': ['ball', 'player']
    }
    
    # Ghi file data.yaml
    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        f.write("# YOLOv8 dataset config\n")
        f.write(f"path: {data_yaml['path']}\n")
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for i, name in enumerate(data_yaml['names']):
            f.write(f"  {i}: {name}\n")
    
    return os.path.join(output_dir, "data.yaml")

def process_match_directory(match_dir, output_dir, frame_interval=10, val_split=0.2):
    """
    Xử lý một thư mục match, trích xuất frames và chuyển đổi annotation
    
    Args:
        match_dir (str): Đường dẫn tới thư mục match
        output_dir (str): Thư mục đầu ra cho dataset
        frame_interval (int): Khoảng cách giữa các frame được trích xuất
        val_split (float): Tỷ lệ dữ liệu dành cho validation
    """
    # Tìm file video và annotation trong thư mục match
    video_path = None
    annotation_path = None
    
    for file in os.listdir(match_dir):
        if file.endswith('.mp4'):
            video_path = os.path.join(match_dir, file)
        elif file.endswith('.json'):
            annotation_path = os.path.join(match_dir, file)
    
    if video_path and annotation_path:
        print(f"Xử lý match: {os.path.basename(match_dir)}")
        print(f"Video: {video_path}")
        print(f"Annotation: {annotation_path}")
        
        # Trích xuất frames và chuyển đổi annotation
        extract_frames(video_path, output_dir, annotation_path, frame_interval, val_split)
    else:
        print(f"Không tìm thấy cả file video và annotation trong thư mục {match_dir}")

def process_all_matches(data_dir, output_dir, frame_interval=10, val_split=0.2, splits=None):
    """
    Xử lý tất cả các thư mục match trong dataset
    
    Args:
        data_dir (str): Thư mục gốc chứa dữ liệu
        output_dir (str): Thư mục đầu ra cho dataset chuẩn YOLO
        frame_interval (int): Khoảng cách giữa các frame được trích xuất
        val_split (float): Tỷ lệ dữ liệu dành cho validation
        splits (list): Danh sách các split cần xử lý (train, test, ...)
    """
    if splits is None:
        splits = ['train', 'test']
    
    for split in splits:
        split_dir = os.path.join(data_dir, f"football_{split}")
        if not os.path.exists(split_dir):
            print(f"Không tìm thấy thư mục {split_dir}")
            continue
        
        print(f"Đang xử lý split: {split}")
        
        # Lấy danh sách các thư mục match
        match_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        
        for match_dir in match_dirs:
            match_path = os.path.join(split_dir, match_dir)
            # Chỉ xử lý thư mục match có cả video và annotation
            has_video = any(f.endswith('.mp4') for f in os.listdir(match_path))
            has_json = any(f.endswith('.json') for f in os.listdir(match_path))
            
            if has_video and has_json:
                process_match_directory(match_path, output_dir, frame_interval, val_split)

def main():
    parser = argparse.ArgumentParser(description='Chuẩn bị dữ liệu cho huấn luyện YOLO')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Thư mục gốc chứa dữ liệu')
    parser.add_argument('--output-dir', type=str, default='./dataset',
                        help='Thư mục đầu ra cho dataset chuẩn YOLO')
    parser.add_argument('--frame-interval', type=int, default=10,
                        help='Khoảng cách giữa các frame được trích xuất')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Tỷ lệ dữ liệu dành cho validation')
    parser.add_argument('--splits', type=str, nargs='+', default=['train'],
                        help='Danh sách các split cần xử lý (train, test)')
    parser.add_argument('--single-match', type=str, default=None,
                        help='Chỉ xử lý một thư mục match cụ thể')
    
    args = parser.parse_args()
    
    # Tạo thư mục đầu ra
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.single_match:
        # Xử lý một thư mục match cụ thể
        match_path = os.path.join(args.data_dir, args.single_match)
        if os.path.exists(match_path) and os.path.isdir(match_path):
            process_match_directory(match_path, args.output_dir, args.frame_interval, args.val_split)
        else:
            print(f"Không tìm thấy thư mục match: {match_path}")
    else:
        # Xử lý tất cả các thư mục match
        process_all_matches(args.data_dir, args.output_dir, args.frame_interval, args.val_split, args.splits)
    
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
    
    # Check data.yaml path
    data_yaml_path = os.path.join(args.output_dir, "data.yaml")
    if os.path.exists(data_yaml_path):
        print(f"\nSử dụng dataset với lệnh:")
        print(f"yolo detect train data={data_yaml_path} model=yolo11n.pt")

if __name__ == "__main__":
    main()