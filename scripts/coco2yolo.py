#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script chuyển đổi annotation từ định dạng COCO sang định dạng YOLO.
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import shutil
import random
import glob

def convert_bbox_coco_to_yolo(bbox, img_width, img_height):
    """
    Chuyển đổi bounding box từ định dạng COCO [x_min, y_min, width, height]
    sang định dạng YOLO [x_center, y_center, width, height] và chuẩn hóa về [0-1]
    
    Args:
        bbox (list): Bounding box theo định dạng COCO [x_min, y_min, width, height]
        img_width (int): Chiều rộng của ảnh
        img_height (int): Chiều cao của ảnh
        
    Returns:
        list: Bounding box theo định dạng YOLO [x_center, y_center, width, height]
    """
    x_min, y_min, width, height = bbox
    
    # Tính tọa độ trung tâm
    x_center = x_min + width / 2
    y_center = y_min + height / 2
    
    # Chuẩn hóa về [0-1]
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Kiểm tra giá trị hợp lệ
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return [x_center, y_center, width, height]

def convert_coco_to_yolo(json_file, output_dir, classes=None, split_val=0.2, frame_dir=None):
    """
    Chuyển đổi annotations từ định dạng COCO sang định dạng YOLO
    
    Args:
        json_file (str): Đường dẫn đến file JSON định dạng COCO
        output_dir (str): Thư mục đầu ra cho dữ liệu YOLO
        classes (list, optional): Danh sách các lớp cần giữ lại. Mặc định = None (giữ tất cả)
        split_val (float): Tỷ lệ dữ liệu dùng cho validation (0-1)
        frame_dir (str, optional): Thư mục chứa các frames đã trích xuất. Nếu có, sẽ copy ảnh sang thư mục YOLO
    
    Returns:
        dict: Thống kê quá trình chuyển đổi
    """
    # Đọc file JSON COCO
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo các thư mục theo cấu trúc YOLO
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    
    train_img_dir = os.path.join(images_dir, 'train')
    val_img_dir = os.path.join(images_dir, 'val')
    train_label_dir = os.path.join(labels_dir, 'train')
    val_label_dir = os.path.join(labels_dir, 'val')
    
    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Phân tích categories
    category_map = {}
    for category in coco_data['categories']:
        category_map[category['id']] = category['name']
    
    # Lọc các categories nếu cần
    if classes is None:
        # Mặc định, giữ lại player và ball
        classes = ['player', 'ball']
    
    # Tạo mapping từ category_id sang class_id của YOLO
    yolo_category_map = {}
    class_names = []
    
    for cat_id, cat_name in category_map.items():
        if cat_name in classes:
            yolo_category_map[cat_id] = len(class_names)
            class_names.append(cat_name)
    
    # Tạo file classes.txt
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    # Tạo mapping từ file_name sang thông tin ảnh và annotations
    image_info_map = {}
    for image in coco_data['images']:
        file_name = Path(image['file_name']).stem  # Lấy tên file không bao gồm phần mở rộng
        image_info_map[file_name] = {
            'id': image['id'],
            'width': image['width'],
            'height': image['height'],
            'annotations': []
        }
    
    # Nhóm annotations theo image_id
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        
        # Bỏ qua các category không thuộc danh sách classes
        if category_id not in yolo_category_map:
            continue
        
        # Tìm file_name tương ứng với image_id
        for file_name, info in image_info_map.items():
            if info['id'] == image_id:
                info['annotations'].append(annotation)
                break
    
    # Kiểm tra frames thực sự tồn tại nếu có frame_dir
    existing_frames = []
    if frame_dir and os.path.exists(frame_dir):
        # Tìm tất cả các file ảnh trong frame_dir
        image_patterns = ['*.jpg', '*.png', '*.PNG']
        for pattern in image_patterns:
            files = glob.glob(os.path.join(frame_dir, pattern))
            for file_path in files:
                file_name = Path(file_path).stem
                existing_frames.append(file_name)
        
        print(f"Tìm thấy {len(existing_frames)} frames trong thư mục {frame_dir}")
    
    # Lọc lại image_info_map chỉ giữ những frame thực sự tồn tại
    filtered_image_info = {}
    for file_name, info in image_info_map.items():
        if not frame_dir or file_name in existing_frames:
            filtered_image_info[file_name] = info
    
    # Chia ngẫu nhiên thành tập train và val
    all_file_names = list(filtered_image_info.keys())
    random.shuffle(all_file_names)
    val_size = int(len(all_file_names) * split_val)
    val_file_names = set(all_file_names[:val_size])
    train_file_names = set(all_file_names[val_size:])
    
    # Thống kê
    stats = {
        'total_images': len(filtered_image_info),
        'train_images': len(train_file_names),
        'val_images': len(val_file_names),
        'class_counts': {class_name: 0 for class_name in class_names}
    }
    
    # Tạo danh sách để lưu đường dẫn hợp lệ cho train.txt và val.txt
    train_paths = []
    val_paths = []
    
    # Xử lý từng ảnh và các annotations của nó
    for file_name in tqdm(all_file_names, desc="Chuyển đổi annotations"):
        info = filtered_image_info[file_name]
        annotations = info['annotations']
        
        # Xác định ảnh thuộc tập train hay val
        is_train = file_name in train_file_names
        
        # Tạo đường dẫn đầu ra cho file label YOLO
        if is_train:
            label_path = os.path.join(train_label_dir, f"{file_name}.txt")
            img_path = os.path.join(train_img_dir, f"{file_name}.jpg")
        else:
            label_path = os.path.join(val_label_dir, f"{file_name}.txt")
            img_path = os.path.join(val_img_dir, f"{file_name}.jpg")
        
        # Nếu có thư mục chứa frames, copy ảnh sang thư mục YOLO
        if frame_dir:
            # Tìm kiếm ảnh .jpg, .png hoặc .PNG
            source_img_paths = [
                os.path.join(frame_dir, f"{file_name}.jpg"),
                os.path.join(frame_dir, f"{file_name}.png"),
                os.path.join(frame_dir, f"{file_name}.PNG")
            ]
            
            for source_img_path in source_img_paths:
                if os.path.exists(source_img_path):
                    # Đảm bảo đường dẫn đến thư mục chứa file đích đã tồn tại
                    os.makedirs(os.path.dirname(img_path), exist_ok=True)
                    # Copy ảnh sang thư mục YOLO
                    shutil.copy(source_img_path, img_path)
                    
                    # Thêm đường dẫn vào danh sách tương ứng
                    if is_train:
                        train_paths.append(img_path)
                    else:
                        val_paths.append(img_path)
                    break
        
        # Ghi annotations ra file YOLO
        with open(label_path, 'w') as f_label:
            for annotation in annotations:
                bbox_coco = annotation['bbox']
                category_id = annotation['category_id']
                
                # Lấy class_id YOLO
                class_id = yolo_category_map[category_id]
                class_name = class_names[class_id]
                stats['class_counts'][class_name] += 1
                
                # Chuyển đổi bounding box sang định dạng YOLO
                bbox_yolo = convert_bbox_coco_to_yolo(
                    bbox_coco, 
                    info['width'], 
                    info['height']
                )
                
                # Ghi ra file theo định dạng YOLO
                f_label.write(f"{class_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")
    
    # Ghi danh sách đường dẫn vào file train.txt và val.txt
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f_train:
        for path in train_paths:
            f_train.write(f"{path}\n")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f_val:
        for path in val_paths:
            f_val.write(f"{path}\n")
    
    # Tạo file data.yaml
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(f"train: {os.path.join(output_dir, 'train.txt')}\n")
        f.write(f"val: {os.path.join(output_dir, 'val.txt')}\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")
    
    return stats

def convert_match_dir(match_dir, output_dir, classes=None, split_val=0.2, frames_dir=None):
    """
    Chuyển đổi dữ liệu của một trận đấu từ định dạng COCO sang YOLO
    
    Args:
        match_dir (str): Đường dẫn đến thư mục chứa dữ liệu của trận đấu
        output_dir (str): Thư mục đầu ra cho dữ liệu YOLO
        classes (list, optional): Danh sách các lớp cần giữ lại
        split_val (float): Tỷ lệ dữ liệu dùng cho validation
        frames_dir (str, optional): Thư mục gốc chứa frames đã trích xuất
        
    Returns:
        dict: Thống kê quá trình chuyển đổi hoặc None nếu không tìm thấy file JSON
    """
    # Lấy tên của trận đấu
    match_name = os.path.basename(match_dir)
    
    # Tìm file JSON trong thư mục của trận đấu
    json_files = [f for f in os.listdir(match_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"Không tìm thấy file JSON trong {match_dir}")
        return None
    
    # Lấy file JSON đầu tiên
    json_file = os.path.join(match_dir, json_files[0])
    
    # Xác định thư mục chứa frames (nếu có)
    frame_dir = None
    if frames_dir:
        # Kiểm tra các vị trí có thể có frames
        possible_dirs = [
            os.path.join(frames_dir, "train", match_name),
            os.path.join(frames_dir, "test", match_name),
            os.path.join(frames_dir, match_name)
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                frame_dir = dir_path
                break
    
    # Thực hiện chuyển đổi
    print(f"Chuyển đổi dữ liệu trận đấu {match_name}...")
    
    if frame_dir:
        print(f"Đã tìm thấy thư mục frames: {frame_dir}")
    else:
        print(f"Không tìm thấy thư mục frames cho {match_name}, chỉ tạo files label")
    
    stats = convert_coco_to_yolo(json_file, output_dir, classes, split_val, frame_dir)
    
    return stats

def convert_all_matches(data_dir, output_dir, classes=None, split_val=0.2, split_types=None, frames_dir=None):
    """
    Chuyển đổi dữ liệu của tất cả các trận đấu từ định dạng COCO sang YOLO
    
    Args:
        data_dir (str): Thư mục gốc chứa dữ liệu
        output_dir (str): Thư mục đầu ra cho dữ liệu YOLO
        classes (list, optional): Danh sách các lớp cần giữ lại
        split_val (float): Tỷ lệ dữ liệu dùng cho validation
        split_types (list, optional): Danh sách các loại split cần xử lý
        frames_dir (str, optional): Thư mục gốc chứa frames đã trích xuất
        
    Returns:
        dict: Thống kê tổng hợp quá trình chuyển đổi
    """
    # Mặc định xử lý cả train và test nếu không chỉ định
    if split_types is None:
        split_types = ['train', 'test']
    
    # Thống kê tổng hợp
    overall_stats = {
        'total_matches': 0,
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'class_counts': {}
    }
    
    # Xử lý từng loại split (train/test)
    for split in split_types:
        split_dir = os.path.join(data_dir, f"football_{split}")
        
        if not os.path.exists(split_dir):
            print(f"Thư mục {split_dir} không tồn tại, bỏ qua.")
            continue
        
        # Lấy danh sách các thư mục match trong split_dir
        match_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        
        print(f"\nĐang xử lý {len(match_dirs)} trận đấu trong tập {split}...")
        
        # Xử lý từng thư mục match
        for match_dir in match_dirs:
            match_path = os.path.join(split_dir, match_dir)
            
            # Tạo thư mục đầu ra cho match này
            match_output_dir = os.path.join(output_dir, split, match_dir)
            os.makedirs(match_output_dir, exist_ok=True)
            
            # Chuyển đổi dữ liệu của trận đấu
            stats = convert_match_dir(match_path, match_output_dir, classes, split_val, frames_dir)
            
            if stats:
                overall_stats['total_matches'] += 1
                overall_stats['total_images'] += stats['total_images']
                overall_stats['train_images'] += stats['train_images']
                overall_stats['val_images'] += stats['val_images']
                
                # Cập nhật thống kê số lượng từng class
                for class_name, count in stats['class_counts'].items():
                    if class_name not in overall_stats['class_counts']:
                        overall_stats['class_counts'][class_name] = 0
                    overall_stats['class_counts'][class_name] += count
    
    # In thống kê tổng hợp
    print("\n=== Thống kê chuyển đổi ===")
    print(f"Tổng số trận đấu: {overall_stats['total_matches']}")
    print(f"Tổng số ảnh: {overall_stats['total_images']}")
    print(f"Số ảnh train: {overall_stats['train_images']}")
    print(f"Số ảnh val: {overall_stats['val_images']}")
    print("Số lượng đối tượng theo class:")
    for class_name, count in overall_stats['class_counts'].items():
        print(f"  - {class_name}: {count}")
    
    return overall_stats

def main():
    parser = argparse.ArgumentParser(description='Chuyển đổi annotations từ định dạng COCO sang YOLO')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Thư mục gốc chứa dữ liệu')
    parser.add_argument('--output-dir', type=str, default='./data/yolo',
                        help='Thư mục đầu ra cho dữ liệu YOLO')
    parser.add_argument('--frames-dir', type=str, default='./data/frames',
                        help='Thư mục gốc chứa frames đã trích xuất')
    parser.add_argument('--classes', type=str, nargs='+', default=['player', 'ball'],
                        help='Danh sách các lớp cần giữ lại')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'test'],
                        help='Loại split cần xử lý (train/test)')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Tỷ lệ dữ liệu dùng cho validation (0-1)')
    parser.add_argument('--no-frames', action='store_true',
                        help='Không sử dụng frames đã trích xuất, chỉ tạo files label')
    parser.add_argument('--single-match', type=str, default=None,
                        help='Chỉ xử lý một trận đấu cụ thể (đường dẫn đầy đủ)')
    
    args = parser.parse_args()
    
    # Xác định thư mục frames
    frames_dir = None if args.no_frames else args.frames_dir
    
    # Nếu chỉ định single_match, chỉ xử lý dữ liệu của trận đó
    if args.single_match:
        if not os.path.isdir(args.single_match):
            print(f"Lỗi: Thư mục {args.single_match} không tồn tại")
            return
        
        match_name = os.path.basename(args.single_match)
        match_output_dir = os.path.join(args.output_dir, match_name)
        os.makedirs(match_output_dir, exist_ok=True)
        
        stats = convert_match_dir(
            args.single_match,
            match_output_dir,
            args.classes,
            args.val_split,
            frames_dir
        )
        
        if stats:
            print("\n=== Thống kê chuyển đổi ===")
            print(f"Tổng số ảnh: {stats['total_images']}")
            print(f"Số ảnh train: {stats['train_images']}")
            print(f"Số ảnh val: {stats['val_images']}")
            print("Số lượng đối tượng theo class:")
            for class_name, count in stats['class_counts'].items():
                print(f"  - {class_name}: {count}")
    else:
        # Xử lý tất cả các trận đấu trong các thư mục đã chỉ định
        convert_all_matches(
            args.data_dir,
            args.output_dir,
            args.classes,
            args.val_split,
            args.splits,
            frames_dir
        )

if __name__ == "__main__":
    main()