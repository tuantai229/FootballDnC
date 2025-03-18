#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để phân tích thông tin từ các file JSON annotation
Hiển thị thống kê về số áo, màu áo, tình trạng số áo, occlusion, v.v.
"""

import os
import json
import argparse
from collections import Counter, defaultdict

def analyze_json_data(data_dir):
    """
    Phân tích dữ liệu từ các file JSON annotation
    
    Args:
        data_dir (str): Thư mục gốc chứa dữ liệu
    """
    # Thống kê chung
    total_images = 0
    total_annotations = 0
    
    # Thống kê theo loại đối tượng
    category_counts = Counter()
    
    # Thống kê cho cầu thủ
    jersey_numbers = Counter()
    team_colors = Counter()
    number_visible_status = Counter()
    occluded_status = Counter()
    facing_side_status = Counter()
    
    # Thống kê chi tiết cho cầu thủ
    player_attributes = defaultdict(lambda: defaultdict(int))
    
    # Duyệt qua các thư mục split (train/test)
    for split in ["train", "test"]:
        split_dir = os.path.join(data_dir, f"football_{split}")
        
        # Kiểm tra thư mục có tồn tại không
        if not os.path.exists(split_dir):
            print(f"Thư mục {split_dir} không tồn tại, bỏ qua...")
            continue
            
        print(f"\n=== PHÂN TÍCH DỮ LIỆU CHO SPLIT: {split} ===")
        
        # Lấy danh sách các thư mục match
        match_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        
        # Xử lý từng match
        for match_dir in match_dirs:
            # Đường dẫn đến file annotation
            anno_path = os.path.join(split_dir, match_dir, f"{match_dir}.json")
            
            # Kiểm tra file annotation có tồn tại không
            if not os.path.exists(anno_path):
                print(f"File annotation {anno_path} không tồn tại, bỏ qua...")
                continue
            
            # Đọc file annotation
            with open(anno_path, 'r') as f:
                annotations = json.load(f)
            
            # Cập nhật thống kê số lượng ảnh
            total_images += len(annotations.get('images', []))
            
            # Duyệt qua các annotation
            for ann in annotations.get('annotations', []):
                total_annotations += 1
                
                # Lấy category_id
                category_id = ann.get('category_id')
                category_counts[category_id] += 1
                
                # Phân tích chi tiết cho cầu thủ (category_id = 4)
                if category_id == 4:  # Cầu thủ
                    attributes = ann.get('attributes', {})
                    
                    # Thu thập thông tin số áo
                    jersey_number = attributes.get('jersey_number', 'unknown')
                    jersey_numbers[jersey_number] += 1
                    
                    # Thu thập thông tin màu áo
                    team_color = attributes.get('team_jersey_color', 'unknown')
                    team_colors[team_color] += 1
                    
                    # Thu thập thông tin hiển thị số áo
                    number_visible = attributes.get('number_visible', 'unknown')
                    number_visible_status[number_visible] += 1
                    
                    # Thu thập thông tin occlusion
                    occluded = attributes.get('occluded', 'unknown')
                    occluded_status[occluded] += 1
                    
                    # Thu thập thông tin facing_side
                    facing_side = attributes.get('facing_side', 'unknown')
                    facing_side_status[facing_side] += 1
                    
                    # Phân tích chi tiết kết hợp giữa các thuộc tính
                    player_attributes['number_visible_occluded'][(number_visible, occluded)] += 1
                    player_attributes['number_visible_jersey'][(number_visible, jersey_number)] += 1
                    player_attributes['team_color_number_visible'][(team_color, number_visible)] += 1
    
    # Hiển thị thống kê tổng quan
    print("\n=== THỐNG KÊ TỔNG QUAN ===")
    print(f"Tổng số frame: {total_images}")
    print(f"Tổng số annotation: {total_annotations}")
    
    # Hiển thị thống kê theo loại đối tượng
    print("\n=== THỐNG KÊ THEO LOẠI ĐỐI TƯỢNG ===")
    category_names = {
        1: "field (sân)",
        2: "bystander (người ngoài)",
        3: "ball (bóng)",
        4: "player (cầu thủ)"
    }
    for category_id, count in sorted(category_counts.items()):
        category_name = category_names.get(category_id, f"unknown_{category_id}")
        print(f"{category_name}: {count} annotations")
    
    # Hiển thị thống kê về số áo
    print("\n=== THỐNG KÊ SỐ ÁO ===")
    for number, count in sorted(jersey_numbers.items()):
        print(f"Số áo {number}: {count} cầu thủ")
    
    # Hiển thị thống kê về màu áo
    print("\n=== THỐNG KÊ MÀU ÁO ===")
    for color, count in sorted(team_colors.items()):
        print(f"Màu {color}: {count} cầu thủ")
    
    # Hiển thị thống kê về tình trạng hiển thị số áo
    print("\n=== THỐNG KÊ TÌNH TRẠNG HIỂN THỊ SỐ ÁO ===")
    for status, count in sorted(number_visible_status.items()):
        print(f"{status}: {count} cầu thủ")
    
    # Hiển thị thống kê về tình trạng bị che khuất
    print("\n=== THỐNG KÊ TÌNH TRẠNG CHE KHUẤT ===")
    for status, count in sorted(occluded_status.items()):
        print(f"{status}: {count} cầu thủ")
    
    # Hiển thị thống kê về hướng nhìn
    print("\n=== THỐNG KÊ HƯỚNG NHÌN ===")
    for status, count in sorted(facing_side_status.items()):
        print(f"{status}: {count} cầu thủ")
    
    # Hiển thị thống kê kết hợp giữa số áo hiển thị và bị che khuất
    print("\n=== THỐNG KÊ KẾT HỢP GIỮA HIỂN THỊ SỐ ÁO VÀ CHE KHUẤT ===")
    for (number_visible, occluded), count in sorted(player_attributes['number_visible_occluded'].items()):
        print(f"number_visible={number_visible}, occluded={occluded}: {count} cầu thủ")
    
    # Hiển thị thống kê kết hợp giữa tình trạng hiển thị số áo và số áo
    print("\n=== THỐNG KÊ KẾT HỢP GIỮA HIỂN THỊ SỐ ÁO VÀ SỐ ÁO ===")
    invisible_numbers = defaultdict(int)
    visible_numbers = defaultdict(int)
    
    for (number_visible, jersey_number), count in sorted(player_attributes['number_visible_jersey'].items()):
        if number_visible == 'invisible':
            if jersey_number.isdigit():
                invisible_numbers[jersey_number] += count
        elif number_visible == 'visible':
            if jersey_number.isdigit():
                visible_numbers[jersey_number] += count
                
    print("\n- Số áo hiển thị 'visible':")
    for number, count in sorted(visible_numbers.items(), key=lambda x: int(x[0])):
        print(f"  Số áo {number}: {count} cầu thủ")
        
    print("\n- Số áo không hiển thị 'invisible':")
    for number, count in sorted(invisible_numbers.items(), key=lambda x: int(x[0])):
        print(f"  Số áo {number}: {count} cầu thủ")
    
    # Hiển thị thống kê kết hợp giữa màu áo và tình trạng hiển thị số áo
    print("\n=== THỐNG KÊ KẾT HỢP GIỮA MÀU ÁO VÀ HIỂN THỊ SỐ ÁO ===")
    for (team_color, number_visible), count in sorted(player_attributes['team_color_number_visible'].items()):
        print(f"team_color={team_color}, number_visible={number_visible}: {count} cầu thủ")

def main():
    parser = argparse.ArgumentParser(description='Phân tích dữ liệu từ các file JSON annotation')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Thư mục gốc chứa dữ liệu')
    
    args = parser.parse_args()
    analyze_json_data(args.data_dir)

if __name__ == "__main__":
    main()