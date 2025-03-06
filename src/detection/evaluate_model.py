#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để đánh giá model YOLO đã huấn luyện.
"""

from ultralytics import YOLO
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Đánh giá model YOLO')
    parser.add_argument('--model', type=str, default='../../models/detection/balanced_train/weights/best.pt',
                        help='Đường dẫn đến model weights')
    parser.add_argument('--data', type=str, default='../../dataset/data.yaml',
                        help='Đường dẫn đến file cấu hình dataset')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Kích thước ảnh để đánh giá')
    parser.add_argument('--device', type=str, default='mps',
                        help='Thiết bị để đánh giá (mps, cpu, 0, ...)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Kiểm tra xem file model tồn tại không
    if not os.path.exists(args.model):
        print(f"Không tìm thấy file model tại: {args.model}")
        return
    
    print(f"Đang đánh giá model: {args.model}")
    model = YOLO(args.model)
    
    # Đánh giá model trên tập validation
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        half=True,       # Sử dụng half precision
        verbose=True     # Hiển thị chi tiết
    )
    
    # Hiển thị kết quả
    print("\nKết quả đánh giá:")
    print(f"mAP50: {results.box.map50:.3f}")
    print(f"mAP50-95: {results.box.map:.3f}")
    
    # Hiển thị metrics cho từng lớp
    print("\nKết quả theo lớp:")
    for i, class_name in enumerate(model.names):
        try:
            class_map = results.box.maps[i]
            print(f"{class_name}: mAP50 = {class_map:.3f}")
        except:
            print(f"{class_name}: Không có thông tin")

if __name__ == "__main__":
    main()