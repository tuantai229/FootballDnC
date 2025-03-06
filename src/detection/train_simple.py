#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script huấn luyện đơn giản cho mô hình YOLO11 
dành cho việc phát hiện cầu thủ và bóng đá.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    """Phân tích tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Huấn luyện YOLO11 cho phát hiện cầu thủ và bóng')
    parser.add_argument('--data', type=str, default='./dataset/data.yaml',
                       help='đường dẫn tới file yaml định nghĩa dataset')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='mô hình khởi tạo, sử dụng yolo11n.pt, yolo11s.pt, etc.')
    parser.add_argument('--epochs', type=int, default=20,
                       help='số epochs để huấn luyện')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='kích thước ảnh đầu vào')
    parser.add_argument('--batch', type=int, default=16,
                       help='batch size hoặc -1 để tự động')
    parser.add_argument('--device', type=str, default='',
                       help='cuda device, ví dụ 0 hoặc 0,1,2,3 hoặc cpu hoặc mps')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='thư mục lưu kết quả')
    parser.add_argument('--name', type=str, default='football_experiment',
                       help='tên thí nghiệm')
    parser.add_argument('--save-period', type=int, default=10,
                       help='lưu checkpoint sau mỗi x epochs')
    parser.add_argument('--patience', type=int, default=10,
                       help='số epochs đợi trước khi early stop')
    
    return parser.parse_args()

def train_model():
    """Huấn luyện mô hình YOLO cho phát hiện cầu thủ và bóng"""
    args = parse_args()
    
    print(f"Đang huấn luyện với dữ liệu: {args.data}")
    print(f"Mô hình khởi tạo: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device if args.device else 'auto'}")
    
    # Kiểm tra file dữ liệu tồn tại
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình dataset: {args.data}")
    
    # Tải mô hình pretrained
    try:
        model = YOLO(args.model)
        print(f"Đã tải mô hình {args.model} thành công")
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        print("Đang tải mô hình yolo11n.yaml để huấn luyện từ đầu...")
        model = YOLO("yolo11n.yaml")
    
    # Cấu hình huấn luyện
    train_args = {
        'data': str(data_path.absolute()),  # Sử dụng đường dẫn tuyệt đối
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'save_period': args.save_period,
        'exist_ok': True  # cho phép ghi đè thư mục
    }
    
    # Thêm device nếu được chỉ định
    if args.device:
        train_args['device'] = args.device
    
    # In ra thông tin cấu hình huấn luyện
    print("Cấu hình huấn luyện:")
    for k, v in train_args.items():
        print(f"  {k}: {v}")
    
    # Bắt đầu huấn luyện
    try:
        results = model.train(**train_args)
        print("Huấn luyện hoàn tất!")
        print(f"Mô hình được lưu tại: {results.save_dir}")
        
        # Đường dẫn tới các checkpoints
        best_model_path = Path(results.save_dir) / "weights" / "best.pt"
        last_model_path = Path(results.save_dir) / "weights" / "last.pt"
        
        print(f"Mô hình tốt nhất: {best_model_path}")
        print(f"Checkpoint cuối cùng: {last_model_path}")
        
        return best_model_path
    
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện: {e}")
        return None

def main():
    """Hàm chính"""
    # Đảm bảo thư mục src/detection tồn tại
    os.makedirs("src/detection", exist_ok=True)
    
    # Huấn luyện mô hình
    best_model_path = train_model()
    
    if best_model_path and best_model_path.exists():
        print("\nĐể đánh giá mô hình trên tập validation, sử dụng lệnh:")
        print(f"yolo detect val model={best_model_path} data={Path(parse_args().data).absolute()}")
        
        print("\nĐể dự đoán trên tập test hoặc video mới:")
        print(f"yolo detect predict model={best_model_path} source=path/to/test/video.mp4")

if __name__ == "__main__":
    main()