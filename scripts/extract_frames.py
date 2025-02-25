#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để trích xuất các frame từ video bóng đá và lưu thành các file ảnh.
Sử dụng cho dự án FootballDnC - nhận diện và phân loại số áo cầu thủ.
"""

import os
import json
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames(video_path, output_dir, frame_interval=1, resize_dim=None):
    """
    Trích xuất frame từ video và lưu vào thư mục đầu ra.
    
    Args:
        video_path (str): Đường dẫn đến file video
        output_dir (str): Thư mục đích để lưu các frame
        frame_interval (int): Chỉ lưu 1 frame sau mỗi frame_interval frames (mặc định: 1 - lưu tất cả)
        resize_dim (tuple, optional): Kích thước để resize frame, định dạng (width, height)
    
    Returns:
        int: Số lượng frame đã trích xuất
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Mở video
    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return 0
    
    # Đọc thông tin về video
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Dimensions: {width}x{height}, FPS: {fps}, Duration: {duration:.2f}s, Frames: {frame_count}")
    
    # Đọc và lưu các frame
    frame_idx = 0
    saved_count = 0
    
    # Sử dụng tqdm để hiển thị tiến trình
    with tqdm(total=frame_count, desc="Trích xuất frames") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # Chỉ lưu frame nếu đạt đến interval
            if frame_idx % frame_interval == 0:
                # Resize frame nếu cần
                if resize_dim:
                    frame = cv2.resize(frame, resize_dim)
                
                # Đặt tên file theo format cho phù hợp với file annotation
                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # Lưu frame
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
    
    # Giải phóng tài nguyên
    video.release()
    
    print(f"Đã trích xuất {saved_count} frames từ {frame_idx} frames của video")
    return saved_count

def extract_all_videos(data_dir, output_base_dir, split_types=None, frame_interval=1, resize_dim=None):
    """
    Trích xuất frame từ tất cả video trong thư mục data
    
    Args:
        data_dir (str): Thư mục gốc chứa dữ liệu
        output_base_dir (str): Thư mục gốc để lưu frames đầu ra
        split_types (list, optional): Danh sách các loại split cần xử lý (e.g., ['train', 'test'])
        frame_interval (int): Khoảng cách giữa các frame được lưu
        resize_dim (tuple, optional): Kích thước để resize frame, định dạng (width, height)
    """
    # Mặc định xử lý cả train và test nếu không chỉ định
    if split_types is None:
        split_types = ['train', 'test']
    
    # Tổng số video và frame đã xử lý
    total_videos = 0
    total_frames = 0
    
    # Xử lý từng loại split (train/test)
    for split in split_types:
        split_dir = os.path.join(data_dir, f"football_{split}")
        
        if not os.path.exists(split_dir):
            print(f"Thư mục {split_dir} không tồn tại, bỏ qua.")
            continue
        
        # Lấy danh sách các thư mục match trong split_dir
        match_dirs = [d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]
        
        print(f"\nĐang xử lý {len(match_dirs)} video trong tập {split}...")
        
        # Xử lý từng thư mục match
        for match_dir in match_dirs:
            match_path = os.path.join(split_dir, match_dir)
            
            # Tìm file video trong thư mục match
            video_files = [f for f in os.listdir(match_path) 
                          if f.endswith('.mp4') and os.path.isfile(os.path.join(match_path, f))]
            
            if not video_files:
                print(f"Không tìm thấy file video trong {match_path}, bỏ qua.")
                continue
            
            # Lấy file video đầu tiên (thường chỉ có 1 file)
            video_file = video_files[0]
            video_path = os.path.join(match_path, video_file)
            
            # Tạo thư mục đầu ra cho match này
            match_output_dir = os.path.join(output_base_dir, split, match_dir)
            os.makedirs(match_output_dir, exist_ok=True)
            
            print(f"\nXử lý video: {video_path}")
            
            # Trích xuất frames
            num_frames = extract_frames(
                video_path, 
                match_output_dir, 
                frame_interval=frame_interval,
                resize_dim=resize_dim
            )
            
            total_videos += 1
            total_frames += num_frames
    
    print(f"\nHoàn thành! Đã xử lý {total_videos} videos và trích xuất {total_frames} frames.")

def main():
    parser = argparse.ArgumentParser(description='Trích xuất frames từ video bóng đá')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Thư mục gốc chứa dữ liệu')
    parser.add_argument('--output-dir', type=str, default='./data/frames',
                        help='Thư mục gốc để lưu frames đầu ra')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'test'],
                        help='Loại split cần xử lý (train/test)')
    parser.add_argument('--interval', type=int, default=1,
                        help='Chỉ lưu 1 frame sau mỗi interval frames (mặc định: 1 - lưu tất cả)')
    parser.add_argument('--resize-width', type=int, default=None,
                        help='Chiều rộng để resize (mặc định: giữ nguyên kích thước)')
    parser.add_argument('--resize-height', type=int, default=None,
                        help='Chiều cao để resize (mặc định: giữ nguyên kích thước)')
    parser.add_argument('--single-video', type=str, default=None,
                        help='Chỉ xử lý một video cụ thể (đường dẫn đầy đủ)')
    
    args = parser.parse_args()
    
    # Xử lý resize dimension
    resize_dim = None
    if args.resize_width and args.resize_height:
        resize_dim = (args.resize_width, args.resize_height)
    
    # Nếu chỉ định single_video, chỉ xử lý file đó
    if args.single_video:
        if not os.path.isfile(args.single_video):
            print(f"Lỗi: File {args.single_video} không tồn tại")
            return
        
        video_name = Path(args.single_video).stem
        output_dir = os.path.join(args.output_dir, video_name)
        extract_frames(
            args.single_video, 
            output_dir, 
            frame_interval=args.interval,
            resize_dim=resize_dim
        )
    else:
        # Xử lý tất cả video trong các thư mục đã chỉ định
        extract_all_videos(
            args.data_dir,
            args.output_dir,
            split_types=args.splits,
            frame_interval=args.interval,
            resize_dim=resize_dim
        )

if __name__ == "__main__":
    main()