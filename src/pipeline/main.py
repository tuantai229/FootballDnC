import os
import argparse
import cv2
from tqdm import tqdm

# Import các module đã tạo
from video_utils import read_video, create_video_writer
from detector import Detector
from classifier import Classifier
from visualizer import draw_detections

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Pipeline phát hiện và phân loại số áo cầu thủ')
    parser.add_argument('--detection-model', type=str, required=True,
                       help='Đường dẫn đến model detection')
    parser.add_argument('--classification-model', type=str, required=True,
                       help='Đường dẫn đến model classification')
    parser.add_argument('--input-video', type=str, required=True,
                       help='Đường dẫn đến video input')
    parser.add_argument('--output-video', type=str, default='output.mp4',
                       help='Đường dẫn lưu video output')
    parser.add_argument('--conf-threshold', type=float, default=0.3,
                       help='Ngưỡng confidence cho detection')
    parser.add_argument('--frame-skip', type=int, default=2,
                       help='Xử lý 1 frame sau mỗi N frame')
    args = parser.parse_args()
    
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    
    # Khởi tạo models
    print("Đang khởi tạo models...")
    detector = Detector(args.detection_model, conf_threshold=args.conf_threshold)
    classifier = Classifier(args.classification_model)
    
    # Đọc video
    print(f"Đang đọc video input: {args.input_video}")
    cap, width, height, fps = read_video(args.input_video)
    
    # Tạo video writer
    writer = create_video_writer(args.output_video, width, height, fps)
    
    # Đếm tổng số frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    # Biến lưu frame xử lý cuối cùng
    last_processed_frame = None
    last_jersey_info = {}
    
    # Progress bar
    with tqdm(total=total_frames, desc='Đang xử lý video') as pbar:
        while cap.isOpened():
            # Đọc frame
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_idx += 1
            pbar.update(1)
            
            # Chỉ xử lý 1 frame sau mỗi N frame
            if frame_idx % args.frame_skip == 0:
                # Phát hiện cầu thủ và bóng
                detections = detector.detect(frame)
                
                # Phân loại số áo và màu áo
                jersey_info = {}
                for i, det in enumerate(detections):
                    if det['class_name'] == 'player':
                        # Lấy tọa độ bbox
                        x1, y1, x2, y2 = det['bbox']
                        
                        # Cắt crop từ frame
                        player_crop = frame[y1:y2, x1:x2]
                        
                        # Kiểm tra kích thước hợp lệ
                        if player_crop.shape[0] > 0 and player_crop.shape[1] > 0:
                            try:
                                # Phân loại số áo và màu áo
                                jersey_number, team_color = classifier.classify(player_crop)
                                jersey_info[i] = (jersey_number, team_color)
                            except Exception as e:
                                print(f"Lỗi khi phân loại: {e}")
                
                # Vẽ kết quả
                result_frame = draw_detections(frame, detections, jersey_info)
                
                # Lưu kết quả của frame này
                last_processed_frame = result_frame.copy()
                last_jersey_info = jersey_info.copy()
            
            else:
                # Nếu không xử lý frame này, dùng kết quả frame trước đó
                if last_processed_frame is not None:
                    result_frame = last_processed_frame.copy()
                else:
                    # Nếu chưa có frame nào được xử lý, dùng frame hiện tại
                    result_frame = frame.copy()
            
            # Ghi frame ra video output
            writer.write(result_frame)
    
    # Giải phóng tài nguyên
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nĐã xử lý xong video! Kết quả được lưu tại: {args.output_video}")

if __name__ == '__main__':
    main()

# python src/pipeline/main.py --detection-model models/detection/exp_custom_model/weights/best.pt --classification-model models/multilabel/best.pt --input-video data/football_test/Match_1864_1_0_subclip/Match_1864_1_0_subclip.mp4 --output-video results/output_Match_1864_1_0_subclip.mp4
# python src/pipeline/main.py --detection-model models/detection/exp_custom_model/weights/best.pt --classification-model models/multilabel/best.pt --input-video data/football_test/Match_1953_2_0_subclip/Match_1953_2_0_subclip.mp4 --output-video results/output_Match_1953_2_0_subclip.mp4