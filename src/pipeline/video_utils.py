import cv2

def read_video(video_path):
    """Đọc video từ đường dẫn"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")
    
    # Lấy thông tin video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    return cap, width, height, fps

def create_video_writer(output_path, width, height, fps):
    """Tạo video writer để ghi video"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer