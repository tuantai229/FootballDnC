import torch
from ultralytics import YOLO

class Detector:
    """Wrapper cho model YOLO để phát hiện cầu thủ và bóng"""
    def __init__(self, model_path, conf_threshold=0.3, device=None):
        # Xác định device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.backends.mps.is_available():
                self.device = 'mps'  # Apple Silicon GPU
        else:
            self.device = device
            
        print(f"Detector sử dụng thiết bị: {self.device}")
            
        # Load model
        self.model = YOLO(model_path)
        
        # Lưu ngưỡng confidence
        self.conf_threshold = conf_threshold
        
    def detect(self, frame):
        """Phát hiện cầu thủ và bóng trong frame"""
        # Thực hiện inference
        results = self.model(frame, conf=self.conf_threshold, device=self.device)
        
        # Xử lý kết quả
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i, box in enumerate(boxes):
                # Lấy tọa độ bbox (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Lấy confidence score
                conf = box.conf[0].cpu().numpy()
                
                # Lấy class id và tên
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]
                
                # Thêm vào danh sách kết quả
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': float(conf)
                })
        
        return detections