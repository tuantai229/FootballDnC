!pip install -q ultralytics
import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

# Kiểm tra GPU
!nvidia-smi

# Đường dẫn dataset
KAGGLE_INPUT_DIR = '/kaggle/input/footballdnc'
DATASET_DIR = f'{KAGGLE_INPUT_DIR}/dataset'
OUTPUT_DIR = '/kaggle/working'

# Tạo file YAML cho Kaggle
def prepare_kaggle_yaml():
    """Tạo file YAML phù hợp với đường dẫn Kaggle"""
    with open(f'{DATASET_DIR}/data.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Điều chỉnh đường dẫn cho Kaggle
    kaggle_yaml = yaml_data.copy()
    kaggle_yaml['path'] = DATASET_DIR
    
    # Lưu file YAML mới
    kaggle_yaml_path = f'{OUTPUT_DIR}/kaggle_dataset.yaml'
    
    with open(kaggle_yaml_path, 'w') as f:
        yaml.dump(kaggle_yaml, f, sort_keys=False)
    
    print(f"Đã tạo file YAML cho Kaggle tại: {kaggle_yaml_path}")
    return kaggle_yaml_path

# Chuẩn bị file YAML
kaggle_yaml_path = prepare_kaggle_yaml()

# Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate(experiment_id="test", model="yolo11n.pt", imgsz=640, 
                       epochs=50, batch=16, box=7.5, cls=0.5, patience=50):
    """
    Huấn luyện và đánh giá mô hình YOLO
    
    Tham số:
        experiment_id (str): ID của thử nghiệm
        model (str): Tên mô hình YOLO (yolo11n.pt, yolo11s.pt, ...)
        imgsz (int): Kích thước ảnh đầu vào
        epochs (int): Số lượng epochs
        batch (int): Kích thước batch
        box (float): Trọng số cho box loss
        cls (float): Trọng số cho class loss
        patience (int): Số epochs chờ đợi trước khi early stopping
    """
    print(f"\n=== THỬ NGHIỆM {experiment_id}: {model}, imgsz={imgsz}, box={box}, cls={cls} ===")
    
    # Xác định device
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Tải mô hình
    yolo_model = YOLO(model)
    
    # Thiết lập tham số huấn luyện (chỉ các tham số thay đổi)
    train_args = {
        'data': kaggle_yaml_path,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': device,
        'box': box,
        'cls': cls,
        'patience': patience,
        'project': OUTPUT_DIR,
        'name': f"exp_{experiment_id}",
        'exist_ok': True,
    }
    
    # Bắt đầu huấn luyện
    print("\nBắt đầu huấn luyện...")
    results = yolo_model.train(**train_args)
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình...")
    val_results = yolo_model.val(data=kaggle_yaml_path)
    
    # In kết quả chính
    print("\n=== KẾT QUẢ ===")
    print(f"mAP50: {val_results.box.map50:.4f}")
    print(f"mAP50-95: {val_results.box.map:.4f}")
    
    # In metrics cho từng lớp
    print("\nMetrics theo lớp:")
    with open(kaggle_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if isinstance(data_config['names'], list):
        class_names = data_config['names']
    else:
        class_names = list(data_config['names'].values())
    
    for i, class_name in enumerate(class_names):
        try:
            class_map = val_results.box.maps[i]
            print(f"{class_name}: mAP50 = {class_map:.4f}")
        except:
            print(f"{class_name}: Không có thông tin")
    
    print(f"\nKết quả đã lưu tại: {OUTPUT_DIR}/exp_{experiment_id}")
    return val_results


# CHẠY THỬ NGHIỆM
# Thử nghiệm 0: Baseline với tham số mặc định
"""
baseline_results = train_and_evaluate(
    experiment_id="0_baseline",
    model="yolo11n.pt",  # Model nhỏ nhất
    imgsz=640,           # Kích thước ảnh mặc định
    epochs=50,           # Epochs
    batch=16,            # Batch size tiêu chuẩn
    box=7.5,             # Box loss mặc định
    cls=0.5,             # Class loss mặc định
    patience=25         # Patience
)
"""

# CÁC THỬ NGHIỆM KHÁC
# Mỗi lần chỉ nên chạy một thử nghiệm để tiết kiệm thời gian và tài nguyên

# Thử nghiệm 1: Model lớn hơn
"""
experiment1_results = train_and_evaluate(
    experiment_id="1_model_s",
    model="yolo11s.pt",  # Model cỡ vừa
    imgsz=640,
    epochs=50,
    batch=16,
    box=7.5,
    cls=0.5,
    patience=25
)
"""

# Thử nghiệm 2: Tăng kích thước ảnh
"""
experiment2_results = train_and_evaluate(
    experiment_id="2_imgsz_1280", 
    model="yolo11n.pt",
    imgsz=1280,          # Tăng kích thước ảnh
    epochs=50,
    batch=8,             # Giảm batch size do kích thước ảnh lớn
    box=7.5,
    cls=0.5,
    patience=25
)
"""

# Thử nghiệm 3: Tăng box và class loss
"""
experiment3_results = train_and_evaluate(
    experiment_id="3_high_loss",
    model="yolo11n.pt",
    imgsz=640,
    epochs=50,
    batch=16,
    box=10.0,            # Tăng box loss
    cls=3.0,             # Tăng class loss đáng kể
    patience=25
)
"""

# Thử nghiệm 4: Kết hợp các tham số
"""
experiment4_results = train_and_evaluate(
    experiment_id="4_combined",
    model="yolo11s.pt",   # Model tốt hơn
    imgsz=1280,           # Kích thước ảnh lớn hơn
    epochs=50,
    batch=8,
    box=10.0,             # Tăng box loss
    cls=3.0,              # Tăng class loss
    patience=25
)
"""

# Thử nghiệm tùy chỉnh mới: Thay đổi tham số cần thiết

experiment_custom = train_and_evaluate(
    experiment_id="custom",   # Thay đổi ID
    model="yolo11m.pt",   # Model m
    imgsz=1280,
    epochs=100,
    batch=8,
    box=10.0,
    cls=3.0,
    patience=50
)