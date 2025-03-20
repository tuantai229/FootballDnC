import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import JerseyNumberDataset
from models_pretrained import JerseyClassifierResNet

def evaluate_model(model_path, dataset_path, batch_size=32, workers=2, device=None):
    """
    Đánh giá mô hình đa nhãn
    
    Tham số:
        model_path (str): Đường dẫn đến file model (.pt)
        dataset_path (str): Đường dẫn đến dataset
        batch_size (int): Kích thước batch
        workers (int): Số lượng worker threads cho dataloader
        device (str): Thiết bị để đánh giá (cpu, cuda, mps)
    """
    # ======= THIẾT LẬP THIẾT BỊ =======
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Sử dụng GPU của Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Sử dụng GPU NVIDIA (CUDA)")
        else:
            device = torch.device("cpu")
            print("Sử dụng CPU")
    else:
        device = torch.device(device)
    
    # ======= CHUẨN BỊ DỮ LIỆU =======
    # Transform cho validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet yêu cầu kích thước 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo dataset
    val_dataset = JerseyNumberDataset(
        root=dataset_path,
        train=False,
        transform=val_transform,
        return_color_label=True
    )
    
    # Tạo dataloader
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )
    
    # Hiển thị thông tin dataset
    print(f"Số lượng lớp số áo: {len(val_dataset.classes)}")
    print(f"Các lớp số áo: {val_dataset.classes}")
    print(f"Số lớp màu áo: 2 (white, black)")
    print(f"Số lượng ảnh validation: {len(val_dataset)}")
    
    # ======= TẠO VÀ LOAD MÔ HÌNH =======
    model = JerseyClassifierResNet(
        num_jersey_classes=len(val_dataset.classes),
        num_color_classes=2
    )
    
    # Load trọng số model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã load model thành công từ: {model_path}")
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return None
    
    model = model.to(device)
    model.eval()  # Chuyển sang chế độ evaluation
    
    # ======= ĐÁNH GIÁ MÔ HÌNH =======
    print("\nĐang đánh giá model...")
    jersey_predictions = []
    jersey_labels = []
    color_predictions = []
    color_labels = []
    
    with torch.no_grad():  # Không tính gradient
        for images, jersey_targets, color_targets in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Forward pass
            jersey_outputs, color_outputs = model(images)
            
            # Lấy class prediction (chỉ số lớn nhất)
            _, jersey_preds = torch.max(jersey_outputs, 1)
            _, color_preds = torch.max(color_outputs, 1)
            
            # Lưu kết quả
            jersey_predictions.extend(jersey_preds.cpu().numpy())
            jersey_labels.extend(jersey_targets.numpy())
            color_predictions.extend(color_preds.cpu().numpy())
            color_labels.extend(color_targets.numpy())
    
    # ======= TÍNH TOÁN METRICS =======
    # Tạo thư mục lưu kết quả
    os.makedirs('evaluation_results/multilabel', exist_ok=True)
    
    # Accuracy cho số áo
    jersey_acc = accuracy_score(jersey_labels, jersey_predictions)
    print(f"Accuracy số áo: {jersey_acc:.4f}")
    
    # Accuracy cho màu áo
    color_acc = accuracy_score(color_labels, color_predictions)
    print(f"Accuracy màu áo: {color_acc:.4f}")
    
    # Combined accuracy
    combined_acc = (jersey_acc + color_acc) / 2
    print(f"Combined accuracy: {combined_acc:.4f}")
    
    # Classification report số áo
    print("\nBáo cáo phân loại số áo:")
    jersey_report = classification_report(
        jersey_labels,
        jersey_predictions,
        target_names=val_dataset.classes
    )
    print(jersey_report)
    
    # Classification report màu áo
    print("\nBáo cáo phân loại màu áo:")
    # Kiểm tra số lớp duy nhất trong dữ liệu
    unique_color_labels = np.unique(color_labels)
    if len(unique_color_labels) == 1:
        if unique_color_labels[0] == 0:
            color_class_names = ["white"]
        else:
            color_class_names = ["black"]
        print(f"Chỉ có một lớp màu áo trong dữ liệu validation: {color_class_names[0]}")
        print(f"Accuracy: 1.0000 (tầm thường vì chỉ có một lớp)")
    else:
        color_report = classification_report(
            color_labels,
            color_predictions,
            target_names=["white", "black"]
        )
        print(color_report)
    
    # ======= TẠO CONFUSION MATRIX =======
    # Confusion matrix cho số áo
    plt.figure(figsize=(12, 10))
    cm_jersey = confusion_matrix(jersey_labels, jersey_predictions)
    sns.heatmap(cm_jersey, annot=True, fmt='d', cmap='Blues',
               xticklabels=val_dataset.classes,
               yticklabels=val_dataset.classes)
    plt.title('Confusion Matrix - Số áo')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_results/multilabel/confusion_matrix_jersey.png')
    
    # Confusion matrix cho màu áo
    plt.figure(figsize=(8, 6))
    cm_color = confusion_matrix(color_labels, color_predictions)
    sns.heatmap(cm_color, annot=True, fmt='d', cmap='Blues',
               xticklabels=["white", "black"],
               yticklabels=["white", "black"])
    plt.title('Confusion Matrix - Màu áo')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation_results/multilabel/confusion_matrix_color.png')
    
    # ======= HIỂN THỊ MỘT SỐ MẪU DỰ ĐOÁN =======
    visualize_predictions(val_dataset, model, device, num_samples=10)
    
    return {
        'jersey_accuracy': jersey_acc,
        'color_accuracy': color_acc,
        'combined_accuracy': combined_acc,
        'jersey_report': jersey_report,
        'color_report': color_report
    }

def visualize_predictions(dataset, model, device, num_samples=10):
    """
    Hiển thị một số mẫu với dự đoán của model
    
    Tham số:
        dataset: Dataset chứa các ảnh và nhãn
        model: Mô hình đã huấn luyện
        device: Thiết bị để chạy model
        num_samples: Số lượng mẫu để hiển thị
    """
    # Chọn ngẫu nhiên một số mẫu từ dataset
    indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    # Tạo lưới ảnh
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            # Lấy ảnh và nhãn từ dataset
            image, jersey_label, color_label = dataset[idx]
            
            # Dự đoán
            image_tensor = image.unsqueeze(0).to(device)  # Thêm batch dimension
            jersey_output, color_output = model(image_tensor)
            _, jersey_pred = torch.max(jersey_output, 1)
            _, color_pred = torch.max(color_output, 1)
            
            # Lấy tên lớp
            jersey_true_class = dataset.classes[jersey_label]
            jersey_pred_class = dataset.classes[jersey_pred.item()]
            color_true_class = "black" if color_label == 1 else "white"
            color_pred_class = "black" if color_pred.item() == 1 else "white"
            
            # Hiển thị ảnh
            img = image.permute(1, 2, 0).cpu().numpy()
            # Đưa về khoảng [0,1] để hiển thị
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[i].imshow(img)
            
            # Tạo tiêu đề với màu phù hợp
            jersey_color = 'green' if jersey_true_class == jersey_pred_class else 'red'
            color_color = 'green' if color_true_class == color_pred_class else 'red'
            
            # Hiển thị kết quả
            axes[i].set_title(f"Số áo: {jersey_true_class}→{jersey_pred_class} ({jersey_color})\n"
                             f"Màu áo: {color_true_class}→{color_pred_class} ({color_color})")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('evaluation_results/multilabel/prediction_samples.png')
    print("Đã lưu mẫu dự đoán tại 'evaluation_results/multilabel/prediction_samples.png'")

def main():
    parser = argparse.ArgumentParser(description='Đánh giá mô hình đa nhãn')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Đường dẫn đến file model (.pt)')
    parser.add_argument('--dataset_path', type=str, default='./classification_dataset',
                       help='Đường dẫn đến dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--workers', type=int, default=2,
                       help='Số lượng worker threads cho dataloader')
    parser.add_argument('--device', type=str, default=None,
                       help='Thiết bị để đánh giá (cpu, cuda, mps)')
    
    args = parser.parse_args()
    evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device
    )

if __name__ == '__main__':
    main()