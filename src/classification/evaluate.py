import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import JerseyNumberDataset
from models import JerseyNumberCNN

def evaluate_model(model_path, dataset_path, img_size=128, batch_size=32, workers=2, device=None):
    """
    Đánh giá mô hình phân loại số áo cầu thủ
    
    Tham số:
        model_path (str): Đường dẫn đến file model (.pt)
        dataset_path (str): Đường dẫn đến dataset
        img_size (int): Kích thước ảnh đầu vào
        batch_size (int): Kích thước batch
        workers (int): Số lượng worker threads cho dataloader
        device (str): Thiết bị để đánh giá (cpu, cuda, mps)
        
    Returns:
        tuple: (accuracy, classification_report, confusion_matrix)
    """
    # ======= THIẾT LẬP THIẾT BỊ =======
    if device is None:
        # Tự động chọn thiết bị phù hợp
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
    # Transform cho validation (chỉ resize và chuẩn hóa)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo validation dataset và dataloader
    val_dataset = JerseyNumberDataset(root=dataset_path, train=False, transform=val_transform)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers
    )
    
    # Thông tin dataset
    num_classes = len(val_dataset.classes)
    print(f"Số lượng lớp: {num_classes}")
    print(f"Các lớp: {val_dataset.classes}")
    print(f"Số lượng ảnh validation: {len(val_dataset)}")
    
    # ======= TẠO VÀ LOAD MÔ HÌNH =======
    model = JerseyNumberCNN(num_classes=num_classes)
    
    # Load trọng số model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Đã load model thành công từ: {model_path}")
    except Exception as e:
        print(f"Lỗi khi load model: {e}")
        return None, None, None
    
    model = model.to(device)
    model.eval()  # Chuyển sang chế độ evaluation
    
    # ======= ĐÁNH GIÁ MÔ HÌNH =======
    print("\nĐang đánh giá model...")
    all_predictions = []
    all_labels = []
    all_scores = []  # Lưu confidence scores
    
    with torch.no_grad():  # Không tính gradient
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            
            # Lấy probability scores bằng softmax
            scores = torch.nn.functional.softmax(outputs, dim=1)
            
            # Lấy class prediction (chỉ số lớn nhất)
            _, predictions = torch.max(outputs, 1)
            
            # Lưu kết quả
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_scores.append(scores.cpu().numpy())
    
    # Ghép các batch scores lại thành một mảng
    all_scores = np.vstack(all_scores)
    
    # ======= TÍNH TOÁN METRICS =======
    # Accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Classification report (precision, recall, f1-score)
    class_names = val_dataset.classes
    report = classification_report(all_labels, all_predictions, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    # ======= TẠO CONFUSION MATRIX =======
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
              xticklabels=class_names,
              yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Tạo thư mục output nếu chưa có
    os.makedirs('src/classification/evaluation_results', exist_ok=True)
    plt.savefig('src/classification/evaluation_results/confusion_matrix.png')
    print("Đã lưu confusion matrix tại 'src/classification/evaluation_results/confusion_matrix.png'")
    
    # ======= HIỂN THỊ MỘT SỐ MẪU DỰ ĐOÁN =======
    visualize_predictions(val_dataset, model, device, num_samples=10)
    
    return accuracy, report, cm

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
    
    # Tạo lưới ảnh 2x5
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    axes = axes.flatten()
    
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            if i >= len(axes):
                break
                
            # Lấy ảnh và nhãn từ dataset
            image, label = dataset[idx]
            
            # Dự đoán
            image_tensor = image.unsqueeze(0).to(device)  # Thêm batch dimension
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            
            # Lấy tên lớp
            true_class = dataset.classes[label]
            pred_class = dataset.classes[predicted.item()]
            
            # Hiển thị ảnh
            img = image.permute(1, 2, 0).cpu().numpy()
            # Đưa về khoảng [0,1] để hiển thị
            img = (img - img.min()) / (img.max() - img.min())
            
            axes[i].imshow(img)
            color = 'green' if true_class == pred_class else 'red'
            axes[i].set_title(f"True: {true_class}\nPred: {pred_class}", color=color)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('src/classification/evaluation_results/prediction_samples.png')
    print("Đã lưu mẫu dự đoán tại 'src/classification/evaluation_results/prediction_samples.png'")

def main():
    """Hàm chính xử lý command line arguments và gọi hàm evaluate_model"""
    parser = argparse.ArgumentParser(description='Đánh giá mô hình phân loại số áo cầu thủ')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Đường dẫn đến file model (.pt)')
    parser.add_argument('--dataset_path', type=str, default='./classification_dataset',
                        help='Đường dẫn đến dataset')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Kích thước ảnh đầu vào')
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
        img_size=args.img_size,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device
    )

if __name__ == '__main__':
    main()