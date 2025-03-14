import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.autonotebook import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix

from datasets import JerseyNumberDataset
from models import JerseyNumberCNN

def train(args):
    """
    Huấn luyện mô hình phân loại số áo cầu thủ
    
    Tham số:
        args: Các đối số từ command line
    """
    # Lấy tham số từ args
    dataset_path = args.dataset_path
    logging_path = args.logging_path
    checkpoint_path = args.checkpoint_path
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    img_size = args.img_size
    
    # ======= THIẾT LẬP THIẾT BỊ (DEVICE) =======
    # Sử dụng GPU nếu có
    if torch.backends.mps.is_available() and args.device == 'mps':
        device = torch.device("mps")  # Apple Silicon GPU
        print("Sử dụng GPU của Apple Silicon (MPS)")
    elif torch.cuda.is_available() and args.device == 'cuda':
        device = torch.device("cuda")  # NVIDIA GPU
        print("Sử dụng GPU NVIDIA (CUDA)")
    else:
        device = torch.device("cpu")
        print("Sử dụng CPU")

    # ======= TẠO THƯ MỤC LƯU KẾT QUẢ =======
    os.makedirs(logging_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # ======= CHUẨN BỊ DỮ LIỆU =======
    # Data augmentation cho tập train
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),  # Lật ngang ngẫu nhiên
        transforms.RandomRotation(10),      # Xoay ảnh ngẫu nhiên ±10 độ
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Điều chỉnh màu sắc
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])  # Chuẩn hóa theo ImageNet
    ])
    
    # Chỉ resize và chuẩn hóa cho tập val (không augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])

    # Tạo datasets
    train_dataset = JerseyNumberDataset(
        root=dataset_path, 
        train=True, 
        transform=train_transform
    )
    
    val_dataset = JerseyNumberDataset(
        root=dataset_path, 
        train=False, 
        transform=val_transform
    )
    
    # Tạo dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,              # Trộn dữ liệu
        num_workers=args.workers,  # Số luồng xử lý
        drop_last=True             # Bỏ batch cuối nếu không đủ kích thước
    )
    
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,             # Không cần trộn dữ liệu validation
        num_workers=args.workers
    )
    
    # Thông tin dataset
    num_classes = len(train_dataset.classes)
    print(f"Số lượng lớp: {num_classes}")
    print(f"Các lớp: {train_dataset.classes}")
    print(f"Số lượng ảnh train: {len(train_dataset)}")
    print(f"Số lượng ảnh validation: {len(val_dataset)}")

    # ======= KHỞI TẠO MÔ HÌNH =======
    model = JerseyNumberCNN(num_classes=num_classes).to(device)

    # ======= THIẾT LẬP LOSS, OPTIMIZER, SCHEDULER =======
    criterion = nn.CrossEntropyLoss()  # Hàm mất mát cho phân loại nhiều lớp
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Scheduler giảm learning rate khi validation loss không giảm
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # ======= TENSORBOARD =======
    writer = SummaryWriter(logging_path)
    
    # ======= TRAINING LOOP =======
    best_accuracy = 0.0
    
    print(f"\n{'='*20} BẮT ĐẦU HUẤN LUYỆN {'='*20}")
    print(f"Thiết bị: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ----- TRAIN PHASE -----
        model.train()  # Chuyển model sang chế độ train
        train_losses = []
        train_predictions = []
        train_labels = []
        
        # Duyệt qua từng batch dữ liệu train
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in progress_bar:
            # Chuyển dữ liệu sang thiết bị (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)
            
            # Reset gradient
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Lưu loss và predictions để tính metrics
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # Cập nhật progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Tính metrics cho epoch
        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_labels, train_predictions)
        
        # ----- VALIDATION PHASE -----
        model.eval()  # Chuyển model sang chế độ evaluation
        val_losses = []
        val_predictions = []
        val_labels = []
        
        # Không tính gradient trong quá trình validation
        with torch.no_grad():
            progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in progress_bar:
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Lưu loss và predictions
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                # Cập nhật progress bar
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Tính metrics cho validation
        val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_predictions)
        
        # Thời gian cho một epoch
        epoch_time = time.time() - start_time
        
        # ----- GHI LOG VÀ HIỂN THỊ METRICS -----
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        
        # In thông tin
        print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Cập nhật learning rate dựa trên validation loss
        scheduler.step(val_loss)
        
        # ----- LƯU MODEL -----
        # Lưu model cuối cùng của epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "last.pt"))
        
        # Lưu model tốt nhất nếu có cải thiện accuracy
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "best.pt"))
            best_accuracy = val_acc
            print(f"[BEST] Đã lưu model với accuracy: {best_accuracy:.4f}")
            
            # Vẽ confusion matrix khi có model tốt nhất mới
            if epoch > 0:  # Bỏ qua epoch đầu
                cm = confusion_matrix(val_labels, val_predictions)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=train_dataset.classes,
                           yticklabels=train_dataset.classes)
                plt.title(f'Confusion Matrix - Epoch {epoch+1}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(os.path.join(checkpoint_path, f'confusion_matrix_epoch_{epoch+1}.png'))
                plt.close()
    
    print(f"\n{'='*20} HUẤN LUYỆN HOÀN TẤT {'='*20}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model đã được lưu tại: {checkpoint_path}")
    
    return model, best_accuracy

def main():
    """Hàm chính xử lý command line arguments và gọi hàm train"""
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình phân loại số áo cầu thủ')
    parser.add_argument('--dataset_path', type=str, default='./classification_dataset',
                       help='Đường dẫn đến dataset')
    parser.add_argument('--logging_path', type=str, default='./logs/classification',
                       help='Đường dẫn lưu tensorboard logs')
    parser.add_argument('--checkpoint_path', type=str, default='./models/classification',
                       help='Đường dẫn lưu model checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Số lượng epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--img_size', type=int, default=128,
                       help='Kích thước ảnh đầu vào')
    parser.add_argument('--workers', type=int, default=2,
                       help='Số lượng worker threads cho dataloader')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'],
                       help='Thiết bị để huấn luyện (cpu, cuda, mps)')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()