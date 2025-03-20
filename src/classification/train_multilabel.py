import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
import matplotlib.pyplot as plt

from datasets import JerseyNumberDataset
from models_pretrained import JerseyClassifierResNet

def train(args):
    """
    Huấn luyện mô hình đa nhãn
    
    Tham số:
        args: Tham số dòng lệnh
    """
    # Lấy tham số
    dataset_path = args.dataset_path
    logging_path = args.logging_path
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    
    # ======= THIẾT LẬP THIẾT BỊ =======
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
    from torchvision import transforms
    
    # Transform cho training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet yêu cầu 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform cho validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo dataset
    train_dataset = JerseyNumberDataset(
        root=dataset_path,
        train=True,
        transform=train_transform,
        return_color_label=True  # Quan trọng: Trả về cả nhãn màu áo
    )
    
    val_dataset = JerseyNumberDataset(
        root=dataset_path,
        train=False,
        transform=val_transform,
        return_color_label=True
    )
    
    # Tạo dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    # Hiển thị thông tin dataset
    print(f"Số lượng lớp số áo: {len(train_dataset.classes)}")
    print(f"Các lớp số áo: {train_dataset.classes}")
    print(f"Số lớp màu áo: 2 (white, black)")
    print(f"Số lượng ảnh train: {len(train_dataset)}")
    print(f"Số lượng ảnh val: {len(val_dataset)}")

    # ======= KHỞI TẠO MÔ HÌNH =======
    model = JerseyClassifierResNet(
        num_jersey_classes=len(train_dataset.classes),
        num_color_classes=2
    ).to(device)

    # ======= THIẾT LẬP LOSS, OPTIMIZER =======
    # Sử dụng CrossEntropyLoss cho cả hai nhiệm vụ
    criterion_jersey = nn.CrossEntropyLoss()
    criterion_color = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # ======= TENSORBOARD =======
    writer = SummaryWriter(logging_path)
    
    # ======= TRAINING LOOP =======
    best_combined_acc = 0.0
    
    print(f"\n{'='*20} BẮT ĐẦU HUẤN LUYỆN {'='*20}")
    print(f"Thiết bị: {device}")
    print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ----- TRAIN PHASE -----
        model.train()
        train_jersey_losses = []
        train_color_losses = []
        train_jersey_preds = []
        train_jersey_labels = []
        train_color_preds = []
        train_color_labels = []
        
        # Duyệt qua từng batch dữ liệu training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, jersey_targets, color_targets in progress_bar:
            # Chuyển dữ liệu sang thiết bị
            images = images.to(device)
            jersey_targets = jersey_targets.to(device)
            color_targets = color_targets.to(device)
            
            # Xóa gradient
            optimizer.zero_grad()
            
            # Forward pass
            jersey_outputs, color_outputs = model(images)
            
            # Tính loss
            jersey_loss = criterion_jersey(jersey_outputs, jersey_targets)
            color_loss = criterion_color(color_outputs, color_targets)
            
            # Tổng loss
            total_loss = jersey_loss + color_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Lưu loss và predictions
            train_jersey_losses.append(jersey_loss.item())
            train_color_losses.append(color_loss.item())
            
            # Lấy chỉ số lớp có xác suất cao nhất
            _, jersey_preds = torch.max(jersey_outputs, 1)
            _, color_preds = torch.max(color_outputs, 1)
            
            # Lưu predictions và labels để tính accuracy
            train_jersey_preds.extend(jersey_preds.cpu().numpy())
            train_jersey_labels.extend(jersey_targets.cpu().numpy())
            train_color_preds.extend(color_preds.cpu().numpy())
            train_color_labels.extend(color_targets.cpu().numpy())
            
            # Cập nhật progress bar
            progress_bar.set_postfix(
                j_loss=f"{jersey_loss.item():.4f}",
                c_loss=f"{color_loss.item():.4f}"
            )
        
        # Tính metrics cho epoch
        train_jersey_loss = np.mean(train_jersey_losses)
        train_color_loss = np.mean(train_color_losses)
        train_jersey_acc = accuracy_score(train_jersey_labels, train_jersey_preds)
        train_color_acc = accuracy_score(train_color_labels, train_color_preds)
        
        # ----- VALIDATION PHASE -----
        model.eval()
        val_jersey_losses = []
        val_color_losses = []
        val_jersey_preds = []
        val_jersey_labels = []
        val_color_preds = []
        val_color_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, jersey_targets, color_targets in progress_bar:
                # Chuyển dữ liệu sang thiết bị
                images = images.to(device)
                jersey_targets = jersey_targets.to(device)
                color_targets = color_targets.to(device)
                
                # Forward pass
                jersey_outputs, color_outputs = model(images)
                
                # Tính loss
                jersey_loss = criterion_jersey(jersey_outputs, jersey_targets)
                color_loss = criterion_color(color_outputs, color_targets)
                
                # Lưu loss và predictions
                val_jersey_losses.append(jersey_loss.item())
                val_color_losses.append(color_loss.item())
                
                # Lấy chỉ số lớp có xác suất cao nhất
                _, jersey_preds = torch.max(jersey_outputs, 1)
                _, color_preds = torch.max(color_outputs, 1)
                
                # Lưu predictions và labels
                val_jersey_preds.extend(jersey_preds.cpu().numpy())
                val_jersey_labels.extend(jersey_targets.cpu().numpy())
                val_color_preds.extend(color_preds.cpu().numpy())
                val_color_labels.extend(color_targets.cpu().numpy())
                
                # Cập nhật progress bar
                progress_bar.set_postfix(
                    j_loss=f"{jersey_loss.item():.4f}",
                    c_loss=f"{color_loss.item():.4f}"
                )
        
        # Tính metrics cho validation
        val_jersey_loss = np.mean(val_jersey_losses)
        val_color_loss = np.mean(val_color_losses)
        val_jersey_acc = accuracy_score(val_jersey_labels, val_jersey_preds)
        val_color_acc = accuracy_score(val_color_labels, val_color_preds)
        
        # Tính combined accuracy (trung bình)
        val_combined_acc = (val_jersey_acc + val_color_acc) / 2
        
        # Thời gian cho epoch
        epoch_time = time.time() - start_time
        
        # ----- GHI LOG VÀ HIỂN THỊ KẾT QUẢ -----
        writer.add_scalar("Train/JerseyLoss", train_jersey_loss, epoch)
        writer.add_scalar("Train/ColorLoss", train_color_loss, epoch)
        writer.add_scalar("Train/JerseyAcc", train_jersey_acc, epoch)
        writer.add_scalar("Train/ColorAcc", train_color_acc, epoch)
        writer.add_scalar("Val/JerseyLoss", val_jersey_loss, epoch)
        writer.add_scalar("Val/ColorLoss", val_color_loss, epoch)
        writer.add_scalar("Val/JerseyAcc", val_jersey_acc, epoch)
        writer.add_scalar("Val/ColorAcc", val_color_acc, epoch)
        writer.add_scalar("Val/CombinedAcc", val_combined_acc, epoch)
        
        # Hiển thị kết quả
        print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
        print(f"Train - Jersey Loss: {train_jersey_loss:.4f}, Jersey Acc: {train_jersey_acc:.4f}")
        print(f"Train - Color Loss: {train_color_loss:.4f}, Color Acc: {train_color_acc:.4f}")
        print(f"Val - Jersey Loss: {val_jersey_loss:.4f}, Jersey Acc: {val_jersey_acc:.4f}")
        print(f"Val - Color Loss: {val_color_loss:.4f}, Color Acc: {val_color_acc:.4f}")
        print(f"Val - Combined Acc: {val_combined_acc:.4f}")
        
        # Cập nhật learning rate
        scheduler.step(val_jersey_loss + val_color_loss)
        
        # ----- LƯU MODEL -----
        # Lưu model cuối cùng của epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_path, "last.pt"))
        
        # Lưu model tốt nhất dựa trên combined accuracy
        if val_combined_acc > best_combined_acc:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "best.pt"))
            best_combined_acc = val_combined_acc
            print(f"[BEST] Đã lưu model với combined accuracy: {best_combined_acc:.4f}")
    
    print(f"\n{'='*20} HUẤN LUYỆN HOÀN TẤT {'='*20}")
    print(f"Best combined accuracy: {best_combined_acc:.4f}")
    print(f"Model đã được lưu tại: {checkpoint_path}")
    
    return model, best_combined_acc

def main():
    parser = argparse.ArgumentParser(description='Huấn luyện mô hình đa nhãn cho phân loại số áo và màu áo')
    parser.add_argument('--dataset_path', type=str, default='./classification_dataset',
                       help='Đường dẫn đến dataset')
    parser.add_argument('--logging_path', type=str, default='./logs/multilabel',
                       help='Đường dẫn lưu tensorboard logs')
    parser.add_argument('--checkpoint_path', type=str, default='./models/multilabel',
                       help='Đường dẫn lưu model checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Số lượng epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--workers', type=int, default=2,
                       help='Số lượng worker threads cho dataloader')
    parser.add_argument('--device', type=str, default='mps', choices=['cpu', 'cuda', 'mps'],
                       help='Thiết bị để huấn luyện (cpu, cuda, mps)')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()