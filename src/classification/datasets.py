import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class JerseyNumberDataset(Dataset):
    """
    Dataset cho bài toán phân loại số áo cầu thủ.
    Cấu trúc thư mục dataset:
    - root/
        - train/
            - class1/
                - img1.jpg
                - img2.jpg
                ...
            - class2/
                ...
        - val/
            - class1/
                ...
    """
    def __init__(self, root, train=True, transform=None, return_color_label=False):
        """
        Khởi tạo dataset
        
        Tham số:
            root (str): Thư mục gốc chứa dataset
            train (bool): Nếu True, sử dụng tập train, ngược lại sử dụng tập val
            transform (callable, optional): Biến đổi áp dụng lên ảnh
            return_color_label (bool): Nếu True, trả về cả nhãn màu áo
        """
        self.root = root
        self.train = train
        self.return_color_label = return_color_label
        
        # Xác định transform
        if transform is None:
            # Transform mặc định nếu không được cung cấp
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),  # Resize về 128x128
                transforms.ToTensor(),  # Chuyển sang tensor [0,1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])  # Chuẩn hóa ImageNet
            ])
        else:
            self.transform = transform
        
        # Xác định đường dẫn dữ liệu (train hoặc val)
        data_dir = os.path.join(root, "train" if train else "val")
        
        # Lấy danh sách các lớp (thư mục con)
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')])
        
        # Danh sách lưu đường dẫn ảnh và nhãn
        self.images = []
        self.labels = []
        
        # Duyệt qua từng lớp và thu thập ảnh
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            
            # Lấy tất cả file ảnh trong thư mục
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.jpeg')) and not img_name.startswith('.'):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        """Trả về số lượng ảnh trong dataset"""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Lấy một mẫu từ dataset
        
        Tham số:
            idx (int): Chỉ số của mẫu
            
        Returns:
            tuple: (ảnh, nhãn) hoặc (ảnh, nhãn, nhãn_màu) nếu return_color_label=True
        """
        # Đọc ảnh từ đường dẫn
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Đảm bảo ảnh là RGB
        
        # Áp dụng transform lên ảnh
        image = self.transform(image)
        label = self.labels[idx]
        
        # Nếu không cần trả về nhãn màu áo
        if not self.return_color_label:
            return image, label
        
        # Trích xuất thông tin màu áo từ tên file
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        # Mặc định là white (0)
        color_label = 0
        
        # Tìm phần có thông tin màu áo
        for part in parts:
            if part.startswith("color"):
                color_value = part[5:].split('.')[0]  # Cắt "color" và loại bỏ phần mở rộng file để lấy giá trị
                if color_value == "black":
                    color_label = 1
                break
        
        # Trả về cả nhãn số áo và nhãn màu áo
        return image, label, color_label


# Test code
if __name__ == '__main__':
    # Test dataset
    train_dataset = JerseyNumberDataset(root="classification_dataset", train=True, return_color_label=True)
    val_dataset = JerseyNumberDataset(root="classification_dataset", train=False, return_color_label=True)

    print(f"Số lượng ảnh train: {len(train_dataset)}")
    print(f"Số lượng ảnh val: {len(val_dataset)}")
    print(f"Các lớp: {train_dataset.classes}")

    # Kiểm tra một mẫu
    image, label, color_label = train_dataset[0]
    print(f"Kích thước ảnh: {image.shape}")
    print(f"Nhãn số áo: {label} ({train_dataset.classes[label]})")
    print(f"Nhãn màu áo: {color_label} ({'black' if color_label == 1 else 'white'})")

    # Test dataloader
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    # In ra kích thước batch đầu tiên
    for data in train_loader:
        if len(data) == 3:
            images, labels, color_labels = data
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Color labels shape: {color_labels.shape}")
        else:
            images, labels = data
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
        break