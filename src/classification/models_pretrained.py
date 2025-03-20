import torch
import torch.nn as nn
import torchvision.models as models

class JerseyClassifierResNet(nn.Module):
    """
    Mô hình phân loại số áo và màu áo sử dụng ResNet50 pretrained
    """
    def __init__(self, num_jersey_classes=12, num_color_classes=2):
        """
        Khởi tạo mô hình
        
        Tham số:
            num_jersey_classes (int): Số lượng lớp số áo (mặc định: 12)
            num_color_classes (int): Số lượng lớp màu áo (mặc định: 2)
        """
        super(JerseyClassifierResNet, self).__init__()
        
        # Bước 1: Tải ResNet50 đã được huấn luyện trên ImageNet
        self.backbone = models.resnet50(pretrained=True)
        
        # Bước 2: Xóa lớp fully connected cuối cùng
        # ResNet50 có đầu ra 2048 đặc trưng từ avgpool
        in_features = self.backbone.fc.in_features  # 2048
        
        # Bước 3: Thay thế bằng hai đầu ra (heads)
        # Xóa lớp FC gốc
        del self.backbone.fc
        
        # Thêm lớp FC cho phân loại số áo
        self.fc_jersey = nn.Linear(in_features, num_jersey_classes)
        
        # Thêm lớp FC cho phân loại màu áo
        self.fc_color = nn.Linear(in_features, num_color_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Tham số:
            x (torch.Tensor): Batch hình ảnh đầu vào, shape [batch_size, 3, H, W]
            
        Returns:
            tuple: (jersey_logits, color_logits) - Logits cho số áo và màu áo
        """
        # Bước 1: Trích xuất đặc trưng từ backbone (không bao gồm lớp FC cuối cùng)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Bước 2: Average pooling để giảm kích thước
        x = self.backbone.avgpool(x)
        
        # Bước 3: Làm phẳng đặc trưng
        x = torch.flatten(x, 1)  # [batch_size, 2048]
        
        # Bước 4: Hai đầu ra song song
        jersey_logits = self.fc_jersey(x)  # [batch_size, num_jersey_classes]
        color_logits = self.fc_color(x)    # [batch_size, num_color_classes]
        
        return jersey_logits, color_logits


# Mã test mô hình
if __name__ == '__main__':
    # Tạo mô hình
    model = JerseyClassifierResNet(num_jersey_classes=12, num_color_classes=2)
    
    # Tạo dữ liệu test giả lập
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch 1, 3 kênh RGB, 224x224 pixels
    
    # Forward pass
    jersey_logits, color_logits = model(dummy_input)
    
    # In ra kích thước đầu ra
    print(f"Kích thước logits số áo: {jersey_logits.shape}")  # [1, 12]
    print(f"Kích thước logits màu áo: {color_logits.shape}")  # [1, 2]