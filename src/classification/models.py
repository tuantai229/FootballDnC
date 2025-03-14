import torch
import torch.nn as nn

class JerseyNumberCNN(nn.Module):
    """
    Mạng CNN đơn giản cho bài toán phân loại số áo cầu thủ
    
    Kiến trúc:
    - 4 khối convolutional (mỗi khối có 2 lớp conv + batch norm + relu + maxpool)
    - 3 lớp fully connected với dropout để chống overfitting
    """
    def __init__(self, num_classes):
        """
        Khởi tạo mô hình
        
        Tham số:
            num_classes (int): Số lượng lớp cần phân loại
        """
        super().__init__()
        
        # Các khối convolutional
        self.conv1 = self.make_block(in_channels=3, out_channels=16)    # Input: 3x128x128, Output: 16x64x64
        self.conv2 = self.make_block(in_channels=16, out_channels=32)   # Output: 32x32x32
        self.conv3 = self.make_block(in_channels=32, out_channels=64)   # Output: 64x16x16
        self.conv4 = self.make_block(in_channels=64, out_channels=128)  # Output: 128x8x8

        # Lớp fully connected
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),  # Dropout 50% để chống overfitting
            nn.Linear(in_features=128 * 8 * 8, out_features=512),
            nn.ReLU()
        )
        
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU()
        )
        
        # Lớp output
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)
    
    def make_block(self, in_channels, out_channels, size=3, stride=1, padding=1):
        """
        Tạo một khối convolutional (2 lớp conv + batchnorm + relu + maxpool)
        
        Tham số:
            in_channels (int): Số kênh đầu vào
            out_channels (int): Số kênh đầu ra
            size (int): Kích thước kernel
            stride (int): Bước nhảy của convolution
            padding (int): Padding
            
        Returns:
            nn.Sequential: Khối các lớp đã được kết hợp
        """
        return nn.Sequential(
            # Conv layer 1
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                     kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            
            # Conv layer 2
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                     kernel_size=size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            
            # Max pooling
            nn.MaxPool2d(kernel_size=2)  # Giảm kích thước xuống 1/2
        )
    
    def forward(self, x):
        """
        Forward pass của mô hình
        
        Tham số:
            x (torch.Tensor): Batch ảnh đầu vào, shape [batch_size, 3, H, W]
            
        Returns:
            torch.Tensor: Logits đầu ra, shape [batch_size, num_classes]
        """
        # Truyền qua các khối convolutional
        x = self.conv1(x)  # 3x128x128 -> 16x64x64
        x = self.conv2(x)  # 16x64x64 -> 32x32x32
        x = self.conv3(x)  # 32x32x32 -> 64x16x16
        x = self.conv4(x)  # 64x16x16 -> 128x8x8
        
        # Flatten 
        x = x.view(x.shape[0], -1)  # 128x8x8 = 8192 features
        
        # Fully connected layers
        x = self.fc1(x)  # 8192 -> 512
        x = self.fc2(x)  # 512 -> 128
        x = self.fc3(x)  # 128 -> num_classes
        
        return x
    
if __name__ == '__main__':
    # Test model
    batch_size = 4
    img_size = 128
    num_classes = 12
    
    # Tạo dữ liệu test
    x = torch.rand(batch_size, 3, img_size, img_size)
    
    # Khởi tạo model
    model = JerseyNumberCNN(num_classes=num_classes)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Phải là [4, 12]