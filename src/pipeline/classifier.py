import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2

class JerseyClassifierResNet(nn.Module):
    """Mô hình phân loại số áo và màu áo"""
    def __init__(self, num_jersey_classes=12, num_color_classes=2):
        super(JerseyClassifierResNet, self).__init__()
        
        # Tải ResNet50 pretrained
        self.backbone = models.resnet50(weights=None)
        
        # Lấy số features đầu ra từ backbone
        in_features = self.backbone.fc.in_features  # 2048
        
        # Xóa lớp FC gốc
        del self.backbone.fc
        
        # Thêm 2 lớp FC cho 2 task
        self.fc_jersey = nn.Linear(in_features, num_jersey_classes)
        self.fc_color = nn.Linear(in_features, num_color_classes)
    
    def forward(self, x):
        # Trích xuất đặc trưng qua backbone
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # [batch_size, 2048]
        
        # Hai đầu ra cho hai nhiệm vụ
        jersey_logits = self.fc_jersey(x)
        color_logits = self.fc_color(x)
        
        return jersey_logits, color_logits

class Classifier:
    """Wrapper cho model classification"""
    def __init__(self, model_path, device=None):
        # Xác định device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if torch.backends.mps.is_available():
                self.device = 'mps'
        else:
            self.device = device
            
        print(f"Classifier sử dụng thiết bị: {self.device}")
        
        # Tạo và load model
        self.model = JerseyClassifierResNet()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Định nghĩa transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Định nghĩa tên các lớp
        self.jersey_classes = ['0', '1', '10', '11', '2', '3', '4', '5', '6', '7', '8', '9']
        self.color_classes = ['white', 'black']
        
    def classify(self, player_crop):
        """Phân loại số áo và màu áo từ ảnh crop cầu thủ"""
        # Chuyển BGR sang RGB
        rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
        
        # Chuyển sang PIL Image và áp dụng transform
        pil_crop = Image.fromarray(rgb_crop)
        input_tensor = self.transform(pil_crop).unsqueeze(0).to(self.device)
        
        # Thực hiện inference
        with torch.no_grad():
            jersey_outputs, color_outputs = self.model(input_tensor)
            
            # Lấy class dự đoán
            _, jersey_pred = torch.max(jersey_outputs, 1)
            _, color_pred = torch.max(color_outputs, 1)
            
            jersey_number = self.jersey_classes[jersey_pred.item()]
            team_color = self.color_classes[color_pred.item()]
        
        return jersey_number, team_color