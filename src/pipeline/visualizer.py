import cv2

def draw_detections(frame, detections, jersey_info=None):
    """Vẽ kết quả phát hiện và phân loại lên frame"""
    # Tạo bản sao của frame để vẽ lên
    result_frame = frame.copy()
    
    # Màu cho các class (BGR)
    colors = {
        'player': (255, 0, 0),    # Xanh dương
        'ball': (0, 0, 255)       # Đỏ
    }
    
    # Màu cho ô chứa số áo (BGR)
    jersey_box_color = (0, 255, 0)  # Xanh lá
    
    # Duyệt qua các detection
    for i, det in enumerate(detections):
        # Lấy thông tin từ detection
        bbox = det['bbox']        # [x1, y1, x2, y2]
        class_name = det['class_name']
        
        # Lấy tọa độ bbox
        x1, y1, x2, y2 = bbox
        
        # Chọn màu dựa trên class
        color = colors.get(class_name, (255, 255, 255))
        
        # Vẽ bounding box
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
        
        # Xử lý riêng cho cầu thủ
        if class_name == 'player' and jersey_info is not None and i in jersey_info:
            # Lấy thông tin số áo và màu áo
            jersey_number, team_color = jersey_info[i]
            
            # Bỏ qua cầu thủ có số áo 0 (không rõ số áo)
            # if jersey_number == "0":
            #     continue

            # Chọn màu chữ dựa trên màu áo
            text_color = (255, 255, 255) if team_color == 'white' else (0, 0, 0)
            
            # Tạo label
            label = f"{jersey_number}"
            
            font_scale = 1.5 # Kích thước font chữ
            thickness = 3  # Độ dày của chữ
            
            # Tính vị trí text và kích thước
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            # Tính toán vị trí để căn giữa theo chiều ngang
            box_width = text_size[0] + 20  # Thêm padding
            box_height = text_size[1] + 20
            
            # Tính tọa độ x để căn giữa
            box_x = max(0, x1 + (x2 - x1) // 2 - box_width // 2)
            
            # Vẽ background cho text ở phía trên bounding box
            # Vị trí y là y1 (trên bounding box) trừ đi chiều cao của text box
            top_y = max(0, y1 - box_height - 5)  # Cách phía trên bounding box 5px
            
            cv2.rectangle(result_frame, 
                         (box_x, top_y), 
                         (box_x + box_width, top_y + box_height), 
                         jersey_box_color, 
                         -1)  # -1 để fill màu
            
            # Vẽ text
            text_x = box_x + 10  # Thêm padding bên trái
            text_y = top_y + box_height - 10  # Thêm padding bên dưới
            
            cv2.putText(result_frame, 
                       label, 
                       (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, 
                       text_color, 
                       thickness)
        
        # Hiển thị text cho bóng
        elif class_name == 'ball':
            cv2.putText(result_frame, 
                      'ball', 
                      (x1, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      (255, 255, 255), 
                      2)
    
    return result_frame