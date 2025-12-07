

import os
import cv2 # Cần cài đặt: pip install opencv-python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # Để lưu model
from matplotlib import colors


def extract_features_from_image(image_path, n_clusters=5):
    try:
        # Đọc ảnh bằng OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"Không đọc được ảnh: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Đổi sang RGB
        
        
        img = cv2.resize(img, (300, 300)) 
        
        
        rgb_flat = img.reshape((-1, 3))
        hsv_flat = colors.rgb_to_hsv(img / 255.0).reshape((-1, 3)) * 255


        # --- [THÊM 1]: Tính độ sáng trung bình toàn ảnh ---
        mean_brightness = np.mean(hsv_flat[:, 2]) 
        # --------------------------------------------------


        features = np.hstack((rgb_flat, hsv_flat))
        
        
        kmeans = KMeans(n_clusters=n_clusters, init="random", n_init=10, max_iter=300, random_state=42)
        kmeans.fit(features)
        
        
        centers = kmeans.cluster_centers_
        
        # Sắp xếp theo V
        centers = centers[centers[:, 5].argsort()]
        
    #    #lấy hsv train cho lẹ
    #     return centers[:, 3:6].flatten()
    
        features_knn = centers[:, 3:6].flatten()

        # --- [THÊM 2]: Chèn độ sáng vào đầu mảng ---
        return np.insert(features_knn, 0, mean_brightness)
        # -------------------------------------------


        
    except Exception as e:
        print(f"Lỗi xử lý ảnh {image_path}: {e}")
        return None



dataset_path = "day_night_images\dataset" # Đường dẫn tương đối (cùng thư mục với code)
categories = ['Night', 'Day'] # 0: Night, 1: Day, đặt tên dataset như vậy luôn để xíu xuống train cho dễ


if not os.path.exists(dataset_path):
    print(f"LỖI: Không tìm thấy thư mục '{dataset_path}'!")
    exit()

data = []
labels = []



for label_id, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    
    if not os.path.exists(folder_path):
        print(f"Không tìm thấy thư mục con '{category}'")
        continue

    # Lọc
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
   
    
    if len(image_files) == 0:
        print(f"Thư mục '{category}' trống! Hãy thêm ảnh vào.")
        continue

    for i, file_name in enumerate(image_files):
        path = os.path.join(folder_path, file_name)
        
        
        feature_vector = extract_features_from_image(path)
        
        if feature_vector is not None:
            data.append(feature_vector)
            labels.append(label_id)
    
       
        if (i + 1) % 10 == 0:
            print(f"   Đã xử lý {i + 1}/{len(image_files)} ảnh...")

print("Xử lý xong")

if len(data) == 0:
    print("LỖI: Không có dữ liệu nào để train. Vui lòng kiểm tra lại ảnh trong thư mục dataset.")
    exit()


X = np.array(data)
y = np.array(labels)

print(f"Tổng số mẫu dữ liệu: {len(X)}")

#tạo frame xuất csv 
# feature_columns = []
feature_columns = ['Mean_Brightness'] # <--- THÊM
for i in range(5): 
    feature_columns.extend([f'H{i+1}', f'S{i+1}', f'V{i+1}'])


df = pd.DataFrame(X, columns=feature_columns)


df['label'] = [categories[label] for label in y] 


csv_filename = 'day_night_features.csv'#mấy ông có thể đổi tên nha
df.to_csv(csv_filename, index=False)# chỗ index này tắt hay mở cũng được, nếu mở thì thêm cột thứ tự thôi