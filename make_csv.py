import os
import csv
import random

def create_dataset_csv(dataset_dir, csv_path):
    # Định nghĩa các lớp và nhãn tương ứng
    class_names = {"red": 0, "green": 1, "blue": 2, "yellow": 3}
    
    # Danh sách để lưu trữ các cặp (đường dẫn, nhãn)
    dataset = []
    
    # Duyệt qua từng thư mục con
    for class_name, label in class_names.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Thư mục không tồn tại: {class_dir}")
            continue
        
        # Duyệt qua tất cả file trong thư mục con
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            # Kiểm tra xem có phải là file ảnh không
            if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                dataset.append((file_path, label))
    
    # Xáo trộn dataset
    random.shuffle(dataset)
    
    # Ghi vào file CSV
    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Ghi header
        writer.writerow(['path', 'label'])
        # Ghi từng dòng dữ liệu
        for file_path, label in dataset:
            writer.writerow([file_path, label])
    
    print(f"Đã tạo file CSV tại: {csv_path} với {len(dataset)} mẫu dữ liệu")

if __name__ == "__main__":
    dataset_dir = "dataset"  # Đường dẫn đến thư mục dataset
    csv_path = "dataset.csv"  # Đường dẫn đến file CSV sẽ tạo
    create_dataset_csv(dataset_dir, csv_path)