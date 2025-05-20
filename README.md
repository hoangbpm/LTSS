# LTSS - CNN Phân Loại Ảnh (C++/OpenMP/OpenCL)

## Giới thiệu
Dự án này triển khai một mô hình mạng nơ-ron tích chập (CNN) cơ bản cho bài toán phân loại ảnh, với 3 chế độ thực thi:
- **main_c.cpp**: Thuần C++ (không song song)
- **main_openmp.cpp**: Sử dụng OpenMP để song song hóa trên CPU
- **main_opencl.cpp**: Sử dụng OpenCL để tăng tốc trên GPU

Các file sử dụng OpenCV để đọc ảnh, và có thể huấn luyện hoặc dự đoán dựa trên file model đã lưu.

## Cấu trúc thư mục
- `main_c.cpp`         : Mã nguồn CNN thuần C++
- `main_openmp.cpp`    : CNN song song hóa bằng OpenMP
- `main_opencl.cpp`    : CNN tăng tốc bằng OpenCL (GPU)
- `dataset.csv`        : File CSV chứa đường dẫn ảnh và nhãn
- `run.txt`            : Hướng dẫn build và chạy chi tiết
- `library/`, `opencv/`, `opencv_contrib/`: Thư viện hỗ trợ (nếu có)

## Hướng dẫn build & chạy
Xem chi tiết trong file `run.txt` (đã có hướng dẫn build và chạy cho từng chế độ, bao gồm cả yêu cầu về OpenCV, OpenMP, OpenCL).

Ví dụ tổng quát:
```sh
# Biên dịch chế độ C++ thuần
# g++ main_c.cpp -o main_c -std=c++11 `pkg-config --cflags --libs opencv`

# Biên dịch chế độ OpenMP
# g++ main_openmp.cpp -o main_openmp -fopenmp -std=c++11 `pkg-config --cflags --libs opencv`

# Biên dịch chế độ OpenCL
# g++ main_opencl.cpp -o main_opencl -lOpenCL -std=c++11 `pkg-config --cflags --libs opencv`
```

## Dataset
- `dataset.csv` có định dạng: `duongdan_anh,label`
- Ảnh sẽ được resize về 64x64 RGB khi huấn luyện/dự đoán.

## Model
- Model sau khi train sẽ được lưu ra file nhị phân (ví dụ: `model_c.bin`, `model_openmp.bin`, `model_opencl.bin`).
- Có thể load lại model để dự đoán.
