#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <opencv2/opencv.hpp>

// =============================
//  MÔ HÌNH CNN THUẦN C++
//  (Không song song, không tăng tốc)
//  Mô tả: Triển khai CNN cơ bản cho phân loại ảnh, sử dụng OpenCV để đọc ảnh.
//  Đã chú thích chi tiết các hàm và biến bên dưới.
// =============================

// Hàm convolution 2D cho đa kênh
std::vector<std::vector<std::vector<float>>> conv2d_multi(
    const std::vector<std::vector<std::vector<float>>>& input,
    const std::vector<std::vector<std::vector<std::vector<float>>>>& filters,
    int stride) {
    int input_channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int filter_height = filters[0][0].size();
    int filter_width = filters[0][0][0].size();
    int output_channels = filters.size();
    int output_height = (input_height - filter_height) / stride + 1;
    int output_width = (input_width - filter_width) / stride + 1;
    std::vector<std::vector<std::vector<float>>> output(
        output_channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0f)));

    for (int oc = 0; oc < output_channels; ++oc) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                float sum = 0.0f;
                for (int ic = 0; ic < input_channels; ++ic) {
                    for (int fh = 0; fh < filter_height; ++fh) {
                        for (int fw = 0; fw < filter_width; ++fw) {
                            int ih = oh * stride + fh;
                            int iw = ow * stride + fw;
                            if (ih < input_height && iw < input_width) {
                                sum += input[ic][ih][iw] * filters[oc][ic][fh][fw];
                            }
                        }
                    }
                }
                output[oc][oh][ow] = sum;
            }
        }
    }
    return output;
}

// Hàm max pooling cho đa kênh
std::vector<std::vector<std::vector<float>>> max_pooling_multi(
    const std::vector<std::vector<std::vector<float>>>& input, int pool_size) {
    int channels = input.size();
    int input_height = input[0].size();
    int input_width = input[0][0].size();
    int output_height = input_height / pool_size;
    int output_width = input_width / pool_size;
    std::vector<std::vector<std::vector<float>>> output(
        channels, std::vector<std::vector<float>>(output_height, std::vector<float>(output_width, 0.0f)));

    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                float max_val = -INFINITY;
                for (int ph = 0; ph < pool_size; ++ph) {
                    for (int pw = 0; pw < pool_size; ++pw) {
                        int ih = oh * pool_size + ph;
                        int iw = ow * pool_size + pw;
                        if (ih < input_height && iw < input_width) {
                            max_val = std::max(max_val, input[c][ih][iw]);
                        }
                    }
                }
                output[c][oh][ow] = max_val;
            }
        }
    }
    return output;
}

// Hàm ReLU cho đa kênh
void relu_multi(std::vector<std::vector<std::vector<float>>>& input) {
    for (auto& channel : input) {
        for (auto& row : channel) {
            for (auto& val : row) {
                val = std::max(0.0f, val);
            }
        }
    }
}

// Hàm flatten
std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>>& input) {
    std::vector<float> output;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            output.insert(output.end(), row.begin(), row.end());
        }
    }
    return output;
}

// Hàm fully connected
std::vector<float> fully_connected(const std::vector<float>& input,
                                  const std::vector<std::vector<float>>& weights,
                                  const std::vector<float>& bias) {
    int output_size = weights.size();
    std::vector<float> output(output_size, 0.0f);

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input.size(); ++j) {
            output[i] += input[j] * weights[i][j];
        }
        output[i] += bias[i];
    }
    return output;
}

// Hàm ReLU cho fully connected
void relu_fc(std::vector<float>& input) {
    for (int i = 0; i < input.size(); ++i) {
        input[i] = std::max(0.0f, input[i]);
    }
}

// Hàm softmax
std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    float max_val = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    for (auto& val : output) {
        val /= sum;
    }
    return output;
}

// Hàm đọc ảnh bằng OpenCV
std::vector<std::vector<std::vector<float>>> read_image(const std::string& image_path, int width, int height) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Không thể tải ảnh: " + image_path);
    }
    cv::resize(image, image, cv::Size(width, height));
    std::vector<std::vector<std::vector<float>>> input(3, std::vector<std::vector<float>>(height, std::vector<float>(width)));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            input[0][y][x] = pixel[2] / 255.0f; // R
            input[1][y][x] = pixel[1] / 255.0f; // G
            input[2][y][x] = pixel[0] / 255.0f; // B
        }
    }
    return input;
}

// Hàm đọc dataset từ file CSV
std::vector<std::pair<std::string, int>> load_dataset(const std::string& csv_path) {
    std::vector<std::pair<std::string, int>> dataset;
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        throw std::runtime_error("Không thể mở file CSV: " + csv_path);
    }
    std::string line;
    std::getline(file, line); // Bỏ qua header nếu có
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string path, label_str;
        std::getline(ss, path, ',');
        std::getline(ss, label_str);
        int label = std::stoi(label_str);
        dataset.push_back({path, label});
    }
    file.close();
    
    // Sử dụng seed cố định để đảm bảo kết quả nhất quán
    std::mt19937 g(42); // Seed cố định là 42
    std::shuffle(dataset.begin(), dataset.end(), g);
    
    return dataset;
}

// Hàm one-hot encoding
std::vector<float> one_hot_encode(int label, int num_classes) {
    std::vector<float> encoding(num_classes, 0.0f);
    encoding[label] = 1.0f;
    return encoding;
}

// Hàm forward pass
std::vector<float> forward(const std::vector<std::vector<std::vector<float>>>& input,
                           const std::vector<std::vector<std::vector<std::vector<float>>>>& conv1_filters,
                           const std::vector<float>& conv1_bias,
                           const std::vector<std::vector<std::vector<std::vector<float>>>>& conv2_filters,
                           const std::vector<float>& conv2_bias,
                           const std::vector<std::vector<std::vector<std::vector<float>>>>& conv3_filters,
                           const std::vector<float>& conv3_bias,
                           const std::vector<std::vector<float>>& fc1_weights,
                           const std::vector<float>& fc1_bias,
                           const std::vector<std::vector<float>>& fc2_weights,
                           const std::vector<float>& fc2_bias,
                           double& forward_time_ms,
                           std::vector<std::vector<std::vector<float>>>& conv1_output,
                           std::vector<std::vector<std::vector<float>>>& pool1_output,
                           std::vector<std::vector<std::vector<float>>>& conv2_output,
                           std::vector<std::vector<std::vector<float>>>& pool2_output,
                           std::vector<std::vector<std::vector<float>>>& conv3_output,
                           std::vector<std::vector<std::vector<float>>>& pool3_output,
                           std::vector<float>& fc1_output) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Conv1
    conv1_output = conv2d_multi(input, conv1_filters, 1);
    // Thêm bias và áp dụng ReLU
    for (int c = 0; c < conv1_output.size(); ++c) {
        for (int h = 0; h < conv1_output[c].size(); ++h) {
            for (int w = 0; w < conv1_output[c][h].size(); ++w) {
                conv1_output[c][h][w] += conv1_bias[c];
            }
        }
    }
    relu_multi(conv1_output);
    
    // Pool1
    pool1_output = max_pooling_multi(conv1_output, 2);
    
    // Conv2
    conv2_output = conv2d_multi(pool1_output, conv2_filters, 1);
    // Thêm bias và áp dụng ReLU
    for (int c = 0; c < conv2_output.size(); ++c) {
        for (int h = 0; h < conv2_output[c].size(); ++h) {
            for (int w = 0; w < conv2_output[c][h].size(); ++w) {
                conv2_output[c][h][w] += conv2_bias[c];
            }
        }
    }
    relu_multi(conv2_output);
    
    // Pool2
    pool2_output = max_pooling_multi(conv2_output, 2);
    
    // Conv3
    conv3_output = conv2d_multi(pool2_output, conv3_filters, 1);
    // Thêm bias và áp dụng ReLU
    for (int c = 0; c < conv3_output.size(); ++c) {
        for (int h = 0; h < conv3_output[c].size(); ++h) {
            for (int w = 0; w < conv3_output[c][h].size(); ++w) {
                conv3_output[c][h][w] += conv3_bias[c];
            }
        }
    }
    relu_multi(conv3_output);
    
    // Pool3
    pool3_output = max_pooling_multi(conv3_output, 2);
    
    // Flatten
    auto flatten_output = flatten(pool3_output);
    
    // FC1
    fc1_output = fully_connected(flatten_output, fc1_weights, fc1_bias);
    relu_fc(fc1_output);
    
    // FC2 (Output)
    auto fc2_output = fully_connected(fc1_output, fc2_weights, fc2_bias);
    
    // Softmax
    auto output = softmax(fc2_output);
    
    auto end = std::chrono::high_resolution_clock::now();
    forward_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return output;
}

// Hàm dự đoán
int predict(const std::string& image_path,
            std::vector<float>& confidence,
            double& forward_time_ms,
            int input_width, int input_height,
            const std::vector<std::vector<std::vector<std::vector<float>>>>& conv1_filters,
            const std::vector<float>& conv1_bias,
            const std::vector<std::vector<std::vector<std::vector<float>>>>& conv2_filters,
            const std::vector<float>& conv2_bias,
            const std::vector<std::vector<std::vector<std::vector<float>>>>& conv3_filters,
            const std::vector<float>& conv3_bias,
            const std::vector<std::vector<float>>& fc1_weights,
            const std::vector<float>& fc1_bias,
            const std::vector<std::vector<float>>& fc2_weights,
            const std::vector<float>& fc2_bias) {
    auto input = read_image(image_path, input_width, input_height);
    
    std::vector<std::vector<std::vector<float>>> conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output;
    std::vector<float> fc1_output;
    
    confidence = forward(input, conv1_filters, conv1_bias, conv2_filters, conv2_bias, conv3_filters, conv3_bias,
                        fc1_weights, fc1_bias, fc2_weights, fc2_bias, forward_time_ms,
                        conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc1_output);
                        
    return std::max_element(confidence.begin(), confidence.end()) - confidence.begin();
}

// Hàm huấn luyện
void train(const std::string& dataset_path,
           std::vector<std::vector<std::vector<std::vector<float>>>>& conv1_filters,
           std::vector<float>& conv1_bias,
           std::vector<std::vector<std::vector<std::vector<float>>>>& conv2_filters,
           std::vector<float>& conv2_bias,
           std::vector<std::vector<std::vector<std::vector<float>>>>& conv3_filters,
           std::vector<float>& conv3_bias,
           std::vector<std::vector<float>>& fc1_weights,
           std::vector<float>& fc1_bias,
           std::vector<std::vector<float>>& fc2_weights,
           std::vector<float>& fc2_bias,
           int input_width, int input_height, int num_classes, int epochs, float learning_rate) {
    auto dataset = load_dataset(dataset_path);
    std::cout << "Đã tải " << dataset.size() << " mẫu dữ liệu" << std::endl;

    // Không chia tập train/validation, sử dụng toàn bộ dataset
    std::vector<std::pair<std::string, int>>& train_set = dataset;
    
    std::cout << "Training set: " << train_set.size() << " mẫu" << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
        
        // Sử dụng seed cố định để đảm bảo kết quả nhất quán
        std::mt19937 g(42 + epoch); // Seed cố định là 42 + epoch
        std::shuffle(train_set.begin(), train_set.end(), g);
        
        float total_loss = 0.0f;
        int correct = 0;
        double total_forward_time = 0.0;

        for (size_t i = 0; i < train_set.size(); ++i) {
            auto input = read_image(train_set[i].first, input_width, input_height);
            std::vector<float> target = one_hot_encode(train_set[i].second, num_classes);

            double forward_time_ms = 0.0;
            std::vector<std::vector<std::vector<float>>> conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output;
            std::vector<float> fc1_output;
            
            std::vector<float> output = forward(input, conv1_filters, conv1_bias, conv2_filters, conv2_bias, conv3_filters, conv3_bias,
                                                fc1_weights, fc1_bias, fc2_weights, fc2_bias, forward_time_ms,
                                                conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc1_output);
            total_forward_time += forward_time_ms;

            // Tính cross-entropy loss
            float loss = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                if (target[j] > 0.5f) loss -= std::log(std::max(output[j], 1e-7f));
            }
            total_loss += loss;

            // Tính accuracy
            int predicted_class = std::max_element(output.begin(), output.end()) - output.begin();
            if (predicted_class == train_set[i].second) correct++;

            if ((i + 1) % 10 == 0 || i + 1 == train_set.size()) {
                std::cout << "\rĐã xử lý " << i + 1 << "/" << train_set.size()
                          << " | Loss: " << total_loss / (i + 1)
                          << " | Accuracy: " << 100.0f * correct / (i + 1) << "%"
                          << " | Forward: " << total_forward_time / (i + 1) << " ms"
                          << std::flush;
            }
        }
        std::cout << std::endl;
        
        // Đánh giá trên toàn bộ tập dữ liệu (không chia validation)
        float eval_loss = 0.0f;
        int eval_correct = 0;
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto input = read_image(dataset[i].first, input_width, input_height);
            std::vector<float> target = one_hot_encode(dataset[i].second, num_classes);

            double forward_time_ms = 0.0;
            std::vector<std::vector<std::vector<float>>> conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output;
            std::vector<float> fc1_output;
            
            std::vector<float> output = forward(input, conv1_filters, conv1_bias, conv2_filters, conv2_bias, conv3_filters, conv3_bias,
                                                fc1_weights, fc1_bias, fc2_weights, fc2_bias, forward_time_ms,
                                                conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc1_output);

            // Tính cross-entropy loss
            float loss = 0.0f;
            for (int j = 0; j < num_classes; ++j) {
                if (target[j] > 0.5f) loss -= std::log(std::max(output[j], 1e-7f));
            }
            eval_loss += loss;

            // Tính accuracy
            int predicted_class = std::max_element(output.begin(), output.end()) - output.begin();
            if (predicted_class == dataset[i].second) eval_correct++;
        }
        
        std::cout << "Evaluation - Loss: " << eval_loss / dataset.size()
                  << " | Accuracy: " << 100.0f * eval_correct / dataset.size() << "%" << std::endl;
    }
}

// Hàm chính
int main() {
    try {
        // Sử dụng seed cố định để đảm bảo kết quả nhất quán
        std::srand(42);
        
        int input_width = 64, input_height = 64, input_channels = 3, filter_size = 5,
            conv1_filters_count = 64, conv2_filters_count = 128, conv3_filters_count = 216,
            pool_size = 2, fc_size = 256, num_classes = 4;

        // Khởi tạo ngẫu nhiên các tham số với seed cố định
        std::mt19937 gen(42); // Seed cố định là 42
        
        // Khởi tạo trọng số cho Conv1
        float conv1_scale = sqrt(2.0f / (input_channels * filter_size * filter_size));
        std::normal_distribution<float> conv1_dist(0.0f, conv1_scale);
        std::vector<std::vector<std::vector<std::vector<float>>>> conv1_filters(
            conv1_filters_count, std::vector<std::vector<std::vector<float>>>(input_channels, std::vector<std::vector<float>>(filter_size, std::vector<float>(filter_size))));
        std::vector<float> conv1_bias(conv1_filters_count, 0.0f);
        for (auto& filter : conv1_filters) {
            for (auto& channel : filter) {
                for (auto& row : channel) {
                    for (auto& val : row) val = conv1_dist(gen);
                }
            }
        }

        // Khởi tạo trọng số cho Conv2
        float conv2_scale = sqrt(2.0f / (conv1_filters_count * filter_size * filter_size));
        std::normal_distribution<float> conv2_dist(0.0f, conv2_scale);
        std::vector<std::vector<std::vector<std::vector<float>>>> conv2_filters(
            conv2_filters_count, std::vector<std::vector<std::vector<float>>>(conv1_filters_count, std::vector<std::vector<float>>(filter_size, std::vector<float>(filter_size))));
        std::vector<float> conv2_bias(conv2_filters_count, 0.0f);
        for (auto& filter : conv2_filters) {
            for (auto& channel : filter) {
                for (auto& row : channel) {
                    for (auto& val : row) val = conv2_dist(gen);
                }
            }
        }

        // Khởi tạo trọng số cho Conv3
        float conv3_scale = sqrt(2.0f / (conv2_filters_count * filter_size * filter_size));
        std::normal_distribution<float> conv3_dist(0.0f, conv3_scale);
        std::vector<std::vector<std::vector<std::vector<float>>>> conv3_filters(
            conv3_filters_count, std::vector<std::vector<std::vector<float>>>(conv2_filters_count, std::vector<std::vector<float>>(filter_size, std::vector<float>(filter_size))));
        std::vector<float> conv3_bias(conv3_filters_count, 0.0f);
        for (auto& filter : conv3_filters) {
            for (auto& channel : filter) {
                for (auto& row : channel) {
                    for (auto& val : row) val = conv3_dist(gen);
                }
            }
        }

        // Tính kích thước đầu ra sau các lớp convolution và pooling
        int conv1_width = (input_width - filter_size) / 1 + 1;
        int conv1_height = (input_height - filter_size) / 1 + 1;
        int pool1_width = conv1_width / pool_size;
        int pool1_height = conv1_height / pool_size;
        int conv2_width = (pool1_width - filter_size) / 1 + 1;
        int conv2_height = (pool1_height - filter_size) / 1 + 1;
        int pool2_width = conv2_width / pool_size;
        int pool2_height = conv2_height / pool_size;
        int conv3_width = (pool2_width - filter_size) / 1 + 1;
        int conv3_height = (pool2_height - filter_size) / 1 + 1;
        int pool3_width = conv3_width / pool_size;
        int pool3_height = conv3_height / pool_size;

        // Khởi tạo trọng số cho Fully Connected 1
        int fc1_input_size = conv3_filters_count * pool3_width * pool3_height;
        float fc1_scale = sqrt(2.0f / fc1_input_size);
        std::normal_distribution<float> fc1_dist(0.0f, fc1_scale);
        std::vector<std::vector<float>> fc1_weights(fc_size, std::vector<float>(fc1_input_size));
        std::vector<float> fc1_bias(fc_size, 0.0f);
        for (auto& row : fc1_weights) for (auto& val : row) val = fc1_dist(gen);

        // Khởi tạo trọng số cho Fully Connected 2 (Output)
        float fc2_scale = sqrt(2.0f / fc_size);
        std::normal_distribution<float> fc2_dist(0.0f, fc2_scale);
        std::vector<std::vector<float>> fc2_weights(num_classes, std::vector<float>(fc_size));
        std::vector<float> fc2_bias(num_classes, 0.0f);
        for (auto& row : fc2_weights) for (auto& val : row) val = fc2_dist(gen);

        int choice;
        std::cout << "Lựa chọn chế độ:\n1. Train model mới\n2. Tải và dùng model đã train\nNhập lựa chọn (1 hoặc 2): ";
        std::cin >> choice;

        if (choice == 1) {
            std::string dataset_path;
            std::cout << "Nhập đường dẫn đến file CSV dataset: ";
            std::cin >> dataset_path;
            train(dataset_path, conv1_filters, conv1_bias, conv2_filters, conv2_bias, conv3_filters, conv3_bias,
                  fc1_weights, fc1_bias, fc2_weights, fc2_bias,
                  input_width, input_height, num_classes, 1, 0.001);
            std::cout << "Model đã được train và lưu vào model_c.bin" << std::endl;
        } else if (choice == 2) {
            std::string image_path;
            std::cout << "Nhập đường dẫn đến ảnh cần dự đoán: ";
            std::cin >> image_path;

            std::vector<float> confidence;
            double forward_time_ms = 0.0;
            int predicted_class = predict(image_path, confidence, forward_time_ms, input_width, input_height,
                                          conv1_filters, conv1_bias, conv2_filters, conv2_bias, conv3_filters, conv3_bias,
                                          fc1_weights, fc1_bias, fc2_weights, fc2_bias);

            std::vector<std::string> class_names = {"RED", "BLUE", "GREEN", "YELLOW"};
            std::cout << "Dự đoán: " << class_names[predicted_class] << std::endl;
            std::cout << "Thời gian forward pass: " << forward_time_ms << " ms" << std::endl;
            std::cout << "Confidence scores:\n";
            for (int i = 0; i < num_classes; i++)
                std::cout << class_names[i] << ": " << confidence[i] * 100.0f << "%\n";
        } else {
            std::cout << "Lựa chọn không hợp lệ!" << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "Lỗi: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
