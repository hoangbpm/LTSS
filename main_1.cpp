#define CL_HPP_TARGET_OPENCL_VERSION 200
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// Kernel OpenCL cho CNN với 3 lớp convolution
const std::string kernel_source = R"(
    __kernel void conv2d(__global const float* input,
                         __global const float* filter,
                         __global float* output,
                         __local float* local_filter,
                         const int input_width,
                         const int input_height,
                         const int input_channels,
                         const int filter_size,
                         const int output_channels,
                         const int stride) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int z = get_global_id(2);
        int lx = get_local_id(0);
        int ly = get_local_id(1);

        int output_width = (input_width - filter_size) / stride + 1;
        int output_height = (input_height - filter_size) / stride + 1;

        if (x >= output_width || y >= output_height || z >= output_channels) return;

        // Sao chép filter từ global memory sang local memory
        int filter_per_channel = filter_size * filter_size;
        for (int c = 0; c < input_channels; c++) {
            int filter_offset = (z * input_channels + c) * filter_per_channel;
            if (lx < filter_size && ly < filter_size) {
                local_filter[c * filter_per_channel + ly * filter_size + lx] = filter[filter_offset + ly * filter_size + lx];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        float sum = 0.0f;
        for (int c = 0; c < input_channels; c++) {
            for (int fy = 0; fy < filter_size; fy++) {
                for (int fx = 0; fx < filter_size; fx++) {
                    int in_x = x * stride + fx;
                    int in_y = y * stride + fy;
                    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                        float input_value = input[(c * input_height + in_y) * input_width + in_x];
                        float filter_value = local_filter[c * filter_per_channel + fy * filter_size + fx];
                        sum += input_value * filter_value;
                    }
                }
            }
        }
        sum += filter[output_channels * input_channels * filter_per_channel + z]; // Bias
        output[(z * output_height + y) * output_width + x] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    __kernel void max_pooling(__global const float* input,
                             __global float* output,
                             const int input_width,
                             const int input_height,
                             const int channels,
                             const int pool_size) {
        int x = get_global_id(0);
        int y = get_global_id(1);
        int c = get_global_id(2);

        int output_width = input_width / pool_size;
        int output_height = input_height / pool_size;

        if (x >= output_width || y >= output_height || c >= channels) return;

        float max_val = -FLT_MAX;
        for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
                int in_x = x * pool_size + px;
                int in_y = y * pool_size + py;
                float val = input[(c * input_height + in_y) * input_width + in_x];
                if (val > max_val) max_val = val;
            }
        }
        output[(c * output_height + y) * output_width + x] = max_val;
    }

    __kernel void fully_connected(__global const float* input,
                                 __global const float* weights,
                                 __global float* output,
                                 const int input_size,
                                 const int output_size) {
        int out_idx = get_global_id(0);
        if (out_idx >= output_size) return;
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++) {
            sum += input[i] * weights[out_idx * input_size + i];
        }
        sum += weights[output_size * input_size + out_idx]; // Bias
        output[out_idx] = sum > 0.0f ? sum : 0.0f; // ReLU
    }

    __kernel void softmax(__global const float* input,
                         __global float* output,
                         const int size) {
        float max_val = -FLT_MAX;
        for (int i = 0; i < size; i++) {
            if (input[i] > max_val) max_val = input[i];
        }
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += exp(input[i] - max_val);
        }
        for (int i = 0; i < size; i++) {
            output[i] = exp(input[i] - max_val) / sum;
        }
    }

    __kernel void cross_entropy_gradient(__global const float* predictions,
                                        __global const float* targets,
                                        __global float* gradient,
                                        const int size) {
        int idx = get_global_id(0);
        if (idx >= size) return;
        gradient[idx] = predictions[idx] - targets[idx];
    }

    __kernel void fc_gradient(__global const float* input,
                             __global const float* weights,
                             __global const float* output_gradient,
                             __global float* input_gradient,
                             __global float* weight_gradient,
                             const int input_size,
                             const int output_size) {
        int idx = get_global_id(0);
        if (idx < input_size) {
            float sum = 0.0f;
            for (int o = 0; o < output_size; o++) {
                float relu_grad = output_gradient[o] > 0.0f ? 1.0f : 0.0f;
                sum += weights[o * input_size + idx] * output_gradient[o] * relu_grad;
            }
            input_gradient[idx] = sum;
        } else if (idx < input_size + output_size * input_size) {
            int w_idx = idx - input_size;
            int o = w_idx / input_size;
            int i = w_idx % input_size;
            float relu_grad = output_gradient[o] > 0.0f ? 1.0f : 0.0f;
            weight_gradient[w_idx] = input[i] * output_gradient[o] * relu_grad;
        } else if (idx < input_size + output_size * input_size + output_size) {
            int b_idx = idx - input_size - output_size * input_size;
            float relu_grad = output_gradient[b_idx] > 0.0f ? 1.0f : 0.0f;
            weight_gradient[output_size * input_size + b_idx] = output_gradient[b_idx] * relu_grad;
        }
    }

    __kernel void sgd_update(__global float* weights,
                            __global const float* gradient,
                            const float learning_rate,
                            const int size) {
        int idx = get_global_id(0);
        if (idx >= size) return;
        weights[idx] -= learning_rate * gradient[idx];
    }
)";

// Định nghĩa cấu trúc mô hình CNN với 3 lớp convolution
struct CNNModel {
    int input_width, input_height, input_channels, filter_size, conv1_filters, conv2_filters, conv3_filters, pool_size, fc_size, num_classes;
    std::vector<float> conv1_weights, conv2_weights, conv3_weights, fc_weights, out_weights;
    int conv1_width, conv1_height, pool1_width, pool1_height, conv2_width, conv2_height, pool2_width, pool2_height;
    int conv3_width, conv3_height, pool3_width, pool3_height;

    CNNModel(int width, int height, int channels, int fsize, int conv1_f, int conv2_f, int conv3_f, int p_size, int fc, int classes)
        : input_width(width), input_height(height), input_channels(channels), filter_size(fsize),
          conv1_filters(conv1_f), conv2_filters(conv2_f), conv3_filters(conv3_f), pool_size(p_size), fc_size(fc), num_classes(classes) {

        // Tính toán kích thước đầu ra của từng lớp
        conv1_width = (input_width - filter_size) / 1 + 1;
        conv1_height = (input_height - filter_size) / 1 + 1;
        pool1_width = conv1_width / pool_size;
        pool1_height = conv1_height / pool_size;
        conv2_width = (pool1_width - filter_size) / 1 + 1;
        conv2_height = (pool1_height - filter_size) / 1 + 1;
        pool2_width = conv2_width / pool_size;
        pool2_height = conv2_height / pool_size;
        conv3_width = (pool2_width - filter_size) / 1 + 1;
        conv3_height = (pool2_height - filter_size) / 1 + 1;
        pool3_width = conv3_width / pool_size;
        pool3_height = conv3_height / pool_size;

        std::random_device rd;
        std::mt19937 gen(rd());

        // Khởi tạo trọng số cho Conv1
        float conv1_scale = sqrt(2.0f / (input_channels * filter_size * filter_size));
        std::normal_distribution<float> conv1_dist(0.0f, conv1_scale);
        int conv1_weights_size = conv1_filters * input_channels * filter_size * filter_size + conv1_filters;
        conv1_weights.resize(conv1_weights_size);
        for (int i = 0; i < conv1_weights_size - conv1_filters; i++) conv1_weights[i] = conv1_dist(gen);
        for (int i = conv1_weights_size - conv1_filters; i < conv1_weights_size; i++) conv1_weights[i] = 0.0f;

        // Khởi tạo trọng số cho Conv2
        float conv2_scale = sqrt(2.0f / (conv1_filters * filter_size * filter_size));
        std::normal_distribution<float> conv2_dist(0.0f, conv2_scale);
        int conv2_weights_size = conv2_filters * conv1_filters * filter_size * filter_size + conv2_filters;
        conv2_weights.resize(conv2_weights_size);
        for (int i = 0; i < conv2_weights_size - conv2_filters; i++) conv2_weights[i] = conv2_dist(gen);
        for (int i = conv2_weights_size - conv2_filters; i < conv2_weights_size; i++) conv2_weights[i] = 0.0f;

        // Khởi tạo trọng số cho Conv3
        float conv3_scale = sqrt(2.0f / (conv2_filters * filter_size * filter_size));
        std::normal_distribution<float> conv3_dist(0.0f, conv3_scale);
        int conv3_weights_size = conv3_filters * conv2_filters * filter_size * filter_size + conv3_filters;
        conv3_weights.resize(conv3_weights_size);
        for (int i = 0; i < conv3_weights_size - conv3_filters; i++) conv3_weights[i] = conv3_dist(gen);
        for (int i = conv3_weights_size - conv3_filters; i < conv3_weights_size; i++) conv3_weights[i] = 0.0f;

        // Khởi tạo trọng số cho Fully Connected
        int fc_input_size = conv3_filters * pool3_width * pool3_height;
        float fc_scale = sqrt(2.0f / fc_input_size);
        std::normal_distribution<float> fc_dist(0.0f, fc_scale);
        int fc_weights_size = fc_size * fc_input_size + fc_size;
        fc_weights.resize(fc_weights_size);
        for (int i = 0; i < fc_weights_size - fc_size; i++) fc_weights[i] = fc_dist(gen);
        for (int i = fc_weights_size - fc_size; i < fc_weights_size; i++) fc_weights[i] = 0.0f;

        // Khởi tạo trọng số cho Output Layer
        float out_scale = sqrt(2.0f / fc_size);
        std::normal_distribution<float> out_dist(0.0f, out_scale);
        int out_weights_size = num_classes * fc_size + num_classes;
        out_weights.resize(out_weights_size);
        for (int i = 0; i < out_weights_size - num_classes; i++) out_weights[i] = out_dist(gen);
        for (int i = out_weights_size - num_classes; i < out_weights_size; i++) out_weights[i] = 0.0f;
    }

    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        file.write(reinterpret_cast<char*>(&input_width), sizeof(input_width));
        file.write(reinterpret_cast<char*>(&input_height), sizeof(input_height));
        file.write(reinterpret_cast<char*>(&input_channels), sizeof(input_channels));
        file.write(reinterpret_cast<char*>(&filter_size), sizeof(filter_size));
        file.write(reinterpret_cast<char*>(&conv1_filters), sizeof(conv1_filters));
        file.write(reinterpret_cast<char*>(&conv2_filters), sizeof(conv2_filters));
        file.write(reinterpret_cast<char*>(&conv3_filters), sizeof(conv3_filters));
        file.write(reinterpret_cast<char*>(&pool_size), sizeof(pool_size));
        file.write(reinterpret_cast<char*>(&fc_size), sizeof(fc_size));
        file.write(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));

        int conv1_size = conv1_weights.size();
        file.write(reinterpret_cast<char*>(&conv1_size), sizeof(conv1_size));
        file.write(reinterpret_cast<char*>(conv1_weights.data()), conv1_size * sizeof(float));

        int conv2_size = conv2_weights.size();
        file.write(reinterpret_cast<char*>(&conv2_size), sizeof(conv2_size));
        file.write(reinterpret_cast<char*>(conv2_weights.data()), conv2_size * sizeof(float));

        int conv3_size = conv3_weights.size();
        file.write(reinterpret_cast<char*>(&conv3_size), sizeof(conv3_size));
        file.write(reinterpret_cast<char*>(conv3_weights.data()), conv3_size * sizeof(float));

        int fc_size_bytes = fc_weights.size();
        file.write(reinterpret_cast<char*>(&fc_size_bytes), sizeof(fc_size_bytes));
        file.write(reinterpret_cast<char*>(fc_weights.data()), fc_size_bytes * sizeof(float));

        int out_size = out_weights.size();
        file.write(reinterpret_cast<char*>(&out_size), sizeof(out_size));
        file.write(reinterpret_cast<char*>(out_weights.data()), out_size * sizeof(float));
        file.close();
        std::cout << "Đã lưu model vào " << filename << std::endl;
    }

    static CNNModel load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Không thể mở file: " + filename);

        int input_width, input_height, input_channels, filter_size, conv1_filters, conv2_filters, conv3_filters, pool_size, fc_size, num_classes;
        file.read(reinterpret_cast<char*>(&input_width), sizeof(input_width));
        file.read(reinterpret_cast<char*>(&input_height), sizeof(input_height));
        file.read(reinterpret_cast<char*>(&input_channels), sizeof(input_channels));
        file.read(reinterpret_cast<char*>(&filter_size), sizeof(filter_size));
        file.read(reinterpret_cast<char*>(&conv1_filters), sizeof(conv1_filters));
        file.read(reinterpret_cast<char*>(&conv2_filters), sizeof(conv2_filters));
        file.read(reinterpret_cast<char*>(&conv3_filters), sizeof(conv3_filters));
        file.read(reinterpret_cast<char*>(&pool_size), sizeof(pool_size));
        file.read(reinterpret_cast<char*>(&fc_size), sizeof(fc_size));
        file.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));

        CNNModel model(input_width, input_height, input_channels, filter_size, conv1_filters, conv2_filters, conv3_filters, pool_size, fc_size, num_classes);

        int conv1_size, conv2_size, conv3_size, fc_size_bytes, out_size;
        file.read(reinterpret_cast<char*>(&conv1_size), sizeof(conv1_size));
        model.conv1_weights.resize(conv1_size);
        file.read(reinterpret_cast<char*>(model.conv1_weights.data()), conv1_size * sizeof(float));

        file.read(reinterpret_cast<char*>(&conv2_size), sizeof(conv2_size));
        model.conv2_weights.resize(conv2_size);
        file.read(reinterpret_cast<char*>(model.conv2_weights.data()), conv2_size * sizeof(float));

        file.read(reinterpret_cast<char*>(&conv3_size), sizeof(conv3_size));
        model.conv3_weights.resize(conv3_size);
        file.read(reinterpret_cast<char*>(model.conv3_weights.data()), conv3_size * sizeof(float));

        file.read(reinterpret_cast<char*>(&fc_size_bytes), sizeof(fc_size_bytes));
        model.fc_weights.resize(fc_size_bytes);
        file.read(reinterpret_cast<char*>(model.fc_weights.data()), fc_size_bytes * sizeof(float));

        file.read(reinterpret_cast<char*>(&out_size), sizeof(out_size));
        model.out_weights.resize(out_size);
        file.read(reinterpret_cast<char*>(model.out_weights.data()), out_size * sizeof(float));
        file.close();
        std::cout << "Đã tải model từ " << filename << std::endl;
        return model;
    }
};

// Class để train và test model CNN
class CNNTrainer {
private:
    CNNModel model;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;
    cl::Device device;
    cl::Kernel conv2d_kernel, max_pooling_kernel, fc_kernel, softmax_kernel, cross_entropy_gradient_kernel, fc_gradient_kernel, sgd_update_kernel;
    float learning_rate;
    int batch_size, epochs;

    std::vector<float> preprocess_image(const cv::Mat& image) {
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(model.input_width, model.input_height));
        std::vector<float> input_data(model.input_width * model.input_height * model.input_channels);

        if (model.input_channels == 1) {
            cv::Mat gray = resized.channels() == 3 ? cv::Mat() : resized;
            if (resized.channels() == 3) cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
            for (int y = 0; y < model.input_height; y++)
                for (int x = 0; x < model.input_width; x++)
                    input_data[y * model.input_width + x] = gray.at<uchar>(y, x) / 255.0f;
        } else if (model.input_channels == 3) {
            cv::Mat rgb = resized.channels() == 1 ? cv::Mat() : resized;
            if (resized.channels() == 1) cv::cvtColor(resized, rgb, cv::COLOR_GRAY2BGR);
            for (int y = 0; y < model.input_height; y++)
                for (int x = 0; x < model.input_width; x++) {
                    cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
                    input_data[(0 * model.input_height + y) * model.input_width + x] = pixel[2] / 255.0f; // R
                    input_data[(1 * model.input_height + y) * model.input_width + x] = pixel[1] / 255.0f; // G
                    input_data[(2 * model.input_height + y) * model.input_width + x] = pixel[0] / 255.0f; // B
                }
        }
        return input_data;
    }

    std::vector<float> one_hot_encode(int label, int num_classes) {
        std::vector<float> encoding(num_classes, 0.0f);
        if (label >= 0 && label < num_classes) encoding[label] = 1.0f;
        return encoding;
    }

    void profile_kernel(cl::Event& event, const std::string& kernel_name, size_t global_work_size) {
        event.wait();
        cl_int err = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
        if (err != CL_COMPLETE) {
            std::cerr << kernel_name << " - Lỗi thực thi kernel: " << err << std::endl;
            return;
        }
        cl_ulong time_start, time_end;
        event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
        event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end);
        double time_ms = (time_end - time_start) / 1000000.0;
        // std::cout << kernel_name << " - Thời gian thực thi: " << time_ms << " ms | Global Work-items: " << global_work_size << std::endl;
    }

    std::vector<float> forward(const std::vector<float>& input_data,
                              std::vector<float>& conv1_output,
                              std::vector<float>& pool1_output,
                              std::vector<float>& conv2_output,
                              std::vector<float>& pool2_output,
                              std::vector<float>& conv3_output,
                              std::vector<float>& pool3_output,
                              std::vector<float>& fc_output,
                              double& forward_time_ms) {
        int conv1_output_size = model.conv1_filters * model.conv1_width * model.conv1_height;
        int pool1_output_size = model.conv1_filters * model.pool1_width * model.pool1_height;
        int conv2_output_size = model.conv2_filters * model.conv2_width * model.conv2_height;
        int pool2_output_size = model.conv2_filters * model.pool2_width * model.pool2_height;
        int conv3_output_size = model.conv3_filters * model.conv3_width * model.conv3_height;
        int pool3_output_size = model.conv3_filters * model.pool3_width * model.pool3_height;
        int fc_output_size = model.fc_size;
        int output_size = model.num_classes;

        conv1_output.resize(conv1_output_size);
        pool1_output.resize(pool1_output_size);
        conv2_output.resize(conv2_output_size);
        pool2_output.resize(pool2_output_size);
        conv3_output.resize(conv3_output_size);
        pool3_output.resize(pool3_output_size);
        fc_output.resize(fc_output_size);
        std::vector<float> output(output_size);

        auto start = std::chrono::high_resolution_clock::now();

        cl::Buffer d_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_data.size(), (void*)input_data.data());
        cl::Buffer d_conv1_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.conv1_weights.size(), (void*)model.conv1_weights.data());
        cl::Buffer d_conv1_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * conv1_output_size);
        cl::Buffer d_pool1_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * pool1_output_size);
        cl::Buffer d_conv2_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.conv2_weights.size(), (void*)model.conv2_weights.data());
        cl::Buffer d_conv2_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * conv2_output_size);
        cl::Buffer d_pool2_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * pool2_output_size);
        cl::Buffer d_conv3_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.conv3_weights.size(), (void*)model.conv3_weights.data());
        cl::Buffer d_conv3_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * conv3_output_size);
        cl::Buffer d_pool3_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * pool3_output_size);
        cl::Buffer d_fc_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.fc_weights.size(), (void*)model.fc_weights.data());
        cl::Buffer d_fc_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * fc_output_size);
        cl::Buffer d_out_weights(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.out_weights.size(), (void*)model.out_weights.data());
        cl::Buffer d_output(context, CL_MEM_WRITE_ONLY, sizeof(float) * output_size);

        cl::Event event;

        // Convolution 1
        conv2d_kernel.setArg(0, d_input);
        conv2d_kernel.setArg(1, d_conv1_weights);
        conv2d_kernel.setArg(2, d_conv1_output);
        conv2d_kernel.setArg(3, cl::Local(sizeof(float) * model.input_channels * model.filter_size * model.filter_size));
        conv2d_kernel.setArg(4, model.input_width);
        conv2d_kernel.setArg(5, model.input_height);
        conv2d_kernel.setArg(6, model.input_channels);
        conv2d_kernel.setArg(7, model.filter_size);
        conv2d_kernel.setArg(8, model.conv1_filters);
        conv2d_kernel.setArg(9, 1);
        queue.enqueueNDRangeKernel(conv2d_kernel, cl::NullRange, cl::NDRange(model.conv1_width, model.conv1_height, model.conv1_filters), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Convolution 1", model.conv1_width * model.conv1_height * model.conv1_filters);

        // Max Pooling 1
        max_pooling_kernel.setArg(0, d_conv1_output);
        max_pooling_kernel.setArg(1, d_pool1_output);
        max_pooling_kernel.setArg(2, model.conv1_width);
        max_pooling_kernel.setArg(3, model.conv1_height);
        max_pooling_kernel.setArg(4, model.conv1_filters);
        max_pooling_kernel.setArg(5, model.pool_size);
        queue.enqueueNDRangeKernel(max_pooling_kernel, cl::NullRange, cl::NDRange(model.pool1_width, model.pool1_height, model.conv1_filters), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Max Pooling 1", model.pool1_width * model.pool1_height * model.conv1_filters);

        // Convolution 2
        conv2d_kernel.setArg(0, d_pool1_output);
        conv2d_kernel.setArg(1, d_conv2_weights);
        conv2d_kernel.setArg(2, d_conv2_output);
        conv2d_kernel.setArg(3, cl::Local(sizeof(float) * model.conv1_filters * model.filter_size * model.filter_size));
        conv2d_kernel.setArg(4, model.pool1_width);
        conv2d_kernel.setArg(5, model.pool1_height);
        conv2d_kernel.setArg(6, model.conv1_filters);
        conv2d_kernel.setArg(7, model.filter_size);
        conv2d_kernel.setArg(8, model.conv2_filters);
        conv2d_kernel.setArg(9, 1);
        queue.enqueueNDRangeKernel(conv2d_kernel, cl::NullRange, cl::NDRange(model.conv2_width, model.conv2_height, model.conv2_filters), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Convolution 2", model.conv2_width * model.conv2_height * model.conv2_filters);

        // Max Pooling 2
        max_pooling_kernel.setArg(0, d_conv2_output);
        max_pooling_kernel.setArg(1, d_pool2_output);
        max_pooling_kernel.setArg(2, model.conv2_width);
        max_pooling_kernel.setArg(3, model.conv2_height);
        max_pooling_kernel.setArg(4, model.conv2_filters);
        max_pooling_kernel.setArg(5, model.pool_size);
        queue.enqueueNDRangeKernel(max_pooling_kernel, cl::NullRange, cl::NDRange(model.pool2_width, model.pool2_height, model.conv2_filters), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Max Pooling 2", model.pool2_width * model.pool2_height * model.conv2_filters);

        // Convolution 3
        conv2d_kernel.setArg(0, d_pool2_output);
        conv2d_kernel.setArg(1, d_conv3_weights);
        conv2d_kernel.setArg(2, d_conv3_output);
        conv2d_kernel.setArg(3, cl::Local(sizeof(float) * model.conv2_filters * model.filter_size * model.filter_size));
        conv2d_kernel.setArg(4, model.pool2_width);
        conv2d_kernel.setArg(5, model.pool2_height);
        conv2d_kernel.setArg(6, model.conv2_filters);
        conv2d_kernel.setArg(7, model.filter_size);
        conv2d_kernel.setArg(8, model.conv3_filters);
        conv2d_kernel.setArg(9, 1);
        queue.enqueueNDRangeKernel(conv2d_kernel, cl::NullRange, cl::NDRange(model.conv3_width, model.conv3_height, model.conv3_filters), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Convolution 3", model.conv3_width * model.conv3_height * model.conv3_filters);

        // Max Pooling 3
        max_pooling_kernel.setArg(0, d_conv3_output);
        max_pooling_kernel.setArg(1, d_pool3_output);
        max_pooling_kernel.setArg(2, model.conv3_width);
        max_pooling_kernel.setArg(3, model.conv3_height);
        max_pooling_kernel.setArg(4, model.conv3_filters);
        max_pooling_kernel.setArg(5, model.pool_size);
        queue.enqueueNDRangeKernel(max_pooling_kernel, cl::NullRange, cl::NDRange(model.pool3_width, model.pool3_height, model.conv3_filters), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Max Pooling 3", model.pool3_width * model.pool3_height * model.conv3_filters);

        // Fully Connected
        int fc_input_size = pool3_output_size;
        fc_kernel.setArg(0, d_pool3_output);
        fc_kernel.setArg(1, d_fc_weights);
        fc_kernel.setArg(2, d_fc_output);
        fc_kernel.setArg(3, fc_input_size);
        fc_kernel.setArg(4, fc_output_size);
        queue.enqueueNDRangeKernel(fc_kernel, cl::NullRange, cl::NDRange(fc_output_size), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Fully Connected", fc_output_size);

        // Output Layer
        fc_kernel.setArg(0, d_fc_output);
        fc_kernel.setArg(1, d_out_weights);
        fc_kernel.setArg(2, d_output);
        fc_kernel.setArg(3, fc_output_size);
        fc_kernel.setArg(4, output_size);
        queue.enqueueNDRangeKernel(fc_kernel, cl::NullRange, cl::NDRange(output_size), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Output Layer", output_size);

        // Softmax
        softmax_kernel.setArg(0, d_output);
        softmax_kernel.setArg(1, d_output);
        softmax_kernel.setArg(2, output_size);
        queue.enqueueNDRangeKernel(softmax_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Softmax", 1);

        queue.enqueueReadBuffer(d_conv1_output, CL_TRUE, 0, sizeof(float) * conv1_output_size, conv1_output.data());
        queue.enqueueReadBuffer(d_pool1_output, CL_TRUE, 0, sizeof(float) * pool1_output_size, pool1_output.data());
        queue.enqueueReadBuffer(d_conv2_output, CL_TRUE, 0, sizeof(float) * conv2_output_size, conv2_output.data());
        queue.enqueueReadBuffer(d_pool2_output, CL_TRUE, 0, sizeof(float) * pool2_output_size, pool2_output.data());
        queue.enqueueReadBuffer(d_conv3_output, CL_TRUE, 0, sizeof(float) * conv3_output_size, conv3_output.data());
        queue.enqueueReadBuffer(d_pool3_output, CL_TRUE, 0, sizeof(float) * pool3_output_size, pool3_output.data());
        queue.enqueueReadBuffer(d_fc_output, CL_TRUE, 0, sizeof(float) * fc_output_size, fc_output.data());
        queue.enqueueReadBuffer(d_output, CL_TRUE, 0, sizeof(float) * output_size, output.data());

        auto end = std::chrono::high_resolution_clock::now();
        forward_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        return output;
    }

    void backward(const std::vector<float>& input_data,
                 const std::vector<float>& conv1_output,
                 const std::vector<float>& pool1_output,
                 const std::vector<float>& conv2_output,
                 const std::vector<float>& pool2_output,
                 const std::vector<float>& conv3_output,
                 const std::vector<float>& pool3_output,
                 const std::vector<float>& fc_output,
                 const std::vector<float>& output,
                 const std::vector<float>& targets,
                 double& backward_time_ms) {
        int conv1_output_size = model.conv1_filters * model.conv1_width * model.conv1_height;
        int pool1_output_size = model.conv1_filters * model.pool1_width * model.pool1_height;
        int conv2_output_size = model.conv2_filters * model.conv2_width * model.conv2_height;
        int pool2_output_size = model.conv2_filters * model.pool2_width * model.pool2_height;
        int conv3_output_size = model.conv3_filters * model.conv3_width * model.conv3_height;
        int pool3_output_size = model.conv3_filters * model.pool3_width * model.pool3_height;
        int fc_output_size = model.fc_size;
        int output_size = model.num_classes;

        auto start = std::chrono::high_resolution_clock::now();

        cl::Buffer d_input(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * input_data.size(), (void*)input_data.data());
        cl::Buffer d_conv1_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * conv1_output.size(), (void*)conv1_output.data());
        cl::Buffer d_pool1_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pool1_output.size(), (void*)pool1_output.data());
        cl::Buffer d_conv2_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * conv2_output.size(), (void*)conv2_output.data());
        cl::Buffer d_pool2_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pool2_output.size(), (void*)pool2_output.data());
        cl::Buffer d_conv3_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * conv3_output.size(), (void*)conv3_output.data());
        cl::Buffer d_pool3_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * pool3_output.size(), (void*)pool3_output.data());
        cl::Buffer d_fc_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * fc_output.size(), (void*)fc_output.data());
        cl::Buffer d_output(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output.size(), (void*)output.data());
        cl::Buffer d_targets(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * targets.size(), (void*)targets.data());
        cl::Buffer d_conv1_weights(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.conv1_weights.size(), (void*)model.conv1_weights.data());
        cl::Buffer d_conv2_weights(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.conv2_weights.size(), (void*)model.conv2_weights.data());
        cl::Buffer d_conv3_weights(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.conv3_weights.size(), (void*)model.conv3_weights.data());
        cl::Buffer d_fc_weights(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.fc_weights.size(), (void*)model.fc_weights.data());
        cl::Buffer d_out_weights(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * model.out_weights.size(), (void*)model.out_weights.data());
        cl::Buffer d_output_gradient(context, CL_MEM_READ_WRITE, sizeof(float) * output_size);
        cl::Buffer d_out_weights_gradient(context, CL_MEM_READ_WRITE, sizeof(float) * model.out_weights.size());
        cl::Buffer d_fc_gradient(context, CL_MEM_READ_WRITE, sizeof(float) * fc_output_size);
        cl::Buffer d_fc_weights_gradient(context, CL_MEM_READ_WRITE, sizeof(float) * model.fc_weights.size());

        cl::Event event;

        cross_entropy_gradient_kernel.setArg(0, d_output);
        cross_entropy_gradient_kernel.setArg(1, d_targets);
        cross_entropy_gradient_kernel.setArg(2, d_output_gradient);
        cross_entropy_gradient_kernel.setArg(3, output_size);
        queue.enqueueNDRangeKernel(cross_entropy_gradient_kernel, cl::NullRange, cl::NDRange(output_size), cl::NullRange, nullptr, &event);
        profile_kernel(event, "Cross Entropy Gradient", output_size);

        fc_gradient_kernel.setArg(0, d_fc_output);
        fc_gradient_kernel.setArg(1, d_out_weights);
        fc_gradient_kernel.setArg(2, d_output_gradient);
        fc_gradient_kernel.setArg(3, d_fc_gradient);
        fc_gradient_kernel.setArg(4, d_out_weights_gradient);
        fc_gradient_kernel.setArg(5, fc_output_size);
        fc_gradient_kernel.setArg(6, output_size);
        queue.enqueueNDRangeKernel(fc_gradient_kernel, cl::NullRange, cl::NDRange(fc_output_size + model.out_weights.size()), cl::NullRange, nullptr, &event);
        profile_kernel(event, "FC Gradient (Output)", fc_output_size + model.out_weights.size());

        fc_gradient_kernel.setArg(0, d_pool3_output);
        fc_gradient_kernel.setArg(1, d_fc_weights);
        fc_gradient_kernel.setArg(2, d_fc_gradient);
        fc_gradient_kernel.setArg(3, d_fc_gradient);
        fc_gradient_kernel.setArg(4, d_fc_weights_gradient);
        fc_gradient_kernel.setArg(5, pool3_output_size);
        fc_gradient_kernel.setArg(6, fc_output_size);
        queue.enqueueNDRangeKernel(fc_gradient_kernel, cl::NullRange, cl::NDRange(pool3_output_size + model.fc_weights.size()), cl::NullRange, nullptr, &event);
        profile_kernel(event, "FC Gradient (FC)", pool3_output_size + model.fc_weights.size());

        sgd_update_kernel.setArg(0, d_out_weights);
        sgd_update_kernel.setArg(1, d_out_weights_gradient);
        sgd_update_kernel.setArg(2, learning_rate);
        sgd_update_kernel.setArg(3, (int)model.out_weights.size());
        queue.enqueueNDRangeKernel(sgd_update_kernel, cl::NullRange, cl::NDRange(model.out_weights.size()), cl::NullRange, nullptr, &event);
        profile_kernel(event, "SGD Update (Output)", model.out_weights.size());

        sgd_update_kernel.setArg(0, d_fc_weights);
        sgd_update_kernel.setArg(1, d_fc_weights_gradient);
        sgd_update_kernel.setArg(2, learning_rate);
        sgd_update_kernel.setArg(3, (int)model.fc_weights.size());
        queue.enqueueNDRangeKernel(sgd_update_kernel, cl::NullRange, cl::NDRange(model.fc_weights.size()), cl::NullRange, nullptr, &event);
        profile_kernel(event, "SGD Update (FC)", model.fc_weights.size());

        queue.enqueueReadBuffer(d_out_weights, CL_TRUE, 0, sizeof(float) * model.out_weights.size(), model.out_weights.data());
        queue.enqueueReadBuffer(d_fc_weights, CL_TRUE, 0, sizeof(float) * model.fc_weights.size(), model.fc_weights.data());

        auto end = std::chrono::high_resolution_clock::now();
        backward_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }

public:
    CNNTrainer(CNNModel& m, float lr = 0.001, int batch = 32, int e = 10)
        : model(m), learning_rate(lr), batch_size(batch), epochs(e) {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) throw std::runtime_error("Không tìm thấy nền tảng OpenCL nào");
        cl::Platform platform = platforms[0];
        std::cout << "Nền tảng OpenCL: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (devices.empty()) throw std::runtime_error("Không tìm thấy thiết bị OpenCL nào");
        device = devices[0];
        std::cout << "Thiết bị: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        context = cl::Context({device});
        queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        program = cl::Program(context, kernel_source);
        program.build({device});

        conv2d_kernel = cl::Kernel(program, "conv2d");
        max_pooling_kernel = cl::Kernel(program, "max_pooling");
        fc_kernel = cl::Kernel(program, "fully_connected");
        softmax_kernel = cl::Kernel(program, "softmax");
        cross_entropy_gradient_kernel = cl::Kernel(program, "cross_entropy_gradient");
        fc_gradient_kernel = cl::Kernel(program, "fc_gradient");
        sgd_update_kernel = cl::Kernel(program, "sgd_update");
    }

    std::vector<std::pair<std::string, int>> load_dataset(const std::string& csv_path) {
        std::vector<std::pair<std::string, int>> dataset;
        std::ifstream file(csv_path);
        std::string line;
        std::getline(file, line); // Bỏ qua header
        while (std::getline(file, line)) {
            size_t pos = line.find(',');
            if (pos != std::string::npos) {
                std::string path = line.substr(0, pos);
                int label = std::stoi(line.substr(pos + 1));
                dataset.push_back({path, label});
            }
        }
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(dataset.begin(), dataset.end(), g);
        return dataset;
    }

    void train(const std::string& dataset_path) {
        std::vector<std::pair<std::string, int>> dataset = load_dataset(dataset_path);
        std::cout << "Đã tải " << dataset.size() << " mẫu dữ liệu" << std::endl;

        size_t train_size = dataset.size() * 0.8;
        std::vector<std::pair<std::string, int>> train_set(dataset.begin(), dataset.begin() + train_size);
        std::vector<std::pair<std::string, int>> val_set(dataset.begin() + train_size, dataset.end());

        std::cout << "Training set: " << train_set.size() << " mẫu" << std::endl;
        std::cout << "Validation set: " << val_set.size() << " mẫu" << std::endl;

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl;
            std::random_device rd;
            std::mt19937 g(rd());
            std::shuffle(train_set.begin(), train_set.end(), g);

            float total_loss = 0.0f;
            int correct = 0;
            double total_forward_time = 0.0, total_backward_time = 0.0;

            for (size_t i = 0; i < train_set.size(); i += batch_size) {
                size_t batch_end = std::min(i + batch_size, train_set.size());
                for (size_t j = i; j < batch_end; j++) {
                    cv::Mat image = cv::imread(train_set[j].first, cv::IMREAD_COLOR);
                    if (image.empty()) {
                        std::cerr << "Không thể tải ảnh: " << train_set[j].first << std::endl;
                        continue;
                    }

                    std::vector<float> input_data = preprocess_image(image);
                    std::vector<float> targets = one_hot_encode(train_set[j].second, model.num_classes);

                    double forward_time_ms = 0.0;
                    std::vector<float> conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output;
                    std::vector<float> output = forward(input_data, conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output, forward_time_ms);
                    total_forward_time += forward_time_ms;

                    float loss = 0.0f;
                    for (int k = 0; k < model.num_classes; k++)
                        if (targets[k] > 0.5f) loss -= std::log(std::max(output[k], 1e-7f));
                    total_loss += loss;

                    int predicted_class = std::max_element(output.begin(), output.end()) - output.begin();
                    if (predicted_class == train_set[j].second) correct++;

                    double backward_time_ms = 0.0;
                    backward(input_data, conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output, output, targets, backward_time_ms);
                    total_backward_time += backward_time_ms;

                    if ((j + 1) % 10 == 0 || j + 1 == train_set.size()) {
                        std::cout << "\rĐã xử lý " << j + 1 << "/" << train_set.size()
                                  << " | Loss: " << total_loss / (j + 1 - i)
                                  << " | Accuracy: " << 100.0f * correct / (j + 1 - i) << "%"
                                  << " | Forward: " << total_forward_time / (j + 1 - i) << " ms"
                                  << " | Backward: " << total_backward_time / (j + 1 - i) << " ms"
                                  << std::flush;
                    }
                }
            }
            std::cout << std::endl;

            float val_loss = 0.0f;
            int val_correct = 0;
            for (size_t i = 0; i < val_set.size(); i++) {
                cv::Mat image = cv::imread(val_set[i].first, cv::IMREAD_COLOR);
                if (image.empty()) continue;
                std::vector<float> input_data = preprocess_image(image);
                std::vector<float> targets = one_hot_encode(val_set[i].second, model.num_classes);

                double forward_time_ms = 0.0;
                std::vector<float> conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output;
                std::vector<float> output = forward(input_data, conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output, forward_time_ms);

                float loss = 0.0f;
                for (int k = 0; k < model.num_classes; k++)
                    if (targets[k] > 0.5f) loss -= std::log(std::max(output[k], 1e-7f));
                val_loss += loss;

                int predicted_class = std::max_element(output.begin(), output.end()) - output.begin();
                if (predicted_class == val_set[i].second) val_correct++;
            }

            val_loss /= val_set.size();
            float val_accuracy = 100.0f * val_correct / val_set.size();
            std::cout << "Validation - Loss: " << val_loss << " | Accuracy: " << val_accuracy << "%" << std::endl;

            model.save("model_epoch_" + std::to_string(epoch + 1) + ".bin");
        }
    }

    int predict(const std::string& image_path, std::vector<float>& confidence, double& forward_time_ms) {
        cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Không thể tải ảnh: " + image_path);

        std::vector<float> input_data = preprocess_image(image);
        std::vector<float> conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output;
        std::vector<float> output = forward(input_data, conv1_output, pool1_output, conv2_output, pool2_output, conv3_output, pool3_output, fc_output, forward_time_ms);

        confidence = output;
        return std::max_element(output.begin(), output.end()) - output.begin();
    }
};

int main() {
    try {
        int input_width = 64, input_height = 64, input_channels = 3, filter_size = 5, 
            conv1_filters = 128, conv2_filters = 256, conv3_filters = 512, 
            pool_size = 2, fc_size = 256, num_classes = 4;
        CNNModel model(input_width, input_height, input_channels, filter_size, conv1_filters, conv2_filters, conv3_filters, pool_size, fc_size, num_classes);
        CNNTrainer trainer(model, 0.001, 32, 10);

        int choice;
        std::cout << "Lựa chọn chế độ:\n1. Train model mới\n2. Tải và dùng model đã train\nNhập lựa chọn (1 hoặc 2): ";
        std::cin >> choice;

        if (choice == 1) {
            std::string dataset_path;
            std::cout << "Nhập đường dẫn đến file CSV dataset: ";
            std::cin >> dataset_path;
            trainer.train(dataset_path);
            model.save("model_final.bin");
        } else if (choice == 2) {
            std::string model_path, image_path;
            std::cout << "Nhập đường dẫn đến file model (.bin): ";
            std::cin >> model_path;
            model = CNNModel::load(model_path);
            std::cout << "Nhập đường dẫn đến ảnh cần dự đoán: ";
            std::cin >> image_path;
            std::vector<float> confidence;
            double forward_time_ms = 0.0;
            int predicted_class = trainer.predict(image_path, confidence, forward_time_ms);
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