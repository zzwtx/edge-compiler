// tensorRT_runner.cpp
// 带详细注释的最小 TensorRT C++ 运行示例。
// 目的：演示如何加载序列化的 TensorRT 引擎（.engine），
// 分配主机/设备缓冲区，运行推理，获取输出并清理资源。

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <cstring>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "cnpy.h"
#include "NvInfer.h"
#include "cuda_runtime.h"

using namespace nvinfer1;

// 简易 Logger（来源于 TensorRT 示例）
// 根据日志级别过滤信息，避免被大量调试输出淹没。
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
    // 仅打印 WARNING 及以上级别的消息
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

// 辅助函数：将整个二进制文件读入 vector<char>
static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open file: " + filename);
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./tensorRT_runner <engine_file.engine>" << std::endl;
        return 1;
    }
    std::string engineFile = argv[1];

    // -----------------------
    // 1) 反序列化引擎
    // -----------------------
    // 序列化的 engine 是由 TensorRT 生成的二进制 blob（例如 trtexec 或 Builder API）。
    // 我们把它加载到内存，然后用 Runtime 将其反序列化为 ICudaEngine。
    auto engineData = readFile(engineFile);
    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) { std::cerr << "Failed to create runtime\n"; return 1; }

    // 注意：若使用自定义插件，deserializeCudaEngine 支持可选的 PluginFactory 参数。
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) { std::cerr << "Failed to deserialize engine\n"; return 1; }

    // ExecutionContext 用于执行引擎（包含优化 profile 状态、绑定信息等）。
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) { std::cerr << "Failed to create context\n"; return 1; }

    engineData.clear(); // 释放序列化的 engine 数据（已不再需要）。

    // -----------------------
    // 2) 准备绑定和缓冲区
    // -----------------------
    int nbBindings = engine->getNbIOTensors();
    // deviceBindings 保存每个 binding 索引对应的设备指针
    std::vector<void*> deviceBindings(nbBindings, nullptr);
    // hostSizes 存储每个主机缓冲区的字节大小
    std::vector<size_t> hostSizes(nbBindings, 0);
    // hostBuffers 保存页锁定（pinned）主机内存指针（可加速主机<->设备拷贝）
    std::vector<void*> hostBuffers(nbBindings, nullptr);
    std::vector<int> bindingIsInput(nbBindings, 0);

    // 对每个 binding（输入或输出）计算缓冲大小，并分配主机与设备内存。
    for (int b = 0; b < nbBindings; ++b) {
    // 通过 tensor 的维度与数据类型可计算所需元素数量与字节数
        const char* tensorName = engine->getIOTensorName(b);
        Dims dims = engine->getTensorShape(tensorName);
        DataType dtype = engine->getTensorDataType(tensorName);
        TensorIOMode ioMode = engine->getTensorIOMode(tensorName);
        bool isInput = (ioMode == TensorIOMode::kINPUT);
        bindingIsInput[b] = isInput ? 1 : 0;

    // 计算元素总量（体积）= 各维度的乘积。若出现动态维（-1），这里保守地将其当作 1。
    // （在实际应用中应根据 optimization profile 查询并使用具体运行时尺寸）
        size_t vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) vol *= (dims.d[i] > 0 ? dims.d[i] : 1);

    // 根据数据类型确定每个元素的字节数（例如 float32/float16）
        size_t typeSize = (dtype == DataType::kFLOAT) ? 4 : (dtype == DataType::kHALF ? 2 : 4);
        size_t bytes = vol * typeSize;

        hostSizes[b] = bytes;
    // 分配页锁定主机内存（cudaMallocHost）以加速主机<->设备拷贝
        cudaMallocHost(&hostBuffers[b], bytes);
    // 为该 binding 分配设备内存
        cudaMalloc(&deviceBindings[b], bytes);
        std::cout << "Binding " << b << " - " << (isInput ? "Input" : "Output") << ", bytes=" << bytes << "\n";
    }

    // -----------------------
    // 3) 加载真实数据
    // -----------------------
    cnpy::NpyArray images_arr;
    cnpy::NpyArray labels_arr;
    std::string npz_path = "../data.npz";
    try {
        std::cout << "Attempting to load .npz file: " << npz_path << std::endl;
        cnpy::npz_t my_npz = cnpy::npz_load(npz_path);
        
        if (my_npz.find("images") == my_npz.end()) throw std::runtime_error("Failed to find 'images' array.");
        images_arr = my_npz["images"];

        if (my_npz.find("labels") == my_npz.end()) throw std::runtime_error("Failed to find 'labels' array.");
        labels_arr = my_npz["labels"];

        std::cout << "Successfully loaded 'images' and 'labels' arrays." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred while loading data: " << e.what() << std::endl;
        return 1;
    }

    // -------------------------------------------------
    // 4) 循环推理、验证并收集结果
    // -------------------------------------------------
    // 创建 CUDA stream 用于异步执行
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    long* true_labels = labels_arr.data<long>();
    float* all_images_data = images_arr.data<float>();
    int correct_predictions = 0;
    size_t num_images = labels_arr.num_vals;

    // 确定输入和输出 binding 的索引
    int inputBindingIndex = -1;
    int outputBindingIndex = -1;
    for (int b = 0; b < nbBindings; ++b) {
        if (bindingIsInput[b]) {
            if (inputBindingIndex != -1) { std::cerr << "Error: Multiple input bindings found. This code handles only one." << std::endl; return 1; }
            inputBindingIndex = b;
        } else {
            if (outputBindingIndex != -1) { std::cerr << "Error: Multiple output bindings found. This code handles only one." << std::endl; return 1; }
            outputBindingIndex = b;
        }
    }
    if (inputBindingIndex == -1 || outputBindingIndex == -1) {
        std::cerr << "Error: Could not find input or output binding." << std::endl;
        return 1;
    }

    // 计算单张图片和单个输出的字节大小
    size_t single_input_bytes = hostSizes[inputBindingIndex];
    size_t single_output_bytes = hostSizes[outputBindingIndex];
    size_t num_classes = single_output_bytes / sizeof(float);

    std::cout << "\n--- Starting Inference Loop ---" << std::endl;
    std::cout << "Total images: " << num_images << std::endl;
    std::cout << "Single image input size: " << single_input_bytes << " bytes" << std::endl;
    std::cout << "Single image output size: " << single_output_bytes << " bytes (" << num_classes << " classes)" << std::endl;

    // 主推理循环，一次处理一张图片
    for (size_t i = 0; i < num_images; ++i) {
        // a. 计算当前图片数据在内存中的指针
        float* current_image_data = all_images_data + (i * single_input_bytes / sizeof(float));

        // b. 将单张图片数据拷贝到主机缓冲区
        memcpy(hostBuffers[inputBindingIndex], current_image_data, single_input_bytes);
        
        // c. 将主机输入拷贝到设备
        cudaMemcpyAsync(deviceBindings[inputBindingIndex], hostBuffers[inputBindingIndex], single_input_bytes, cudaMemcpyHostToDevice, stream);

        // d. 设置输入和输出张量的设备地址（TensorRT 10.x 要求）
        const char* inputTensorName = engine->getIOTensorName(inputBindingIndex);
        const char* outputTensorName = engine->getIOTensorName(outputBindingIndex);
        context->setInputTensorAddress(inputTensorName, deviceBindings[inputBindingIndex]);
        context->setOutputTensorAddress(outputTensorName, deviceBindings[outputBindingIndex]);

        // e. 异步执行推理
        context->enqueueV3(stream);

        // f. 将设备输出拷回主机
        cudaMemcpyAsync(hostBuffers[outputBindingIndex], deviceBindings[outputBindingIndex], single_output_bytes, cudaMemcpyDeviceToHost, stream);

        // g. 同步 stream，确保当前图片的 H2D, D2H 和推理都已完成
        cudaStreamSynchronize(stream);

        // h. 后处理当前图片的输出
        float* out = reinterpret_cast<float*>(hostBuffers[outputBindingIndex]);
        int argmax = 0;
        float maxv = out[0];
        for (size_t j = 1; j < num_classes; ++j) {
            if (out[j] > maxv) {
                maxv = out[j];
                argmax = static_cast<int>(j);
            }
        }

        if (argmax == true_labels[i] + 1) {
            correct_predictions++;
        }

        // 打印前 10 张图片的结果用于调试
        if (i < 10) {
             std::cout << "Image " << i << ": Predicted class=" << argmax << ", True label=" << true_labels[i] + 1 << std::endl;
        }
    }

    // -----------------------
    // 5) 打印最终结果
    // -----------------------
    std::cout << "--------------------" << std::endl;
    std::cout << "Final Accuracy: " << (float)correct_predictions / num_images * 100.0f << "% (" << correct_predictions << "/" << num_images << ")" << std::endl;

    int warmUp = 50;  // 热身运行次数，确保 GPU 达到稳定状态
    int profileRuns = 100;  // 实际用于计时的推理次数

    // 为基准测试准备一个符合模型输入的随机虚拟数据，以获得更真实的性能指标
    // 注：使用随机数而非全 0，能避免 GPU 缓存优化导致的虚假高性能结果
    size_t input_num_elements = single_input_bytes / sizeof(float);
    std::vector<float> dummy_input(input_num_elements);
    
    // 使用标准正态分布生成随机数（与 Python np.random.randn() 一致）
    // ImageNet 标准归一化后的数据分布：均值≈0，方差≈1
    std::mt19937 gen(12345);  // 固定种子以保证可重复性
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input_num_elements; ++i) {
        dummy_input[i] = dist(gen);
    }

    // 收集延迟数据用于统计
    std::vector<double> latencies_ms;
    latencies_ms.reserve(profileRuns);

    // 先做 warmUp 次热身，然后做 profileRuns 次计时运行
    std::cout << "\n--- Starting Performance Profiling ---" << std::endl;
    std::cout << "Warm-up runs: " << warmUp << std::endl;
    std::cout << "Profile runs: " << profileRuns << std::endl;
    
    for (int iter = 0; iter < warmUp + profileRuns; ++iter) {
        // a. 将 dummy 数据拷贝到主机输入缓冲区
        memcpy(hostBuffers[inputBindingIndex], dummy_input.data(), single_input_bytes);
        
        // b. 将主机输入拷贝到设备
        cudaMemcpyAsync(deviceBindings[inputBindingIndex], hostBuffers[inputBindingIndex], single_input_bytes, cudaMemcpyHostToDevice, stream);

        // c. 设置输入和输出张量的设备地址（TensorRT 10.x 要求）
        const char* inputTensorName = engine->getIOTensorName(inputBindingIndex);
        const char* outputTensorName = engine->getIOTensorName(outputBindingIndex);
        context->setInputTensorAddress(inputTensorName, deviceBindings[inputBindingIndex]);
        context->setOutputTensorAddress(outputTensorName, deviceBindings[outputBindingIndex]);

        // 只在计时阶段记录时间
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // d. 异步执行推理
        context->enqueueV3(stream);

        // e. 将设备输出拷回主机
        cudaMemcpyAsync(hostBuffers[outputBindingIndex], deviceBindings[outputBindingIndex], single_output_bytes, cudaMemcpyDeviceToHost, stream);

        // f. 同步 stream，确保当前图片的 H2D, D2H 和推理都已完成
        cudaStreamSynchronize(stream);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        // 只在计时阶段计算和记录延迟
        if (iter >= warmUp) {
            double latency_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            latencies_ms.push_back(latency_ms);
        }

        // g. 后处理（仅在实际计时阶段可选地读取输出）
        if (iter >= warmUp) {
            float* out = reinterpret_cast<float*>(hostBuffers[outputBindingIndex]);
            int argmax = 0;
            float maxv = out[0];
            for (size_t j = 1; j < num_classes; ++j) {
                if (out[j] > maxv) {
                    maxv = out[j];
                    argmax = static_cast<int>(j);
                }
            }
            // 可选：记录或打印部分结果以便调试（这里保持静默以便纯粹计时）
        }
    }

    // -----------------------
    // 性能统计与输出
    // -----------------------
    if (latencies_ms.size() == profileRuns) {
        // 排序用于计算百分位数
        std::vector<double> sorted_latencies = latencies_ms;
        std::sort(sorted_latencies.begin(), sorted_latencies.end());
        
        // 计算统计数据
        double sum = 0.0;
        for (double lat : latencies_ms) sum += lat;
        double avg_latency = sum / latencies_ms.size();
        
        double median_latency = sorted_latencies[sorted_latencies.size() / 2];
        
        // 计算标准差
        double variance = 0.0;
        for (double lat : latencies_ms) {
            variance += (lat - avg_latency) * (lat - avg_latency);
        }
        double std_dev = std::sqrt(variance / latencies_ms.size());
        
        double min_latency = sorted_latencies.front();
        double max_latency = sorted_latencies.back();
        
        // P95: 95th percentile
        int p95_idx = static_cast<int>(sorted_latencies.size() * 0.95);
        double p95_latency = sorted_latencies[p95_idx];
        
        // P99: 99th percentile
        int p99_idx = static_cast<int>(sorted_latencies.size() * 0.99);
        double p99_latency = sorted_latencies[p99_idx];
        
        // 吞吐量（FPS）= 1000 / 平均延迟（ms）
        double fps = 1000.0 / avg_latency;
        
        // 打印结果
        std::cout << "\n--- Latency (ms) ---" << std::endl;
        std::cout << "Average:         " << std::fixed << std::setprecision(2) << avg_latency << " ms" << std::endl;
        std::cout << "Median:          " << std::fixed << std::setprecision(2) << median_latency << " ms" << std::endl;
        std::cout << "Std Dev:         " << std::fixed << std::setprecision(2) << std_dev << " ms" << std::endl;
        std::cout << "Min:             " << std::fixed << std::setprecision(2) << min_latency << " ms" << std::endl;
        std::cout << "Max:             " << std::fixed << std::setprecision(2) << max_latency << " ms" << std::endl;
        std::cout << "P95:             " << std::fixed << std::setprecision(2) << p95_latency << " ms" << std::endl;
        std::cout << "P99:             " << std::fixed << std::setprecision(2) << p99_latency << " ms" << std::endl;
        
        std::cout << "\n--- Throughput ---" << std::endl;
        std::cout << "FPS (Frames/Sec): " << std::fixed << std::setprecision(2) << fps << std::endl;
    }


    // 销毁 stream
    cudaStreamDestroy(stream);

    // -----------------------
    // 6) 清理资源
    // -----------------------
    for (int b = 0; b < nbBindings; ++b) {
        if (deviceBindings[b]) cudaFree(deviceBindings[b]);
        if (hostBuffers[b]) cudaFreeHost(hostBuffers[b]);
    }
    // 按创建顺序的逆序销毁 TensorRT 对象
    // 注：在此环境的 TensorRT 头文件中，这些接口提供虚析构函数（例如 ~IExecutionContext(), ~ICudaEngine(), ~IRuntime()）。
    // 早期示例中可能使用 destroy() 成员函数；但在当前头文件版本中该方法不存在。
    // 使用 delete 会调用虚析构函数，从而释放运行时分配的资源。
    delete context;
    delete engine;
    delete runtime;
    return 0;
}