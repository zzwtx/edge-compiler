// tensorRT_profiling.cpp (简化版 - 修复了编译问题)
// 带 TensorRT 内置 Profiler 的性能分析版本

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <numeric>
#include <cstring>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include "NvInfer.h"
#include "cuda_runtime.h"

using namespace nvinfer1;

// 简易 Logger
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

// 自定义 Profiler 实现
class ProfilerCallback : public IProfiler {
public:
    struct LayerProfile {
        float time_ms;
        std::string layer_name;
        std::string layer_type;
        int execution_count;
    };

    // IProfiler::reportLayerTime provides only layer name and time (signature varies by TRT version).
    // Match the common signature: (const char* layerName, float ms)
    void reportLayerTime(const char* layerName, float ms) noexcept override {
        LayerProfile profile;
        profile.layer_name = layerName ? layerName : "";
        profile.time_ms = ms;
        profile.execution_count = 1;
        profile.layer_type = "Unknown"; // LayerType is not provided in this callback signature
        
        profiles.push_back(profile);
    }

    void printLayerProfile() const {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "              TensorRT Layer Profiling Results" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // 计算总时间
        float total_time = 0.0f;
        for (const auto& p : profiles) {
            total_time += p.time_ms;
        }
        
        // 按时间排序
        std::vector<LayerProfile> sorted_profiles = profiles;
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                  [](const LayerProfile& a, const LayerProfile& b) {
                      return a.time_ms > b.time_ms;
                  });
        
        // 打印表头
        std::cout << std::left 
                  << std::setw(40) << "Layer Name"
                  << std::setw(20) << "Type"
                  << std::setw(15) << "Time (ms)"
                  << std::setw(10) << "% of Total"
                  << std::endl;
        std::cout << std::string(80, '-') << std::endl;
        
        // 打印每层数据
        for (size_t i = 0; i < sorted_profiles.size(); ++i) {
            const auto& p = sorted_profiles[i];
            float percentage = (total_time > 0) ? (p.time_ms / total_time * 100.0f) : 0.0f;
            
            std::cout << std::left 
                      << std::setw(40) << p.layer_name
                      << std::setw(20) << p.layer_type
                      << std::setw(15) << std::fixed << std::setprecision(4) << p.time_ms
                      << std::setw(10) << std::fixed << std::setprecision(2) << percentage << "%"
                      << std::endl;
        }
        
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "Total GPU Execution Time: " << std::fixed << std::setprecision(4) 
                  << total_time << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // 找出消耗时间最多的前 5 层
        std::cout << "\n🔥 Top 5 Time-Consuming Layers (Optimization Candidates):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), sorted_profiles.size()); ++i) {
            const auto& p = sorted_profiles[i];
            float percentage = (total_time > 0) ? (p.time_ms / total_time * 100.0f) : 0.0f;
            std::cout << "  " << (i+1) << ". [" << std::fixed << std::setprecision(2) 
                      << percentage << "%] " << p.layer_name 
                      << " (" << p.layer_type << ") - " 
                      << std::fixed << std::setprecision(4) << p.time_ms << " ms" << std::endl;
        }
    }

    std::vector<LayerProfile> profiles;
};

// ===================================
// 主函数
// ===================================
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
        std::cerr << "Usage: ./tensorrt_profiling <engine_file.engine>" << std::endl;
        return 1;
    }
    std::string engineFile = argv[1];

    // 1. 反序列化引擎
    auto engineData = readFile(engineFile);
    IRuntime* runtime = createInferRuntime(gLogger);
    if (!runtime) { std::cerr << "Failed to create runtime\n"; return 1; }

    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) { std::cerr << "Failed to deserialize engine\n"; return 1; }

    IExecutionContext* context = engine->createExecutionContext();
    if (!context) { std::cerr << "Failed to create context\n"; return 1; }

    // 2. 创建并注册 Profiler
    ProfilerCallback profiler;
    context->setProfiler(&profiler);
    context->setOptimizationProfileAsync(0, nullptr);  // 使用第一个优化配置

    // 3. 准备绑定和缓冲区
    int nbBindings = engine->getNbIOTensors();
    std::vector<void*> deviceBindings(nbBindings, nullptr);
    std::vector<size_t> hostSizes(nbBindings, 0);
    std::vector<void*> hostBuffers(nbBindings, nullptr);
    std::vector<int> bindingIsInput(nbBindings, 0);

    for (int b = 0; b < nbBindings; ++b) {
        const char* tensorName = engine->getIOTensorName(b);
        Dims dims = engine->getTensorShape(tensorName);
        DataType dtype = engine->getTensorDataType(tensorName);
        TensorIOMode ioMode = engine->getTensorIOMode(tensorName);
        bool isInput = (ioMode == TensorIOMode::kINPUT);
        bindingIsInput[b] = isInput ? 1 : 0;

        size_t vol = 1;
        for (int i = 0; i < dims.nbDims; ++i) vol *= (dims.d[i] > 0 ? dims.d[i] : 1);

        size_t typeSize = (dtype == DataType::kFLOAT) ? 4 : (dtype == DataType::kHALF ? 2 : 4);
        size_t bytes = vol * typeSize;

        hostSizes[b] = bytes;
        cudaMallocHost(&hostBuffers[b], bytes);
        cudaMalloc(&deviceBindings[b], bytes);
    }

    // 4. 确定输入/输出 binding 索引
    int inputBindingIndex = -1;
    int outputBindingIndex = -1;
    for (int b = 0; b < nbBindings; ++b) {
        if (bindingIsInput[b]) {
            if (inputBindingIndex != -1) { std::cerr << "Multiple input bindings\n"; return 1; }
            inputBindingIndex = b;
        } else {
            if (outputBindingIndex != -1) { std::cerr << "Multiple output bindings\n"; return 1; }
            outputBindingIndex = b;
        }
    }
    if (inputBindingIndex == -1 || outputBindingIndex == -1) {
        std::cerr << "Could not find input or output binding\n"; return 1;
    }

    size_t single_input_bytes = hostSizes[inputBindingIndex];
    size_t single_output_bytes = hostSizes[outputBindingIndex];

    // 5. 生成随机输入
    size_t input_num_elements = single_input_bytes / sizeof(float);
    std::vector<float> dummy_input(input_num_elements);
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < input_num_elements; ++i) {
        dummy_input[i] = dist(gen);
    }

    // 6. 热身运行（消除冷启动效应）
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    std::cout << "Performing warm-up runs (10 iterations)..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        memcpy(hostBuffers[inputBindingIndex], dummy_input.data(), single_input_bytes);
        cudaMemcpyAsync(deviceBindings[inputBindingIndex], hostBuffers[inputBindingIndex], 
                       single_input_bytes, cudaMemcpyHostToDevice, stream);
        
        const char* inputTensorName = engine->getIOTensorName(inputBindingIndex);
        const char* outputTensorName = engine->getIOTensorName(outputBindingIndex);
        context->setInputTensorAddress(inputTensorName, deviceBindings[inputBindingIndex]);
        context->setOutputTensorAddress(outputTensorName, deviceBindings[outputBindingIndex]);
        
        context->enqueueV3(stream);
        
        cudaMemcpyAsync(hostBuffers[outputBindingIndex], deviceBindings[outputBindingIndex],
                       single_output_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    std::cout << "Warm-up complete." << std::endl;

    // 7. 性能分析运行（启用 Profiler）
    std::cout << "\nPerforming profiling runs (10 iterations with layer-level profiling)..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        memcpy(hostBuffers[inputBindingIndex], dummy_input.data(), single_input_bytes);
        cudaMemcpyAsync(deviceBindings[inputBindingIndex], hostBuffers[inputBindingIndex],
                       single_input_bytes, cudaMemcpyHostToDevice, stream);
        
        const char* inputTensorName = engine->getIOTensorName(inputBindingIndex);
        const char* outputTensorName = engine->getIOTensorName(outputBindingIndex);
        context->setInputTensorAddress(inputTensorName, deviceBindings[inputBindingIndex]);
        context->setOutputTensorAddress(outputTensorName, deviceBindings[outputBindingIndex]);
        
        context->enqueueV3(stream);
        
        cudaMemcpyAsync(hostBuffers[outputBindingIndex], deviceBindings[outputBindingIndex],
                       single_output_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    // 8. 输出 Profiling 结果
    profiler.printLayerProfile();

    // 清理
    cudaStreamDestroy(stream);
    for (int b = 0; b < nbBindings; ++b) {
        if (deviceBindings[b]) cudaFree(deviceBindings[b]);
        if (hostBuffers[b]) cudaFreeHost(hostBuffers[b]);
    }
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
