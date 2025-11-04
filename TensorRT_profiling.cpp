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
#include <map>
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
        float total_time_ms;
        std::string layer_name;
        std::string layer_type;
        int execution_count;
    };

    // 标志：是否在收集 profiling 数据（非预热）
    bool is_profiling = false;
    
    // 使用 map 来累加同一层的时间
    std::map<std::string, LayerProfile> layer_stats;

    // IProfiler::reportLayerTime provides only layer name and time (signature varies by TRT version).
    // Match the common signature: (const char* layerName, float ms)
    void reportLayerTime(const char* layerName, float ms) noexcept override {
        // 只在 profiling 阶段统计（跳过预热）
        if (!is_profiling) {
            return;
        }

        std::string name_str = layerName ? layerName : "";
        
        // 查找或创建该层的统计数据
        auto it = layer_stats.find(name_str);
        if (it != layer_stats.end()) {
            // 该层已存在，累加时间
            it->second.total_time_ms += ms;
            it->second.execution_count++;
        } else {
            // 首次出现该层
            LayerProfile profile;
            profile.layer_name = name_str;
            profile.total_time_ms = ms;
            profile.execution_count = 1;
            profile.layer_type = "Unknown";
            layer_stats[name_str] = profile;
        }
    }

    void printLayerProfile() const {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "              TensorRT Layer Profiling Results" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        std::cout << "Note: Times are AVERAGED over " << (layer_stats.empty() ? 0 : layer_stats.begin()->second.execution_count) 
                  << " iterations (excluding warm-up)\n" << std::endl;
        
        // 计算总时间
        float total_time = 0.0f;
        for (const auto& pair : layer_stats) {
            total_time += pair.second.total_time_ms;
        }
        
        // 转换为 vector 并按时间排序
        std::vector<std::pair<std::string, LayerProfile>> sorted_profiles;
        for (const auto& pair : layer_stats) {
            sorted_profiles.push_back(pair);
        }
        std::sort(sorted_profiles.begin(), sorted_profiles.end(),
                  [](const auto& a, const auto& b) {
                      return a.second.total_time_ms > b.second.total_time_ms;
                  });
        
        // 打印表头
        std::cout << std::left 
                  << std::setw(40) << "Layer Name"
                  << std::setw(20) << "Type"
                  << std::setw(15) << "Avg Time (ms)"
                  << std::setw(10) << "% of Total"
                  << std::setw(12) << "Exec Count"
                  << std::endl;
        std::cout << std::string(97, '-') << std::endl;
        
        // 打印每层数据（平均时间）
        for (const auto& pair : sorted_profiles) {
            const auto& p = pair.second;
            float avg_time = p.total_time_ms / p.execution_count;
            float percentage = (total_time > 0) ? (p.total_time_ms / total_time * 100.0f) : 0.0f;
            
            std::cout << std::left 
                      << std::setw(40) << p.layer_name
                      << std::setw(20) << p.layer_type
                      << std::setw(15) << std::fixed << std::setprecision(4) << avg_time
                      << std::setw(10) << std::fixed << std::setprecision(2) << percentage << "%"
                      << std::setw(12) << p.execution_count
                      << std::endl;
        }
        
        std::cout << std::string(97, '-') << std::endl;
        
        int num_iterations = sorted_profiles.empty() ? 0 : sorted_profiles[0].second.execution_count;
        float avg_total_time = (num_iterations > 0) ? (total_time / num_iterations) : 0.0f;
        
        std::cout << "Total GPU Execution Time (accumulated): " << std::fixed << std::setprecision(4) 
                  << total_time << " ms" << std::endl;
        std::cout << "Average per iteration: " << std::fixed << std::setprecision(4) 
                  << avg_total_time << " ms (" << num_iterations << " iterations profiled)" << std::endl;
        std::cout << std::string(80, '=') << std::endl;
        
        // 找出消耗时间最多的前 5 层
        std::cout << "\n🔥 Top 5 Time-Consuming Layers (Averaged, Optimization Candidates):" << std::endl;
        for (size_t i = 0; i < std::min(size_t(5), sorted_profiles.size()); ++i) {
            const auto& p = sorted_profiles[i].second;
            float avg_time = p.total_time_ms / p.execution_count;
            float percentage = (total_time > 0) ? (p.total_time_ms / total_time * 100.0f) : 0.0f;
            std::cout << "  " << (i+1) << ". [" << std::fixed << std::setprecision(2) 
                      << percentage << "%] " << p.layer_name 
                      << " (" << p.layer_type << ") - " 
                      << std::fixed << std::setprecision(4) << avg_time << " ms/iter" << std::endl;
        }
    }
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

    std::cout << "Performing warm-up runs (50 iterations)..." << std::endl;
    for (int i = 0; i < 50; ++i) {
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
    std::cout << "\nPerforming profiling runs (100 iterations with layer-level profiling)..." << std::endl;
    profiler.is_profiling = true;  // ◄ 启用 profiling（跳过预热数据）
    for (int i = 0; i < 100; ++i) {
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
