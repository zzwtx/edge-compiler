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
    // 3) 加载真实数据并填充输入
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

    // 使用加载的数据填充输入缓冲区
    for (int b = 0; b < nbBindings; ++b) {
        if (!bindingIsInput[b]) continue; // 只处理输入

        // 验证数据尺寸是否匹配
        size_t loaded_data_size_bytes = images_arr.num_vals * sizeof(float);
        if (hostSizes[b] != loaded_data_size_bytes) {
            std::cerr << "Error: Input data size mismatch for binding " << b << ".\n";
            std::cerr << "Engine expects " << hostSizes[b] << " bytes, but loaded data has " << loaded_data_size_bytes << " bytes.\n";
            return 1;
        }

        // 使用 memcpy 将加载的图像数据直接拷贝到主机缓冲区
        std::cout << "Copying loaded image data to host buffer for binding " << b << std::endl;
        memcpy(hostBuffers[b], images_arr.data<float>(), loaded_data_size_bytes);
        
        // 将准备好的主机输入拷贝到设备
        cudaMemcpy(deviceBindings[b], hostBuffers[b], hostSizes[b], cudaMemcpyHostToDevice);
    }

    // -----------------------
    // 4) 运行推理（入队）
    // -----------------------
    // 创建 CUDA stream 用于异步执行
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // enqueueV3 在给定的 stream 上入队并执行（异步）。
    // 如果希望同步阻塞，可使用 context->executeV2(...)。
    context->enqueueV3(stream);

    // -----------------------
    // 5) 将输出从设备拷回主机并后处理
    // -----------------------
    long* true_labels = labels_arr.data<long>();
    int correct_predictions = 0;
    size_t num_images = labels_arr.num_vals;

    for (int b = 0; b < nbBindings; ++b) {
        if (bindingIsInput[b]) continue; // 跳过输入

        // 将设备输出拷回主机缓冲区
        cudaMemcpy(hostBuffers[b], deviceBindings[b], hostSizes[b], cudaMemcpyDeviceToHost);

        // 假设输出是 [100, 1000] 的形状，代表100张图片，每张1000个分类得分
        float* out = reinterpret_cast<float*>(hostBuffers[b]);
        size_t num_classes = hostSizes[b] / sizeof(float) / num_images;

        if (num_classes != 1000) {
             std::cout << "Warning: Output does not seem to be for 1000 classes. Verification might be incorrect." << std::endl;
        }

        std::cout << "\n--- Verification ---" << std::endl;
        for (size_t i = 0; i < num_images; ++i) {
            float* current_image_output = out + i * num_classes;
            int argmax = 0;
            float maxv = current_image_output[0];
            for (size_t j = 1; j < num_classes; ++j) {
                if (current_image_output[j] > maxv) {
                    maxv = current_image_output[j];
                    argmax = static_cast<int>(j);
                }
            }
            if (i < 10) { // 只打印前10张图片的结果
                 std::cout << "Image " << i << ": Predicted class=" << argmax << " (score=" << maxv << "), True label=" << true_labels[i] << std::endl;
            }
            if (argmax == true_labels[i]) {
                correct_predictions++;
            }
        }
        std::cout << "--------------------" << std::endl;
        std::cout << "Accuracy: " << (float)correct_predictions / num_images * 100.0f << "% (" << correct_predictions << "/" << num_images << ")" << std::endl;
    }

    // 在清理前确保 stream 上的所有工作已完成
    cudaStreamSynchronize(stream);
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