# TensorRT 性能瓶颈分析与优化指南

## 📊 目录
1. [快速开始](#快速开始)
2. [分析方法](#分析方法)
3. [优化策略](#优化策略)
4. [实践例子](#实践例子)

---

## 🚀 快速开始

### Step 1: 编译 Profiling 工具

```bash
cd /home/zzwtx/fl/edge/build
cmake ..
make tensorrt_profiling
```

### Step 2: 运行分析

```bash
./tensorrt_profiling ../mobilenetv2_fp16.engine
```

### 输出示例
```
================================================================================
              TensorRT Layer Profiling Results
================================================================================
Layer Name                              Type                Time (ms)   % of Total
--------------------------------------------------------------------------------
conv1                                   Convolution         2.1234      15.50%
bn1                                     Scale               0.3456       2.51%
relu1                                   Activation         0.2345       1.71%
layer1/block0/conv                      Convolution         1.8901      13.76%
layer1/block1/conv                      Convolution         1.7654      12.85%
...
total_time_sum                          (all layers)       13.7654     100.00%
--------------------------------------------------------------------------------
Total GPU Execution Time: 13.7654 ms

🔥 Top 5 Time-Consuming Layers (Optimization Candidates):
  1. [18.45%] conv1 (Convolution) - 2.5345 ms
  2. [12.34%] layer2/conv (Convolution) - 1.6923 ms
  3. [11.23%] layer3/conv (Convolution) - 1.5432 ms
  4. [9.87%] layer4/conv (Convolution) - 1.3567 ms
  5. [7.65%] fc1 (FullyConnected) - 1.0523 ms

================================================================================
```

---

## 🔍 分析方法

### 方法1: TensorRT 内置 Profiler（推荐 - 最简单）

**优点**：
- ✅ 无需额外工具，代码集成
- ✅ 逐层时间统计
- ✅ 自动分类 Top 5 瓶颈

**局限性**：
- ❌ 不显示 GPU 利用率
- ❌ 不显示内存带宽占用
- ❌ 不显示缓存效率

**最佳用途**：快速找出时间消耗最多的层

---

### 方法2: NVIDIA Nsys（深度性能分析）

**安装 Nsys**
```bash
# 通常随 CUDA Toolkit 一起安装
which nsys

# 如果没有，可以单独下载
# https://developer.nvidia.com/nsys/nvidia-systems-profiler
```

**使用方法**
```bash
# 1. 运行带 profiling 的程序
nsys profile -o profile_report ./tensorrt_profiling ../mobilenetv2_fp16.engine

# 2. 生成报告
# 输出文件：profile_report.nsys-rep

# 3. 在 VS Code 中打开或使用 nsys-ui 查看
nsys-ui profile_report.nsys-rep
```

**输出内容**：
- GPU 时间轴图表
- 核函数执行时间
- GPU 内存传输
- CPU-GPU 同步点
- 占用率和效率指标

---

### 方法3: NVIDIA NCU（微架构级分析）

**最详细的性能分析**（但执行时间长）

```bash
# 运行 NCU profiler
ncu -o profile_ncu ./tensorrt_profiling ../mobilenetv2_fp16.engine

# 查看结果
ncu --import profile_ncu.ncu-rep
```

**提供的指标**：
- SM (Streaming Multiprocessor) 利用率
- 内存带宽
- 缓存命中率
- 指令吞吐量
- 分支预测效率

---

## 📈 分析结果解读

### 时间消耗分布典型模式

#### **模式 1: 卷积主导（MobileNetV2 典型）**
```
Convolution: 85% ← 优化重点
Activation:  8%
Pooling:     4%
FullyConnected: 3%
```
→ **优化策略**：算子融合（Conv+BN+Activation）、Kernel优化

#### **模式 2: 内存受限**
```
H2D Memory Copy: 30%
Inference:       50%
D2H Memory Copy: 20%
```
→ **优化策略**：减少数据传输、使用异步传输、pinned memory

#### **模式 3: 算子多样化**
```
Conv: 30%, FC: 25%, Gather: 20%, Reshape: 15%, Others: 10%
```
→ **优化策略**：针对性优化多个算子

---

## 🛠️ 优化策略

### 优化层级 1: 算子融合（Operator Fusion）

**什么是算子融合**：
将多个相邻的小算子合并成一个大算子，减少内存往返。

**典型融合方案**：

```
原始图：
Input → Conv → BatchNorm → ReLU → Output
                  ↓           ↓
                 分别调用3个Kernel

融合后：
Input → Conv+BatchNorm+ReLU → Output
              ↓
            一个Kernel，减少内存访问
```

**MobileNetV2 常见融合**：
1. **Conv + BatchNorm**
   - 减少 ~30% 时间
   - TensorRT 通常自动做这个

2. **Conv + Activation** (ReLU/Hardswish)
   - 减少 ~20% 时间
   - 通过 `ITensor::kISH_ACTIVATION` 实现

3. **Depthwise Conv + Pointwise Conv**
   - 减少 ~15% 时间

**如何在 TensorRT 中启用**：

```cpp
// 在 builder 中设置融合选项
config->setFlag(BuilderFlag::kFP16);  // 启用 FP16，促进融合
config->setFlag(BuilderFlag::kINT8);  // INT8 也会触发融合

// 检查融合情况
for (int i = 0; i < engine->getNbLayers(); ++i) {
    ILayer* layer = engine->getLayer(i);
    std::cout << "Layer " << i << ": " << layer->getName() 
              << " - Type: " << (int)layer->getType() << std::endl;
}
```

---

### 优化层级 2: 自定义 Kernel 实现

**什么时候需要**：
- TensorRT 的融合还不够
- 特定的算子组合没有对应的优化 kernel
- 需要超出预期的性能

**实现步骤**：

#### Step 1: 识别关键算子
从 profiling 结果中找出：
```
Top 1: depthwise_conv_layer - 3.2 ms (22%)  ← 这个
Top 2: pointwise_conv_layer - 2.8 ms (19%)
```

#### Step 2: 编写自定义 CUDA Kernel

```cuda
// custom_kernel.cu
// 融合 Depthwise Conv + BatchNorm + ReLU

__global__ void depthwise_conv_bn_relu_kernel(
    const float* input,           // (N, C, H, W)
    const float* weight,          // (C, 1, K, K) for depthwise
    const float* bn_scale,        // (C,)
    const float* bn_bias,         // (C,)
    float* output,                // (N, C, H', W')
    int N, int C, int H, int W, int K, int stride, int padding
) {
    // 线程映射到输出位置
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 计算卷积
    // 应用批归一化
    // 应用 ReLU
    // 写回输出
}
```

#### Step 3: 包装为 TensorRT Plugin

```cpp
// custom_plugin.cpp
class DepthwiseConvBnReLU : public nvinfer1::IPluginV2DynamicExt {
public:
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { /* ... */ }
    
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override { /* ... */ }
    
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in,
        int nbInputs,
        const nvinfer1::DynamicPluginTensorDesc* out,
        int nbOutputs) noexcept override { /* ... */ }
    
    int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs,
                void* const* outputs,
                void* workspace,
                cudaStream_t stream) noexcept override {
        // 调用 custom_kernel
        depthwise_conv_bn_relu_kernel<<<blocks, threads, 0, stream>>>(
            (const float*)inputs[0],
            (const float*)inputs[1],
            // ... 其他参数
        );
        return 0;
    }
};
```

---

### 优化层级 3: 量化 (Quantization)

如果主要瓶颈不是时间而是功耗/内存：

```cpp
// INT8 量化（通常快 3-4 倍）
config->setFlag(BuilderFlag::kINT8);
config->setInt8Calibrator(calibrator);

// FP16（通常快 1.5-2 倍）
config->setFlag(BuilderFlag::kFP16);
```

---

## 📋 实践例子

### 场景：MobileNetV2 优化

**Step 1: Profile 基线**
```bash
./tensorrt_profiling ../mobilenetv2_fp32.engine
```

可能输出：
```
Top Time Consumers:
  1. [18%] conv1_0 (Convolution) - 2.5 ms
  2. [12%] layer2_0_conv (Convolution) - 1.8 ms
  3. [10%] layer3_0_conv (Convolution) - 1.5 ms
  4. [8%] layer4_0_conv (Convolution) - 1.2 ms
  5. [7%] fc (FullyConnected) - 1.0 ms
Total: 14.5 ms → ~69 FPS
```

**Step 2: 启用 TensorRT 融合**
```python
# 在 tensorRT_example.py 中
config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16，促进融合

# 重新构建引擎
engine = builder.build_engine(network, config)
```

新结果：
```
Total: 9.2 ms → ~109 FPS
改善: (14.5 - 9.2) / 14.5 = 36% ✅
```

**Step 3: 写入自定义 Depthwise Conv 融合 Kernel**
（如果上述还不够）

新结果：
```
Total: 7.1 ms → ~141 FPS
累积改善: (14.5 - 7.1) / 14.5 = 51% ✅
```

---

## 🎯 优化决策树

```
观察到性能瓶颈
    ↓
问题是什么类型？
    ├─ 卷积层慢 → Conv + BN + Activation 融合
    │            → 自定义 Kernel（Depthwise）
    │            → INT8 量化
    │
    ├─ 内存传输慢 → 减少 H2D/D2H 次数
    │              → 使用 pinned memory
    │              → 异步传输
    │
    ├─ 多个算子慢 → 检查能否融合
    │              → 使用 Constant Folding
    │              → 图优化（移除冗余节点）
    │
    └─ 功耗高 → INT8 或 FP16 量化
              → 降低 batch size
              → 使用稀疏性
```

---

## 📚 关键资源

| 资源 | 用途 |
|------|------|
| `nsys profile` | GPU 时间轴分析 |
| `ncu` | 微架构级细节 |
| TensorRT Profiler | 逐层时间 |
| NVIDIA Docs | 最佳实践 |

---

## ✅ 检查清单

- [ ] 运行 TensorRT profiler，找出 Top 5 耗时层
- [ ] 检查 profiler 输出中"Top 5"部分
- [ ] 统计卷积层时间占比（通常 > 80%）
- [ ] 评估 FP16 或 INT8 收益
- [ ] 考虑是否需要自定义 Kernel
- [ ] 验证优化后的性能提升

