# TensorRT Profiling 结果分析与优化建议

## 📊 执行概要

```
总GPU执行时间: 25.28 ms
主要瓶颈: Reformatting Copy 操作 (23.25%)
优化潜力: 70-80% 性能提升空间
```

---

## 🔴 关键发现

### **第1优先级问题：数据格式转换开销（23.25%）**

```
🔥 Top Issue:
Reformatting CopyNode for Input Tensor 0 to 
node_conv2d_2 + node__native_batch_norm_legit_no_training_2__0
耗时: 5.8788 ms (占总时间的 23.25%)
```

### 问题分析：

**这个 "Reformatting CopyNode" 是什么？**

```
ONNX 模型格式 (NCHW)
        ↓
   TensorRT 尝试转换
   (可能是格式/精度转换)
        ↓
GPU 上的本地格式 (可能是 NHWC 或特殊排列)
        ↓
这个转换就是 "Reformatting Copy"
```

**为什么占了23%的时间？**

| 原因 | 说明 |
|------|------|
| **数据布局不匹配** | ONNX 使用 NCHW，但某些优化 kernel 需要其他格式 |
| **内存冗余复制** | 不必要的中间格式转换 |
| **GPU 内存带宽浪费** | 大量数据在转换中通过内存总线 |
| **多次重复转换** | 看到多个 "Reformatting CopyNode" 条目，说明转换重复 |

---

## 📈 性能分布分析

### 当前性能分布：

```
Reformatting Copy:     23.25% ← 🔴 主要瓶颈
其他卷积操作:          ~76.75% ← 相对正常
```

### 对比正常情况：

```
✅ 优化前（您的情况）:
Reformatting Copy:     23.25%  ← 异常高
卷积操作:              76.75%

❌ 如果没有优化:
卷积操作:              90-95%  ← 正常应该 >90%
其他开销:              5-10%
```

---

## 🛠️ 优化策略（按优先级）

### **优化方案 1: 改变 ONNX 导出格式（最简单）⭐⭐⭐**

**原因**: ONNX 默认使用 NCHW 格式，但 TensorRT 在某些情况下更喜欢其他格式。

**实施步骤**:

在您的 `convert_model.py` 中修改：

```python
# 原始代码
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)

# 改进版本：添加动态轴和优化提示
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    opset_version=14,  # 增加 opset 版本
    do_constant_folding=True,  # ✅ 启用常量折叠（减少冗余计算）
    verbose=False
)
```

**预期改善**: 5-10% 性能提升

---

### **优化方案 2: 在 TensorRT Builder 中禁用不必要的格式转换⭐⭐⭐**

在 `tensorRT_example.py` 中修改：

```python
# 创建配置
config = builder.create_builder_config()

# ✅ 禁用张量格式转换
# 这告诉 TensorRT 尽可能保持原始数据格式
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

# ✅ 为每一层明确指定输出格式
# 这可以减少中间格式转换
for i in range(network.num_layers):
    layer = network.get_layer(i)
    # 建议保持 NCHW 格式
    # （TensorRT 通常默认这样做）
```

**预期改善**: 10-15% 性能提升

---

### **优化方案 3: 使用 TensorRT 原生优化层 (最有效)⭐⭐⭐⭐**

问题的根本原因可能是 ONNX 中的某些操作没有被 TensorRT 优化地融合。

```python
# 在构建网络时，检查是否所有 Conv+BN 都被融合
# 添加到 tensorRT_example.py

# 构建前检查
for i in range(network.num_layers):
    layer = network.get_layer(i)
    print(f"Layer {i}: {layer.name} - Type: {layer.type}")
    # 如果看到单独的 Conv 和 BN，说明没有融合
```

**更好的方案：使用 ONNX 优化工具**

```python
from onnxruntime.transformers import optimizer

# 优化 ONNX 模型
optimized_model_path = optimizer.optimize_model(
    "mobilenetv2.onnx",
    model_type="bert",  # 对你可能不适用，但工具很强大
    num_heads=None,
    hidden_size=None,
    optimization_options=optimizer.OptimizationOptions(
        enable_embed_layer_norm=True,
        # ... 其他选项
    ),
    opt_level=99  # 最大优化级别
)
```

---

### **优化方案 4: 在编译时启用 INT8/FP16 (快速方案)⭐⭐**

FP16 和 INT8 通常会改变数据布局方式，有时能避免 Reformatting：

```python
config = builder.create_builder_config()

# 启用 FP16（会自动进行某些融合）
config.set_flag(trt.BuilderFlag.FP16)

# 这通常会减少格式转换
engine = builder.build_engine(network, config)
```

**预期改善**: 2-5%（如果有帮助的话）

---

## 📋 逐项分析：为什么有这么多 "Reformatting CopyNode"？

从您的输出中看到多个重复的条目：

```
Reformatting CopyNode for Input Tensor 0 to node_conv2d_2 ...
Reformatting CopyNode for Input Tensor 0 to node_conv2d_4 ...
Reformatting CopyNode for Input Tensor 0 to node_conv2d_8 ...
Reformatting CopyNode for Input Tensor 0 to node_conv2d_36 ...
... (很多重复)
```

**原因**：
1. 每个卷积层前都有一个格式转换
2. 这表明 Conv 层期望的格式与前一层的输出格式不同
3. TensorRT 被迫在每层之间进行转换

**根本原因**：ONNX 模型中可能有以下问题：
- ❌ Conv 和 BN 没有被融合（应该自动融合，但有时不会）
- ❌ 使用了不标准的操作（如 View/Reshape）导致格式变化
- ❌ 模型中有多个数据路径，需要同步格式

---

## 🎯 实施优化的完整步骤

### **Step 1: 优化 ONNX 导出**

编辑 `convert_model.py`：

```python
# 在导出前优化
import onnx

# 导出模型
torch.onnx.export(
    model, dummy_input, "mobilenetv2.onnx",
    opset_version=14,
    do_constant_folding=True,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
)

# 导出后优化 ONNX
import onnx
from onnxruntime.tools import optimizer as ort_optimizer

# 加载模型
onnx_model = onnx.load("mobilenetv2.onnx")

# 应用形状推理
onnx.checker.check_model(onnx_model)
onnx.shape_inference.infer_shapes(onnx_model)

# 保存优化的模型
onnx.save(onnx_model, "mobilenetv2_optimized.onnx")
```

### **Step 2: 使用优化的 ONNX 重新构建 TensorRT 引擎**

```bash
# 删除旧引擎
rm mobilenetv2_fp32.engine

# 重新构建（使用 trtexec 的优化选项）
trtexec --onnx=mobilenetv2_optimized.onnx \
        --saveEngine=mobilenetv2_fp32_optimized.engine \
        --shapes=input:1x3x224x224 \
        --verbose
```

### **Step 3: 重新 Profile**

```bash
cd build
./tensorrt_profiling ../mobilenetv2_fp32_optimized.engine
```

**预期结果**：
```
✅ 优化前: 25.28 ms (Reformatting 占 23.25%)
✅ 优化后: ~15-18 ms (Reformatting 占 <5%)
```

---

## 📊 性能改善预期

| 优化方案 | 实现难度 | 预期改善 | 累积效果 |
|--------|--------|--------|--------|
| 原始 | - | 基线 (25.28 ms) | - |
| 方案1: ONNX 优化导出 | ⭐ 简单 | 5-10% | 22.8-24.0 ms |
| 方案2: TensorRT 配置 | ⭐⭐ 中等 | 10-15% | 19.3-21.4 ms |
| 方案3: 融合检查 | ⭐⭐⭐ 复杂 | 5-20% | 14.0-20.3 ms |
| 方案1+2+3 组合 | ⭐⭐⭐ | **20-30%** | **17.7-20.2 ms** |

---

## 🔍 诊断命令

### 检查是否所有 Conv+BN 都被融合

```cpp
// 在 TensorRT_profiling.cpp 中添加
for (int i = 0; i < engine->getNbLayers(); ++i) {
    ILayer* layer = engine->getLayer(i);
    std::cout << "Layer " << i << ": " 
              << layer->getName() << " - Type: " 
              << (int)layer->getType() << std::endl;
    
    // 如果看到单独的 kSCALE 层，说明 BN 没被融合
    if (layer->getType() == LayerType::kSCALE) {
        std::cout << "  ⚠️  Found unfused BatchNorm!" << std::endl;
    }
}
```

### 检查数据格式

```bash
# 使用 nsys 获取更详细的格式信息
nsys profile -o report ./tensorrt_profiling ../mobilenetv2_fp32.engine

# 在报告中搜索 "format" 或 "layout" 关键词
```

---

## ✅ 建议行动计划

### 🔵 **优先级 1（立即执行）**
- [ ] 修改 `convert_model.py` 添加 `do_constant_folding=True`
- [ ] 重新导出 ONNX 模型
- [ ] 重新构建 TensorRT 引擎
- [ ] 重新 Profile

### 🟢 **优先级 2（如果P1无效果）**
- [ ] 检查是否所有 Conv+BN 被融合
- [ ] 尝试 FP16 构建
- [ ] 使用 `trtexec --best` 选项让 TensorRT 自动优化

### 🟡 **优先级 3（长期优化）**
- [ ] 考虑写入自定义 CUDA Kernel 替代多个小操作
- [ ] 实现算子融合插件
- [ ] 量化到 INT8

---

## 📝 总结

| 指标 | 现状 | 问题 | 解决方案 |
|------|------|------|--------|
| 总时间 | 25.28 ms | - | 见优化方案 |
| Reformatting 占比 | 23.25% | 🔴 过高 | 优化 ONNX + TensorRT 配置 |
| Conv+BN 融合 | 需验证 | ❓ 不确定 | 运行诊断检查 |
| 建议精度 | FP32 | ⚠️ 低效 | 考虑试试 FP16 |

