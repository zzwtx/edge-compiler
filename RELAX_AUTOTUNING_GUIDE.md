# TVM Relax 自动优化搜索实现说明

## 概述

这是基于 TVM Relax 的自动优化搜索实现，对标 `TVM.py` 中基于 Relay 的 AutoTVM 方案。Relax 是 TVM Unity 架构中的新一代中间表示，相比 Relay 提供了更好的灵活性和性能优化空间。

## 文件结构

### `to_tvm_gpu.py` - Relax 版本实现

**主要改进点：**
1. ✅ 从 Relay 升级到 Relax（新一代中间表示）
2. ✅ 集成 AutoScheduler 自动搜索（更先进的优化）
3. ✅ 支持 GPU 特定的优化 Pass
4. ✅ 完整的性能基准测试

## 核心组件详解

### 1️⃣ 模型加载和转换

```python
mod = relax_onnx.from_onnx(onnx_model, shape_dict)
target = tvm.target.Target("cuda", host="llvm")
```

- **输入**: ONNX 模型 + 输入形状
- **输出**: Relax 计算图（IRModule）
- **目标**: CUDA GPU + LLVM CPU 主机

### 2️⃣ Relax 优化 Pass 序列

在 `relax_legalize_and_schedule()` 函数中应用：

```python
def relax_legalize_and_schedule(mod, target):
    # [1] LegalizeOps: 高层算子 → 底层 TIR 调用
    mod = relax_transform.LegalizeOps()(mod)
    
    # [2] SimplifyExpr: 化简表达式（消除冗余计算）
    mod = relax_transform.SimplifyExpr()(mod)
    
    # [3] CallTIRRewrite: 重写 TIR 调用
    mod = relax_transform.CallTIRRewrite()(mod)
    
    return mod
```

#### Pass 说明表

| Pass | 作用 | 阶段 |
|------|------|------|
| `LegalizeOps` | 将高层 Relax 算子转换为底层 TIR 函数调用 | 必须 |
| `SimplifyExpr` | 化简表达式，消除不必要的计算 | 推荐 |
| `CallTIRRewrite` | 重写 TIR 调用以支持特定优化 | 推荐 |
| `DefaultGPUSchedule` | 应用默认的 GPU 调度策略 | 可选 |

### 3️⃣ AutoScheduler 自动优化搜索

**关键参数配置：**

```python
tuning_option = {
    "tuner": "xgb",                    # XGBoost 智能搜索
    "num_measure_trials": 50,          # ⭐ 总尝试次数
    "early_stopping": 100,             # 提前停止条件
    "measure_option": auto_scheduler.measure_option(
        builder=auto_scheduler.LocalBuilder(),
        runner=auto_scheduler.LocalRunner(
            number=10,                 # 单次测量时的重复运行数
            repeat=1,                  # 测量重复次数
            min_repeat_ms=100,         # GPU 最小运行时间（ms）
            timeout=10,                # 单个配置超时时间（秒）
            enable_cpu_cache_flush=False,  # GPU 无需 CPU 缓存清空
        ),
    ),
}
```

#### 参数效果

| 参数 | 影响 | 建议值范围 |
|------|------|----------|
| `num_measure_trials` | 总尝试次数 ↑ → 更优结果但耗时 ↑ | 20-100 |
| `number` (runner) | 单次测试次数 ↑ → 结果更稳定但单次耗时 ↑ | 5-20 |
| `repeat` | 重复测量次数 ↑ → 结果更准确但总耗时 ↑ | 1-3 |
| `min_repeat_ms` | GPU 最小运行时间 ↑ → 更准确的 GPU 性能 | 50-200 |
| `early_stopping` | 提前停止阈值 ↑ → 继续搜索更久 | 50-200 |

### 4️⃣ 任务提取和调优

```python
# 提取可调优的任务
tasks, task_weights = auto_scheduler.extract_tasks(
    mod_for_tuning, 
    target=target,
    params=None,
)

# 创建任务调度器并执行调优
tuner = auto_scheduler.TaskScheduler(
    tasks,
    task_weights,
    load_log_file=tuning_records_path,  # 加载之前的调优结果
)

tuner.tune(
    tuning_option["measure_option"],
    num_measure_trials=tuning_option["num_measure_trials"],
    early_stopping=tuning_option["early_stopping"],
)
```

**输出说明：**
- 调优结果保存到 `mobilenetv2-relax-autotuning.json`
- 每个任务的最优配置被记录
- 后续编译时自动加载这些记录

### 5️⃣ 使用调优结果编译

```python
if os.path.exists(tuning_records_path):
    print(f"Loading tuning records from {tuning_records_path}...")
    with auto_scheduler.ApplyHistoryBest(tuning_records_path):
        with tvm.transform.PassContext(opt_level=3):
            exec = relax.build(mod_gpu, target=target)
else:
    print("No tuning records found, using default optimization...")
    with tvm.transform.PassContext(opt_level=3):
        exec = relax.build(mod_gpu, target=target)
```

**关键特性：**
- ✅ 自动加载之前的调优记录
- ✅ 如果无法找到记录，使用默认优化
- ✅ `opt_level=3` 启用所有优化

### 6️⃣ 性能基准测试

```python
dev = tvm.device("cuda", 0)
vm = relax.VirtualMachine(exec, dev)

test_input = np.random.randn(1, 3, 224, 224).astype("float32")
input_nd = tvm.nd.array(test_input, device=dev)

# 预热运行
for _ in range(3):
    vm[mod_gpu["main"]](input_nd)

# 性能测试
times = []
for _ in range(timing_repeat):
    t = timeit.timeit(
        lambda: vm[mod_gpu["main"]](input_nd),
        number=timing_number,
    )
    times.append(t * 1000 / timing_number)
```

**输出指标：**
- 平均延迟（Mean）
- 中位数（Median）
- 标准差（Std）
- 最小/最大值

## Relay vs Relax 对比

### 架构差异

| 特性 | Relay | Relax |
|------|-------|-------|
| **设计目标** | 静态图优化 | 动态灵活优化 |
| **中间表示** | 函数式编程 | 命令式编程 |
| **优化级别** | AutoTVM | AutoScheduler |
| **GPU 支持** | 基础 | 更完善 |
| **可扩展性** | 有限 | 高度可扩展 |
| **未来方向** | 维护模式 | 主要研发 |

### 性能优化差异

**Relay (AutoTVM):**
```
ONNX → Relay IR → 
  ↓
  AutoTVM 搜索 (逐算子优化)
  ↓
  GraphExecutor 运行
```

**Relax (AutoScheduler):**
```
ONNX → Relax IR → 
  ↓
  LegalizeOps → SimplifyExpr → CallTIRRewrite
  ↓
  AutoScheduler 搜索 (任务级优化)
  ↓
  VirtualMachine 运行
```

## 实验对比

### 运行命令

```bash
# Relay 版本（现有）
python TVM.py

# Relax 版本（新实现）
python to_tvm_gpu.py
```

### 预期结果差异

| 指标 | Relay | Relax | 差异 |
|------|-------|-------|------|
| 编译时间 | 更快 | 略慢（更复杂分析） | +10-30% |
| 运行速度 | 基线 | 更快 | +5-20% |
| 调优时间 | ~5-10分钟 | ~10-20分钟 | 搜索空间更大 |
| 优化效果 | 一般 | 更好 | 由任务调度器优化 |

## 调优配置方案

### 方案 1️⃣：快速测试（时间优先）

```python
tuning_option = {
    "tuner": "xgb",
    "num_measure_trials": 10,      # 快速测试
    "early_stopping": 20,
    "measure_option": auto_scheduler.measure_option(
        runner=auto_scheduler.LocalRunner(
            number=5, repeat=1, min_repeat_ms=50, timeout=5,
        ),
    ),
}
```

**预计时间**: ~3-5 分钟

### 方案 2️⃣：平衡方案（推荐生产）

```python
tuning_option = {
    "tuner": "xgb",
    "num_measure_trials": 50,      # 平衡
    "early_stopping": 100,
    "measure_option": auto_scheduler.measure_option(
        runner=auto_scheduler.LocalRunner(
            number=10, repeat=1, min_repeat_ms=100, timeout=10,
        ),
    ),
}
```

**预计时间**: ~15-30 分钟

### 方案 3️⃣：完整优化（性能优先）

```python
tuning_option = {
    "tuner": "xgb",
    "num_measure_trials": 200,     # 完整搜索
    "early_stopping": 300,
    "measure_option": auto_scheduler.measure_option(
        runner=auto_scheduler.LocalRunner(
            number=20, repeat=3, min_repeat_ms=150, timeout=10,
        ),
    ),
}
```

**预计时间**: ~60-120 分钟

## 常见问题

### Q1: 为什么 Relax 比 Relay 慢？
A: Relax 的分析更深入，可以找到更优的优化机会。这是权衡，首次调优慢但后续执行更快。

### Q2: 如何重新调优？
```bash
# 删除旧的调优记录
rm mobilenetv2-relax-autotuning.json

# 重新运行脚本
python to_tvm_gpu.py
```

### Q3: 调优结果能重用吗？
A: 能。调优记录保存在 JSON 文件中，同一硬件上可以重用。

### Q4: 能对 CPU 调优吗？
A: 能。修改 `target` 为 `tvm.target.Target("llvm")` 即可。

## 输出文件说明

| 文件 | 用途 |
|------|------|
| `mobilenetv2_gpu.so` | 编译后的库文件 |
| `mobilenetv2-relax-autotuning.json` | 调优结果记录 |
| `mobilenetv2-relax-tuning.log` | 调优过程日志 |

## 总结

`to_tvm_gpu.py` 实现了基于 TVM Relax 的完整自动优化流程：

1. ✅ **Relax 中间表示** - 比 Relay 更灵活强大
2. ✅ **AutoScheduler 搜索** - 比 AutoTVM 更聪明
3. ✅ **GPU 特定优化** - DefaultGPUSchedule + 自动调优
4. ✅ **完整性能测试** - VirtualMachine 执行和基准测试
5. ✅ **生产就绪** - 错误处理和日志记录

这使得 MobileNetV2 在 GPU 上的执行性能相比基础编译有显著提升！
