"""
TensorRT 量化对比脚本
比较 FP32、FP16、INT8 的性能和精度
"""

import tensorrt as trt
import numpy as np
import time
from pathlib import Path

logger = trt.Logger(trt.Logger.WARNING)

def build_engine_with_quantization(onnx_path, engine_path, quantization_type):
    """
    使用不同精度构建 TensorRT 引擎
    
    Args:
        onnx_path: ONNX 模型路径
        engine_path: 输出引擎文件路径
        quantization_type: 'FP32', 'FP16', 或 'INT8'
    """
    
    # 创建 builder 和网络
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析 ONNX
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print(f"❌ Failed to parse ONNX: {onnx_path}")
            return None
    
    # 创建 Config
    config = builder.create_builder_config()
    
    # 设置最大工作空间
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # 根据精度配置
    if quantization_type == 'FP16':
        print("[Config] Enabling FP16 precision...")
        config.set_flag(trt.BuilderFlag.FP16)
        
    elif quantization_type == 'INT8':
        print("[Config] Enabling INT8 precision...")
        print("⚠️  Note: INT8 requires calibration data. Using simplified calibrator.")
        config.set_flag(trt.BuilderFlag.INT8)
        
        # 简单的校准器实现
        class SimpleCalibrator(trt.IInt8Calibrator):
            def __init__(self, batch_size=1):
                self.batch_size = batch_size
                self.current_batch = 0
                
            def get_batch_size(self):
                return self.batch_size
            
            def get_batch(self, names):
                # 生成随机校准数据
                if self.current_batch < 10:  # 仅 10 个 batch
                    batch = np.random.randn(self.batch_size, 3, 224, 224).astype(np.float32)
                    self.current_batch += 1
                    # 返回 numpy 数组
                    return [batch]
                return None
            
            def read_calibration_cache(self):
                return None
            
            def write_calibration_cache(self, cache):
                pass
        
        config.int8_calibrator = SimpleCalibrator(batch_size=1)
    
    else:  # FP32 (默认)
        print("[Config] Using FP32 precision (default)...")
    
    # 构建引擎
    print(f"[Building] Engine with {quantization_type}...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print(f"❌ Failed to build engine with {quantization_type}")
        return None
    
    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"✅ Engine saved to {engine_path}")
    
    return engine


def benchmark_engine(engine_path, num_runs=100, warmup_runs=10):
    """
    对引擎进行基准测试（使用 TensorRT runtime，不依赖 pycuda）
    
    Args:
        engine_path: 引擎文件路径
        num_runs: 性能测试运行次数
        warmup_runs: 热身运行次数
        
    Returns:
        dict: 包含延迟统计的字典
    """
    
    # 反序列化引擎
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # 获取输入/输出信息
    input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))
    output_shape = engine.get_tensor_shape(engine.get_tensor_name(1))
    
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {output_shape}")
    
    # 分配内存
    input_data = np.random.randn(*input_shape).astype(np.float32)
    output_data = np.empty(output_shape, dtype=np.float32)
    
    # 热身
    print(f"  ▶ Warm-up ({warmup_runs} runs)...")
    for _ in range(warmup_runs):
        context.execute_v2([input_data, output_data])
    print(f"  ✓ Warm-up complete")
    
    # 基准测试
    print(f"  ▶ Profiling ({num_runs} runs)...")
    latencies = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        context.execute_v2([input_data, output_data])
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # 每 20 次打印进度
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{num_runs}...", end=" ")
    print("\n  ✓ Profiling complete")
    
    # 统计
    latencies = np.array(latencies)
    stats = {
        'mean': np.mean(latencies),
        'median': np.median(latencies),
        'std': np.std(latencies),
        'min': np.min(latencies),
        'max': np.max(latencies),
        'p95': np.percentile(latencies, 95),
        'p99': np.percentile(latencies, 99),
        'fps': 1000.0 / np.mean(latencies),
    }
    
    # 清理
    del context
    del engine
    
    return stats


def compare_quantization():
    """对比三种精度的性能"""
    
    onnx_path = "mobilenetv2.onnx"
    
    if not Path(onnx_path).exists():
        print(f"❌ ONNX model not found at {onnx_path}")
        return
    
    quantization_types = ['FP32', 'FP16', 'INT8']
    results = {}
    
    print("=" * 80)
    print("TensorRT Quantization Comparison (FP32 vs FP16 vs INT8)")
    print("=" * 80)
    
    for qtype in quantization_types:
        engine_path = f"mobilenetv2_{qtype.lower()}.engine"
        
        print(f"\n[{qtype}] ========================================")
        print(f"Building engine...")
        engine = build_engine_with_quantization(onnx_path, engine_path, qtype)
        
        if engine is None:
            print(f"❌ Failed to build {qtype} engine")
            continue
        
        print(f"Benchmarking...")
        stats = benchmark_engine(engine_path, num_runs=100, warmup_runs=10)
        results[qtype] = stats
        
        print(f"\n[{qtype}] Results:")
        print(f"  Average Latency: {stats['mean']:.3f} ms")
        print(f"  Median Latency:  {stats['median']:.3f} ms")
        print(f"  Std Dev:         {stats['std']:.3f} ms")
        print(f"  Min - Max:       {stats['min']:.3f} - {stats['max']:.3f} ms")
        print(f"  P95 - P99:       {stats['p95']:.3f} - {stats['p99']:.3f} ms")
        print(f"  FPS:             {stats['fps']:.1f}")
    
    # 对比总结
    if len(results) >= 2:
        print("\n" + "=" * 80)
        print("Performance Comparison Summary")
        print("=" * 80)
        print(f"{'Precision':<15} {'Latency (ms)':<20} {'FPS':<20} {'Speedup vs FP32':<20}")
        print("-" * 80)
        
        fp32_latency = results['FP32']['mean'] if 'FP32' in results else float('inf')
        
        for qtype in quantization_types:
            if qtype in results:
                latency = results[qtype]['mean']
                fps = results[qtype]['fps']
                speedup = fp32_latency / latency if fp32_latency != float('inf') else 1.0
                print(f"{qtype:<15} {latency:<20.3f} {fps:<20.1f} {speedup:<20.2f}x")
        
        print("=" * 80)
        print("\n📊 Analysis:")
        print("  FP16: Usually provides ~2x speedup with minimal accuracy loss")
        print("  INT8: Usually provides ~4x speedup but requires careful calibration")
        print("\n💡 Recommendation:")
        if 'FP16' in results and 'FP32' in results:
            fp16_speedup = results['FP32']['mean'] / results['FP16']['mean']
            print(f"  FP16 provides {fp16_speedup:.2f}x speedup with typically <1% accuracy loss")
        if 'INT8' in results and 'FP32' in results:
            int8_speedup = results['FP32']['mean'] / results['INT8']['mean']
            print(f"  INT8 provides {int8_speedup:.2f}x speedup but needs proper calibration")


if __name__ == "__main__":
    try:
        compare_quantization()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

