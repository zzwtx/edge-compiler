import onnxruntime
import numpy as np
import time
import psutil
import os
from datasets import load_dataset
from itertools import islice
from transformers import AutoImageProcessor

# --- 配置参数 ---
ONNX_MODEL_PATH = "mobilenetv2.onnx"
WARMUP_RUNS = 50  # 热身运行次数，确保 GPU 达到稳定状态
PROFILE_RUNS = 100  # 实际用于计时的推理次数

def profile_onnx():
    """
    对指定的 ONNX 模型进行性能分析，测量其延迟、吞吐量和资源使用情况。
    """
    print("--- ONNX Runtime Performance Profiler ---")
    
    # --- 1. 环境和模型设置 ---
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ Error: Model file not found at '{ONNX_MODEL_PATH}'")
        return

    print(f"ONNX Runtime version: {onnxruntime.__version__}")
    print(f"Available Providers: {onnxruntime.get_available_providers()}")

    # 创建推理会话，优先使用 CUDA
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    selected_provider = session.get_providers()[0]
    print(f"Using Provider: {selected_provider}")

    # 获取模型输入信息
    input_details = session.get_inputs()[0]
    input_name = input_details.name
    input_shape = input_details.shape
    
    # 将动态维度（如 'batch_size'）替换为 1
    input_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
    
    print(f"Model Input: name='{input_name}', shape={input_shape}")

    # 创建一个符合模型输入的随机虚拟数据
    # dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # 加载 ImageNet-1k 验证集的前100张图片
    ds = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    images = []
    labels = []
    for sample in islice(ds, 100):
        img = sample["image"]
        img = img.convert("RGB")  # 强制转换为三通道彩色图像
        images.append(img)
        labels.append(sample["label"])

    print(f"已加载图片数量: {len(images)}")
    
    processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")  # 或您的本地模型目录
    # 批量处理图片
    inputs = processor(images=images, return_tensors="np")  # 返回 numpy 数组
    dummy_input = inputs["pixel_values"]  # 形状: [batch, 3, 224, 224]，float32

    # --- 2. 热身运行 ---
    print(f"\nPerforming {WARMUP_RUNS} warm-up runs...")
    for _ in range(WARMUP_RUNS):
        for i in range(len(dummy_input)):
            session.run(None, {input_name: dummy_input[i:i+1]})  # 只输入一张图片
    print("Warm-up complete.")

    # --- 3. 性能评测 ---
    print(f"Performing {PROFILE_RUNS} timed runs...")
    latencies = []
    
    # 获取当前进程以监控资源
    process = psutil.Process(os.getpid())
    max_cpu_usage = 0
    max_ram_usage = 0

    for _ in range(PROFILE_RUNS):
        max_cpu_usage = max(max_cpu_usage, process.cpu_percent())
        max_ram_usage = max(max_ram_usage, process.memory_info().rss)
        start_time = time.perf_counter()
        for i in range(len(dummy_input)):
            session.run(None, {input_name: dummy_input[i:i+1]})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

    print("Profiling complete.")

    # --- 4. 结果分析与报告 ---
    latencies_np = np.array(latencies)
    
    avg_latency = np.mean(latencies_np)
    median_latency = np.median(latencies_np)
    std_dev = np.std(latencies_np)
    min_latency = np.min(latencies_np)
    max_latency = np.max(latencies_np)
    p95 = np.percentile(latencies_np, 95)
    p99 = np.percentile(latencies_np, 99)
    throughput = 1000 / avg_latency if avg_latency > 0 else 0

    print("\n--- ONNX Model Performance ---")
    print(f"Total runs:      {PROFILE_RUNS}")
    print(f"Warm-up runs:    {WARMUP_RUNS}")
    
    print("\n--- Latency (ms) ---")
    print(f"Average:         {avg_latency:.2f} ms")
    print(f"Median:          {median_latency:.2f} ms")
    print(f"Std Dev:         {std_dev:.2f} ms")
    print(f"Min:             {min_latency:.2f} ms")
    print(f"Max:             {max_latency:.2f} ms")
    print(f"P95:             {p95:.2f} ms")
    print(f"P99:             {p99:.2f} ms")

    print("\n--- Throughput ---")
    print(f"FPS (Frames/Sec): {throughput:.2f}")

    print("\n--- Resource Usage (during profiling) ---")
    print(f"CPU Usage (max): {max_cpu_usage:.1f}%")
    print(f"RAM Usage (max): {max_ram_usage / (1024**2):.2f} MB")


if __name__ == "__main__":
    profile_onnx()