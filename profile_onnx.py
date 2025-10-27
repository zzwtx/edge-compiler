import onnxruntime
import numpy as np
import time
import psutil
import os
from datasets import load_dataset
from itertools import islice
from transformers import AutoImageProcessor
import torch
from transformers import AutoModelForImageClassification

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

    # --- 1a. 加载模型 ---
    print("--- Model Loading ---")
    # 加载 ONNX 模型
    
    # 加载原始 PyTorch 模型用于对比
    print("Loading PyTorch model for verification...")
    pytorch_model = AutoModelForImageClassification.from_pretrained("/home/zzwtx/fl/edge/model_files")
    pytorch_model.eval()
    # 如果有可用的GPU，将PyTorch模型也移到GPU上
    if torch.cuda.is_available():
        pytorch_model.to("cuda")
    
    # --- 1b. 加载数据并预处理 ---
    print("\nLoading and preprocessing 100 images from ImageNet-1k...")
    # 使用 streaming=True 避免下载整个数据集
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    
    images = []
    labels = []
    # 使用 islice 高效地只获取前100个样本
    for sample in islice(dataset, 100):
        # 确保所有图片都是三通道的 RGB 格式
        images.append(sample["image"].convert("RGB"))
        labels.append(sample["label"])

    print(f"Loaded {len(images)} images and labels.")

    # 加载与模型匹配的预处理器
    # 我们从本地加载，确保与转换模型时使用的预处理器完全一致
    processor = AutoImageProcessor.from_pretrained("/home/zzwtx/fl/edge/model_files")
    
    # 批量预处理所有图片，返回 NumPy 数组
    pixel_values = processor(images=images, return_tensors="np")["pixel_values"]
    print(f"Images preprocessed into tensor with shape: {pixel_values.shape}")

    # --- 2. 正确率/一致性评测 (Accuracy/Consistency Profiling) ---
    print("\n--- Model Consistency Calculation ---")
    consistent_predictions = 0
    right_predictions = 0
    for i in range(len(images)):
        # 每次只取一张图片进行推理，注意保持 batch 维度
        # pixel_values[i:i+1] 的 shape 是 (1, 3, 224, 224)
        input_tensor = pixel_values[i:i+1]
        
        # 运行推理
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        
        # 获取预测结果
        onnx_predicted_label = np.argmax(logits, axis=1)[0]
        
        # PyTorch 推理
        with torch.no_grad():
            # 将numpy数组转为torch tensor，并移到GPU
            torch_input = torch.from_numpy(input_tensor)
            if torch.cuda.is_available():
                torch_input = torch_input.to("cuda")
            
            pytorch_outputs = pytorch_model(torch_input)
            pytorch_predicted_label = torch.argmax(pytorch_outputs.logits, dim=1).item()

        # 比较两个模型的预测结果
        # print(onnx_predicted_label, pytorch_predicted_label, labels[i])
        if onnx_predicted_label == pytorch_predicted_label:
            consistent_predictions += 1
        if onnx_predicted_label == labels[i] + 1:
            right_predictions += 1
            
    consistency_rate = (consistent_predictions / len(images)) * 100
    print(f"Consistent Predictions: {consistent_predictions} / {len(images)}")
    print(f"Model Consistency Rate: {consistency_rate:.2f}%")
    accuracy_rate = (right_predictions / len(images)) * 100
    print(f"Correct Predictions: {right_predictions} / {len(images)}")
    print(f"Model Accuracy Rate: {accuracy_rate:.2f}%")

    # --- 3. 性能评测 (Performance Profiling) ---
    # 创建一个符合模型输入的随机虚拟数据用于性能测试
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # 热身运行
    print(f"\nPerforming {WARMUP_RUNS} warm-up runs...")
    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy_input})
    print("Warm-up complete.")

    # 计时运行
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
        session.run(None, {input_name: dummy_input})
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


def safe_exit(code=0):
    """安全退出函数，避免在Linux上因清理冲突而崩溃"""
    try:
        # 尝试进行一些清理操作
        print("Performing safe exit...")
        # 例如，关闭打开的文件、释放资源等
    except Exception as e:
        print(f"Error during safe exit: {e}")
    finally:
        os._exit(code)  # 强制退出


if __name__ == "__main__":
    try:
        profile_onnx()
    finally:
        print("\nProfiling finished. Exiting safely.")
        safe_exit(0)