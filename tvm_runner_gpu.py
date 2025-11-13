"""
TVM 运行程序 - 加载编译后的模型并进行推理测试

功能:
  1. 加载编译后的 mobilenetv2_gpu.so
  2. 从 data.npz 加载 100 张预处理图片和标签
  3. 运行推理并计算准确率（Top-1, Top-5）
  4. 性能基准测试（延迟、吞吐量）
  5. 详细的结果分析和报告
"""

import numpy as np
import tvm
from tvm import relax
import os
import time
from typing import Tuple, Dict, List
import json

# ============================================================================
# 配置参数
# ============================================================================

LIB_PATH = "mobilenetv2_gpu_fixed.so"
DATA_PATH = "data.npz"
BATCH_SIZE = 1
INPUT_NAME = "input"
NUM_CLASSES = 1000
DEVICE = tvm.device("cuda", 0)

# ============================================================================
# 数据加载和预处理
# ============================================================================

def load_data(data_path: str, max_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 npz 文件加载图片和标签
    
    Args:
        data_path: npz 文件路径
        max_samples: 最多加载的样本数
        
    Returns:
        (images, labels) - 图片数据和标签
    """
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载 npz 文件
    data = np.load(data_path)
    print(f"NPZ 文件中的键: {list(data.keys())}")
    
    # 提取图片和标签
    if 'images' not in data:
        raise KeyError("'images' 键不存在在 npz 文件中")
    if 'labels' not in data:
        raise KeyError("'labels' 键不存在在 npz 文件中")
    
    images = data['images']
    labels = data['labels']
    
    print(f"原始图片形状: {images.shape}")
    print(f"原始标签形状: {labels.shape}")
    print(f"图片数据类型: {images.dtype}")
    print(f"标签数据类型: {labels.dtype}")
    
    # 限制样本数量
    num_samples = min(len(images), max_samples)
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    print(f"\n使用的图片数: {num_samples}")
    print(f"图片范围: min={images.min():.3f}, max={images.max():.3f}")
    print(f"图片均值: {images.mean():.3f}, 标准差: {images.std():.3f}")
    
    # 确保数据类型正确
    if images.dtype != np.float32:
        images = images.astype(np.float32)
        print("图片转换为 float32")
    
    if labels.dtype != np.int64:
        labels = labels.astype(np.int64)
        print("标签转换为 int64")
    
    print(f"\n最终图片形状: {images.shape}, 类型: {images.dtype}")
    print(f"最终标签形状: {labels.shape}, 类型: {labels.dtype}")
    
    return images, labels


# ============================================================================
# 模型加载和推理
# ============================================================================

def load_compiled_model(lib_path: str):
    """
    加载编译后的 TVM 模型
    
    Args:
        lib_path: .so 库文件路径
        
    Returns:
        VirtualMachine 执行器
    """
    print("\n" + "="*60)
    print("加载编译后的模型...")
    print("="*60)
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"库文件不存在: {lib_path}")
    
    # 加载库
    lib = tvm.runtime.load_module(lib_path)
    
    # 创建虚拟机
    vm = relax.VirtualMachine(lib, DEVICE)
    
    print(f"✓ 模型加载成功")
    print(f"  库文件: {lib_path}")
    print(f"  设备: {DEVICE}")
    
    return vm


def run_inference(vm, images: np.ndarray) -> np.ndarray:
    """
    运行批量推理
    
    Args:
        vm: TVM VirtualMachine
        images: 输入图片数据 (N, 3, 224, 224)
        
    Returns:
        输出 logits (N, 1000)
    """
    num_images = len(images)
    all_outputs = []
    
    print(f"\n运行推理 ({num_images} 张图片)...")
    
    # 逐个处理图片
    for i in range(num_images):
        # 准备单张图片
        img_batch = images[i:i+1]  # (1, 3, 224, 224)
        
        # 推理 - 使用有状态 API: set_input + invoke_stateful + get_outputs
        vm.set_input("main", img_batch)
        vm.invoke_stateful("main")
        output_nd = vm.get_outputs("main")
        
        # 转换为 numpy
        if hasattr(output_nd, 'numpy'):
            output = output_nd.numpy()
        else:
            output = np.asarray(output_nd)
        
        all_outputs.append(output)
        
        if (i + 1) % 20 == 0:
            print(f"  已完成: {i+1}/{num_images}")
    
    # 合并所有输出
    all_outputs = np.concatenate(all_outputs, axis=0)
    
    print(f"✓ 推理完成")
    print(f"  输出形状: {all_outputs.shape}")
    print(f"  输出范围: min={all_outputs.min():.3f}, max={all_outputs.max():.3f}")
    
    return all_outputs


# ============================================================================
# 精度评估
# ============================================================================

def compute_accuracy(outputs: np.ndarray, labels: np.ndarray, k: int = 1) -> Tuple[float, np.ndarray]:
    """
    计算 Top-K 准确率
    
    Args:
        outputs: 模型输出 (N, 1000)
        labels: 真实标签 (N,)
        k: Top-K 的 K 值
        
    Returns:
        (准确率百分比, 预测类别)
    """
    predictions = np.argsort(outputs, axis=1)[:, -k:][:, ::-1]  # Top-K 类别
    
    # 检查真实标签是否在 Top-K 中
    matches = np.any(predictions == labels.reshape(-1, 1) + 1, axis=1)
    accuracy = np.mean(matches) * 100
    
    return accuracy, predictions


def analyze_results(outputs: np.ndarray, labels: np.ndarray, predictions_top1: np.ndarray) -> Dict:
    """
    分析推理结果
    
    Args:
        outputs: 模型输出
        labels: 真实标签
        predictions_top1: Top-1 预测
        
    Returns:
        分析结果字典
    """
    results = {}
    
    # 准确率
    top1_acc, _ = compute_accuracy(outputs, labels, k=1)
    top5_acc, _ = compute_accuracy(outputs, labels, k=5)
    
    results['top1_accuracy'] = top1_acc
    results['top5_accuracy'] = top5_acc
    
    # 置信度分析
    max_scores = np.max(outputs, axis=1)
    results['confidence_mean'] = float(np.mean(max_scores))
    results['confidence_min'] = float(np.min(max_scores))
    results['confidence_max'] = float(np.max(max_scores))
    results['confidence_std'] = float(np.std(max_scores))
    
    # 错误分析
    top1_predictions = np.argmax(outputs, axis=1)
    is_correct = top1_predictions == labels
    
    results['correct_count'] = int(np.sum(is_correct))
    results['error_count'] = int(np.sum(~is_correct))
    results['total_count'] = len(labels)
    
    # 类别分布分析
    unique_labels, counts = np.unique(labels, return_counts=True)
    results['num_classes_in_batch'] = len(unique_labels)
    results['labels_distribution'] = {
        int(label): int(count) 
        for label, count in zip(unique_labels, counts)
    }
    
    return results


# ============================================================================
# 性能基准测试
# ============================================================================

def benchmark_inference(vm, images: np.ndarray, num_warmup: int = 50, num_runs: int = 100) -> Dict:
    """
    基准测试推理性能
    
    Args:
        vm: VirtualMachine
        images: 输入数据
        num_warmup: 预热次数 (默认50)
        num_runs: 测试运行次数 (默认100)
        
    Returns:
        性能指标字典
    """
    print("\n" + "="*60)
    print("性能基准测试...")
    print("="*60)
    
    # 准备数据 - 生成150张随机图片用于性能测试
    print("生成150张随机图片进行基准测试...")
    random_images = np.random.randn(150, 3, 224, 224).astype(np.float32)
    print(f"✓ 随机图片生成成功，形状: {random_images.shape}")
    
    # 预热
    print(f"预热 ({num_warmup} 次)...")
    for idx in range(num_warmup):
        test_input = random_images[idx % 150:idx % 150 + 1]
        vm.set_input("main", test_input)
        vm.invoke_stateful("main")
        _ = vm.get_outputs("main")
        
        if (idx + 1) % 10 == 0:
            print(f"  预热进度: {idx+1}/{num_warmup}")
    
    # 同步 GPU
    DEVICE.sync()
    
    # 测试
    print(f"运行 {num_runs} 次推理...")
    times = []
    
    for i in range(num_runs):
        # 轮流使用150张随机图片
        test_input = random_images[i % 150:i % 150 + 1]
        
        start = time.time()
        vm.set_input("main", test_input)
        vm.invoke_stateful("main")
        _ = vm.get_outputs("main")
        DEVICE.sync()
        end = time.time()
        times.append((end - start) * 1000)  # 转为毫秒
        
        if (i + 1) % 20 == 0:
            print(f"  完成: {i+1}/{num_runs}")

    return {'mean_latency_ms': np.mean(times),
            'median_latency_ms': np.median(times),
            'std_latency_ms': np.std(times),
            'throughput_fps': 1000.0 / np.mean(times)
            }

# ============================================================================
# 结果报告
# ============================================================================

def print_detailed_results(results: Dict, performance: Dict, num_samples: int = 10):
    """
    打印详细的结果报告
    
    Args:
        results: 精度分析结果
        performance: 性能指标
        num_samples: 显示的样本数
    """
    print("\n" + "="*60)
    print("详细结果报告")
    print("="*60)
    
    # 精度指标
    print("\n【精度指标】")
    print(f"  Top-1 准确率: {results['top1_accuracy']:.2f}%")
    print(f"  Top-5 准确率: {results['top5_accuracy']:.2f}%")
    print(f"  总样本数:     {results['total_count']}")
    print(f"  正确预测:     {results['correct_count']} ({results['correct_count']/results['total_count']*100:.2f}%)")
    print(f"  错误预测:     {results['error_count']} ({results['error_count']/results['total_count']*100:.2f}%)")
    
    # 置信度分析
    print("\n【置信度分析】")
    print(f"  平均置信度: {results['confidence_mean']:.4f}")
    print(f"  最小置信度: {results['confidence_min']:.4f}")
    print(f"  最大置信度: {results['confidence_max']:.4f}")
    print(f"  标准差:     {results['confidence_std']:.4f}")
    
    # 类别分布
    print(f"\n【类别分布】")
    print(f"  批次中的类别数: {results['num_classes_in_batch']}")
    
    # 性能指标
    print("\n【性能指标】")
    print(f"  平均延迟:   {performance['mean_latency_ms']:.2f} ms")
    print(f"  中位延迟:   {performance['median_latency_ms']:.2f} ms")
    print(f"  标准差:     {performance['std_latency_ms']:.2f} ms")
    print(f"  吞吐量:     {performance['throughput_fps']:.2f} FPS")


def save_results_json(results: Dict, performance: Dict, output_path: str = "tvm_inference_results.json"):
    """
    保存结果为 JSON 文件
    
    Args:
        results: 精度分析结果
        performance: 性能指标
        output_path: 输出文件路径
    """
    combined_results = {
        'accuracy': {
            'top1': results['top1_accuracy'],
            'top5': results['top5_accuracy'],
            'correct_count': results['correct_count'],
            'error_count': results['error_count'],
            'total_count': results['total_count'],
        },
        'confidence': {
            'mean': results['confidence_mean'],
            'min': results['confidence_min'],
            'max': results['confidence_max'],
            'std': results['confidence_std'],
        },
        'performance': performance,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\n✓ 结果已保存到: {output_path}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    try:
        # 加载数据
        images, labels = load_data(DATA_PATH, max_samples=100)
        
        # 加载模型
        vm = load_compiled_model(LIB_PATH)
        
        # 运行推理
        outputs = run_inference(vm, images)
        # outputs = [[0]*1000]*100  # --- IGNORE ---
        
        # 分析结果
        print("\n" + "="*60)
        print("计算精度指标...")
        print("="*60)
        results = analyze_results(outputs, labels, np.argmax(outputs, axis=1))
        
        # 性能基准测试（使用150张随机图片，50次预热，100次正式测试）
        performance = benchmark_inference(vm, images, num_warmup=50, num_runs=100)
        
        # 打印详细报告
        print_detailed_results(results, performance)
        
        # 保存结果
        save_results_json(results, performance)
        
        # 最终总结
        print("\n" + "="*60)
        print("✓ 推理测试完成!")
        print("="*60)
        print(f"\n关键指标总结:")
        print(f"  • Top-1 准确率: {results['top1_accuracy']:.2f}%")
        print(f"  • Top-5 准确率: {results['top5_accuracy']:.2f}%")
        print(f"  • 平均延迟: {performance['mean_latency_ms']:.2f} ms")
        print(f"  • 吞吐量: {performance['throughput_fps']:.2f} FPS")
        
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
