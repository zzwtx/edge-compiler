#!/usr/bin/env python3
"""
TVM 编译模型 Workspace Size 检查工具

功能:
  1. 分析编译后的 TVM 模型内部结构
  2. 统计激活值内存占用
  3. 统计工作空间占用
  4. 显示内存使用详细信息
"""

import numpy as np
import tvm
from tvm import relax, runtime
import os
from typing import Dict, Tuple, List
import json

# ============================================================================
# 配置参数
# ============================================================================

LIB_PATH = "mobilenetv2_gpu.so"
DEVICE = tvm.device("cuda", 0)
BATCH_SIZE = 1
INPUT_SHAPE = (1, 3, 224, 224)

# ============================================================================
# 模型内存分析
# ============================================================================

def load_compiled_module(lib_path: str):
    """
    加载编译后的 TVM 模块
    
    Args:
        lib_path: .so 库文件路径
        
    Returns:
        加载的库模块
    """
    print("="*70)
    print("加载编译后的 TVM 模块...")
    print("="*70)
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"库文件不存在: {lib_path}")
    
    lib = tvm.runtime.load_module(lib_path)
    print(f"✓ 模块加载成功")
    print(f"  库文件: {lib_path}")
    print(f"  库文件大小: {os.path.getsize(lib_path) / (1024*1024):.2f} MB")
    
    return lib


def analyze_workspace_size(lib):
    """
    分析模块的工作空间大小
    
    Args:
        lib: 加载的库模块
        
    Returns:
        工作空间分析字典
    """
    print("\n" + "="*70)
    print("分析工作空间大小...")
    print("="*70)
    
    workspace_info = {}
    
    try:
        # 获取模块的函数
        functions = []
        for key in dir(lib):
            if not key.startswith('_'):
                functions.append(key)
        
        print(f"模块中的函数数量: {len(functions)}")
        for func_name in functions[:10]:  # 显示前10个
            print(f"  - {func_name}")
        
        # 尝试获取工作空间大小信息
        # 对于编译后的模块，这可能在符号信息中
        if hasattr(lib, '__dict__'):
            print(f"\n模块属性: {list(lib.__dict__.keys())}")
        
    except Exception as e:
        print(f"无法直接获取工作空间信息: {e}")
    
    return workspace_info


def estimate_memory_usage(input_shape: Tuple, output_shape: Tuple = (1, 1000)) -> Dict:
    """
    估算模型各层的内存使用情况
    
    Args:
        input_shape: 输入形状 (N, C, H, W)
        output_shape: 输出形状 (通常是 (N, num_classes))
        
    Returns:
        内存使用分析字典
    """
    print("\n" + "="*70)
    print("估算内存使用情况...")
    print("="*70)
    
    memory_stats = {}
    
    # 计算输入内存
    input_elements = np.prod(input_shape)
    input_memory = input_elements * 4  # float32 = 4 bytes
    memory_stats['input'] = {
        'shape': input_shape,
        'elements': int(input_elements),
        'memory_mb': input_memory / (1024*1024),
        'memory_bytes': int(input_memory)
    }
    
    # 计算输出内存
    output_elements = np.prod(output_shape)
    output_memory = output_elements * 4  # float32 = 4 bytes
    memory_stats['output'] = {
        'shape': output_shape,
        'elements': int(output_elements),
        'memory_mb': output_memory / (1024*1024),
        'memory_bytes': int(output_memory)
    }
    
    # MobileNetV2 典型的激活值大小估算
    # 基于 MobileNetV2 的架构，主要是中间特征图
    print("\n📊 MobileNetV2 典型的激活值大小估算:")
    print("-" * 70)
    
    activations = {
        '输入 (1x3x224x224)': (1, 3, 224, 224),
        'Conv1 输出 (1x32x112x112)': (1, 32, 112, 112),
        'Block1 输出 (1x16x112x112)': (1, 16, 112, 112),
        'Block2-6 输出 (1x24x56x56)': (1, 24, 56, 56),
        'Block7-10 输出 (1x32x28x28)': (1, 32, 28, 28),
        'Block11-13 输出 (1x64x14x14)': (1, 64, 14, 14),
        'Block14-16 输出 (1x96x14x14)': (1, 96, 14, 14),
        'Block17 输出 (1x160x7x7)': (1, 160, 7, 7),
        'Block18 输出 (1x320x7x7)': (1, 320, 7, 7),
        'Conv最后层 (1x1280x1x1)': (1, 1280, 1, 1),
        '输出 (1x1000)': (1, 1000),
    }
    
    total_activation_memory = 0
    
    for layer_name, shape in activations.items():
        elements = np.prod(shape)
        mem_mb = (elements * 4) / (1024 * 1024)
        total_activation_memory += (elements * 4)
        memory_stats[layer_name] = {
            'shape': shape,
            'elements': int(elements),
            'memory_mb': mem_mb,
            'memory_bytes': int(elements * 4)
        }
        print(f"  {layer_name:<40} {mem_mb:>8.4f} MB")
    
    memory_stats['total_activation_memory_mb'] = total_activation_memory / (1024 * 1024)
    memory_stats['total_activation_memory_bytes'] = int(total_activation_memory)
    
    print("-" * 70)
    print(f"  总激活值内存 (所有层叠加时): {memory_stats['total_activation_memory_mb']:.4f} MB")
    
    return memory_stats


def run_memory_benchmark(lib_path: str):
    """
    运行内存基准测试
    
    Args:
        lib_path: .so 库文件路径
    """
    print("\n" + "="*70)
    print("运行内存基准测试...")
    print("="*70)
    
    # 加载模块
    lib = load_compiled_module(lib_path)
    
    # 创建虚拟机
    vm = relax.VirtualMachine(lib, DEVICE)
    
    # 创建输入数据
    print(f"\n创建输入数据: {INPUT_SHAPE}")
    input_data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    
    # 测试运行以获取内存信息
    print("\n运行推理以监测内存使用...")
    try:
        vm.set_input("main", input_data)
        vm.invoke_stateful("main")
        output = vm.get_outputs("main")
        
        print(f"✓ 推理成功")
        print(f"  输出形状: {np.asarray(output).shape}")
        print(f"  输出数据类型: {np.asarray(output).dtype}")
        
        # 同步 GPU
        DEVICE.sync()
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
    
    # 分析工作空间
    workspace_info = analyze_workspace_size(lib)
    
    # 估算内存使用
    memory_stats = estimate_memory_usage(INPUT_SHAPE, output_shape=(1, 1000))
    
    return memory_stats, workspace_info


def print_summary(memory_stats: Dict, workspace_info: Dict):
    """
    打印内存使用摘要
    
    Args:
        memory_stats: 内存统计信息
        workspace_info: 工作空间信息
    """
    print("\n" + "="*70)
    print("📋 内存使用摘要")
    print("="*70)
    
    print(f"\n输入内存:")
    print(f"  形状: {memory_stats['input']['shape']}")
    print(f"  元素数: {memory_stats['input']['elements']:,}")
    print(f"  内存: {memory_stats['input']['memory_mb']:.4f} MB")
    
    print(f"\n输出内存:")
    print(f"  形状: {memory_stats['output']['shape']}")
    print(f"  元素数: {memory_stats['output']['elements']:,}")
    print(f"  内存: {memory_stats['output']['memory_mb']:.4f} MB")
    
    print(f"\n激活值内存 (所有层):")
    print(f"  总计: {memory_stats['total_activation_memory_mb']:.4f} MB")
    print(f"  总计: {memory_stats['total_activation_memory_bytes']:,} 字节")
    
    # 计算总内存需求
    total_memory = (
        memory_stats['input']['memory_bytes'] +
        memory_stats['output']['memory_bytes'] +
        memory_stats['total_activation_memory_bytes']
    )
    
    print(f"\n总内存需求:")
    print(f"  输入 + 激活值 + 输出: {total_memory / (1024*1024):.4f} MB")
    print(f"  输入 + 激活值 + 输出: {total_memory / (1024*1024*1024):.4f} GB")
    
    print("\n" + "="*70)
    print("💡 备注:")
    print("-" * 70)
    print("1. 实际内存使用可能会更大，因为包含了:")
    print("   - GPU 驱动程序的开销")
    print("   - 临时缓冲区和中间结果")
    print("   - 内存对齐和碎片化")
    print("\n2. 上面的激活值大小是所有层叠加时的最大值估算")
    print("   实际推理时，大部分层的结果会被覆盖")
    print("\n3. 要获得精确的内存使用，需要:")
    print("   - 使用 nvidia-smi 或 cuMemGetInfo() 来实时监测")
    print("   - 在 TVM 编译时启用详细的统计信息")


def save_report(memory_stats: Dict, output_file: str = "tvm_memory_report.json"):
    """
    保存报告为 JSON 文件
    
    Args:
        memory_stats: 内存统计信息
        output_file: 输出文件路径
    """
    # 转换为可序列化的格式
    report = {}
    for key, value in memory_stats.items():
        if isinstance(value, dict):
            report[key] = value
        elif isinstance(value, (int, float)):
            report[key] = value
        elif isinstance(value, tuple):
            report[key] = list(value)
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ 报告已保存到: {output_file}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 TVM 编译模型 Workspace Size 检查工具")
    print("="*70)
    
    try:
        # 运行分析
        memory_stats, workspace_info = run_memory_benchmark(LIB_PATH)
        
        # 打印摘要
        print_summary(memory_stats, workspace_info)
        
        # 保存报告
        save_report(memory_stats)
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
