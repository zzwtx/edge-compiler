import numpy as np
import tvm
from tvm import relax

# --- 配置 ---
# 编译后的模型库路径
lib_path = "mobilenetv2.so"
# 数据集路径
data_path = "data.npz"
# ONNX 模型的输入节点名称 (需要和 to_tvm.py 中一致)
input_name = "input"

# --- 1. 加载 TVM 模型和数据集 ---
print("加载编译好的模型和数据集...")
# 获取 CPU 设备
dev = tvm.cpu(0)
# 加载编译好的模型库，并将其加载到 Relax 虚拟机中
lib = tvm.runtime.load_module(lib_path)
vm = relax.VirtualMachine(lib, dev)

# 加载数据集
with np.load(data_path) as data:
    images = data['images']
    labels = data['labels']

print(f"数据集加载成功，包含 {len(images)} 张图片。")

# --- 2. 运行推理并评估准确率 ---
correct_predictions = 0
total_samples = len(images)

print("开始在 CPU 上运行推理 (使用 Relax VM)...")

for i in range(total_samples):
    # 准备输入数据
    # TVM 需要一个 TVM NDArray 作为输入
    # 注意：需要将数据也移动到目标设备
    input_data = tvm.nd.array(images[i].astype('float32'), dev)
    
    # 使用 Relax VM 运行模型
    # "main" 是 relax.build 默认的主函数名
    output = vm["main"](input_data)
    
    # 计算预测结果
    # 输出的 shape 通常是 (1, num_classes)，我们取 argmax 得到预测的类别索引
    predicted_class = np.argmax(output.numpy())
    
    # 与真实标签比较 (移除了之前+1的修正)
    if predicted_class == labels[i]:
        correct_predictions += 1
    
    # 打印进度
    if (i + 1) % 20 == 0:
        print(f"已处理 {i + 1}/{total_samples} 张图片...")

# --- 3. 计算并打印最终结果 ---
accuracy = (correct_predictions / total_samples) * 100
print("\n--- 推理完成 ---")
print(f"模型在 {total_samples} 张图片上的准确率: {accuracy:.2f}%")
print(f"正确预测数量: {correct_predictions}")
print(f"总样本数量: {total_samples}")
