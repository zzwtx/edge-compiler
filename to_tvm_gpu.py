import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx
# 导入 transform 模块
from tvm.relax import transform as relax_transform
from tvm.tir import transform as tir_transform

# 定义 ONNX 模型路径和编译后库的保存路径
onnx_model_path = "mobilenetv2.onnx"
lib_path = "mobilenetv2_gpu.so"

# 加载 ONNX 模型
onnx_model = onnx.load(onnx_model_path)

# 定义模型输入
input_name = "input"
input_shape = (1, 3, 224, 224)
shape_dict = {input_name: input_shape}

# 将 ONNX 模型转换为 TVM Relax 的计算图
mod = relax_onnx.from_onnx(onnx_model, shape_dict)

# # 目标和设备
# device = tvm.cuda(0)
# target = tvm.target.Target.from_device(device)

# 定义编译目标
target = tvm.target.Target("cuda", host="llvm")

# 按照开发者建议的顺序应用编译 Pass
# 1. LegalizeOps: 将高层算子转换为底层 TIR 调用
print("应用 LegalizeOps Pass...")
mod = relax_transform.LegalizeOps()(mod)

# 2. DefaultGPUSchedule: 为 TIR 函数应用默认的 GPU 调度
print("应用 DefaultGPUSchedule Pass...")
with tvm.target.Target("cuda", host="llvm"):
    mod = tir_transform.DefaultGPUSchedule()(mod)

print("开始使用 Relax 编译模型...")

# 3. Build: 使用 PassContext 进行最终优化和编译
with tvm.transform.PassContext(opt_level=3):
    exec = relax.build(mod, target=target)

# 从 Executable 中获取编译好的库并导出
lib = exec.mod
lib.export_library(lib_path)

print(f"模型已成功编译并保存到: {lib_path}")
