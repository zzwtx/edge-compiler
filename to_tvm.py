
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx

# 定义 ONNX 模型路径和编译后库的保存路径
onnx_model_path = "mobilenetv2.onnx"
lib_path = "mobilenetv2.so"

# 加载 ONNX 模型
onnx_model = onnx.load(onnx_model_path)

# 定义模型输入，MobileNetV2 通常使用 1x3x224x224 的输入
# 请根据你的模型实际输入名称修改 "input"
input_name = "input"  # ONNX 模型的输入节点名称
input_shape = (1, 3, 224, 224)
shape_dict = {input_name: input_shape}

# 将 ONNX 模型转换为 TVM Relax 的计算图
# Relax 是 TVM Unity 架构中的新一代高级中间表示
mod = relax_onnx.from_onnx(onnx_model, shape_dict)

# 定义编译目标
# 我们将使用 LLVM (CPU) 进行编译，以绕过 GPU 架构兼容性问题
target = tvm.target.Target("llvm")

print("开始使用 Relax 编译模型...")

# 使用 TVM 的 PassContext 进行优化编译
with tvm.transform.PassContext(opt_level=3):
    # relax.build 返回一个 VMExecutable 或 Executable 对象
    exec = relax.build(mod, target=target)

# 从 Executable 中获取编译好的库并导出
lib = exec.mod
lib.export_library(lib_path)

print(f"模型已成功编译并保存到: {lib_path}")
