import onnx
import tvm
from tvm import relax
from tvm.relax.frontend import onnx as relax_onnx
# 导入 transform 模块
from tvm.relax import transform as relax_transform
from tvm.tir import transform as tir_transform
from tvm import dlight as dl

# 定义 ONNX 模型路径和编译后库的保存路径
onnx_model_path = "mobilenetv2.onnx"
lib_path = "mobilenetv2_gpu_dlite.so"

# 加载 ONNX 模型
onnx_model = onnx.load(onnx_model_path)

# 定义模型输入
input_name = "input"
input_shape = (1, 3, 224, 224)
shape_dict = {input_name: input_shape}

# 将 ONNX 模型转换为 TVM Relax 的计算图
result = relax_onnx.from_onnx(onnx_model, shape_dict)

# 处理返回值 - 可能是 (mod, params) 或仅是 mod
if isinstance(result, tuple):
    mod, params = result
else:
    mod = result
    params = None

# 定义编译目标
target = tvm.target.Target("cuda", host="llvm")

# 手动应用优化 Pass
# 1. 绑定符号变量（如果有参数的话）
if params is not None:
    # 获取主函数名称
    main_func_name = list(mod.functions.keys())[0]
    mod = relax.transform.BindParams(main_func_name, params)(mod)

# # 2. 应用图级优化
# # Fusing ops, constant folding, etc.
# seq = tvm.transform.Sequential(
#     [
#         relax_transform.FoldConstant(),
#         relax_transform.EliminateCommonSubexpr(),
#         relax_transform.CanonicalizeBindings(),
#         relax_transform.FuseOps(),
#     ]
# )
# mod = seq(mod)

# # 3. 将高层算子转换为底层 TIR 调用
# print("应用 LegalizeOps Pass...")
# mod = relax_transform.LegalizeOps()(mod)

# 4. 应用 DLight 规则进行底层 TIR 优化
print("应用 DLight 规则...")
with target:
    mod = tvm.ir.transform.Sequential(
        [
            relax.get_pipeline("zero"),
            dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.Reduction(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            ),
        ]
    )(mod)

print("开始使用 Relax 编译模型...")

# 5. Build: 进行最终编译
# 注意：这里不再使用 opt_level=3，因为我们已经手动应用了优化
with tvm.transform.PassContext(opt_level=0):
    exec = relax.build(mod, target=target)

# 从 Executable 中获取编译好的库并导出
lib = exec.mod
lib.export_library(lib_path)

print(f"模型已成功编译并保存到: {lib_path}")
