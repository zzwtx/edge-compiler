# edge-compiler

## 环境与依赖

本项目包含多种类型的依赖，需要不同的管理方式。详细信息见以下文档：

| 文件 | 内容 | 说明 |
| :--- | :--- | :--- |
| **requirements.txt** | Python包列表 | pip安装的所有Python包及版本 |
| **SYSTEM_DEPENDENCIES.md** | 系统级依赖版本记录 | CUDA、LLVM、GCC等系统工具的实际版本 |
| **BUILD_DEPENDENCIES.md** | 源码编译依赖 | TVM、CNPY等从源代码编译安装的组件 |

## 快速开始

### 前置条件

确保系统中已安装以下工具（版本参考 `SYSTEM_DEPENDENCIES.md`）：
- CUDA 13.0
- LLVM 20.1.2
- CMake 3.28.3
- GCC 13.3.0

### 配置Python虚拟环境

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 激活虚拟环境
source .venv/bin/activate  # Linux/macOS
# 或
.venv\Scripts\activate     # Windows

# 3. 安装Python依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import tvm; print(f'TVM: {tvm.__version__}'); print(f'CUDA available: {tvm.cuda.is_gpu_available()}')"
```

## 环境信息

**当前开发环境版本**:
- **OS**: Ubuntu 24.04 LTS (WSL2)
- **CUDA**: 13.0
- **LLVM**: 20.1.2
- **Python**: 3.12.3
- **CMake**: 3.28.3
- **TensorRT**: 10.13.3.9

详细版本信息见 [`SYSTEM_DEPENDENCIES.md`](./SYSTEM_DEPENDENCIES.md)

**已编译的本地组件**:
- Apache TVM 0.22.0 (with CUDA, LLVM, CUTLASS)
- CNPY (C++ NumPy I/O)
- TensorRT 10.13.3.9

详细编译信息见 [`BUILD_DEPENDENCIES.md`](./BUILD_DEPENDENCIES.md)