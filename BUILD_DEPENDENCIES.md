# 源代码编译安装的依赖
# 记录通过源代码编译安装的组件，而非通过pip或apt安装

## Apache TVM v0.22.0

**源代码位置**: `/home/zzwtx/fl/edge/apache-tvm-src-v0.22.0/`

**版本**: 0.22.0

**编译状态**: ✓ 已编译

**编译配置**:
- CUDA 支持: 是 (CUDA 13.0)
- LLVM 支持: 是 (LLVM 20.1.2)
- CUTLASS 支持: 是 (包含在3rdparty中)

**依赖项**:
- CUDA Toolkit 13.0
- LLVM 20.1.2
- CMake 3.28.3
- GCC 13.3.0

**输出产物**:
- 构建目录: `/home/zzwtx/fl/edge/apache-tvm-src-v0.22.0/build/`
- Python绑定: 在 requirements.txt 中注册为 `tvm @ file://...`

**验证方式**:
```bash
python -c "import tvm; print(tvm.__version__); print('CUDA:', tvm.cuda.is_gpu_available())"
```

---

## CNPY (C++ NumPy I/O 库)

**源代码位置**: `/home/zzwtx/fl/edge/cnpy/`

**用途**: 在C++代码中读写NumPy的.npy和.npz文件格式

**编译状态**: ✓ 已编译

**构建信息**:
- 构建目录: `/home/zzwtx/fl/edge/cnpy/build/`
- 生成的库: `libcnpy.a` (静态库) 或 `libcnpy.so` (动态库，取决于编译选项)
- 头文件: `/home/zzwtx/fl/edge/cnpy/cnpy.h`

**依赖项**:
- CMake 3.28.3+
- GCC 13.3.0+
- zlib (用于压缩)

**C++使用示例**:
```cpp
#include "cnpy.h"
// 读取.npz文件
cnpy::npz_t my_npz = cnpy::npz_load("file.npz");
```

---

## TensorRT 10.13.3.9

**安装状态**: ✓ 已安装

**安装路径**: `/usr/local/TensorRT-10.13.3.9/`

**版本**: 10.13.3.9

**关键工具**:
- 推理引擎库: `libnvinfer.so`
- 优化工具: `/usr/local/TensorRT-10.13.3.9/bin/trtexec`
- 插件库: `libnvinfer_plugin.so`

**依赖项**:
- CUDA 13.0
- cuDNN 9.10.2

**Python绑定**:
在 requirements.txt 中通过pip包安装:
- `tensorrt==10.13.3.9`
- `tensorrt_cu13==10.13.3.9`
- 及相关库

---

## CUTLASS (NVIDIA 张量计算库)

**集成方式**: 作为TVM的3rdparty子模块

**位置**: `/home/zzwtx/fl/edge/apache-tvm-src-v0.22.0/3rdparty/cutlass/`

**用途**: 为TVM提供高性能张量计算核心

**编译状态**: 随TVM一起编译

---

## 注意事项

1. **本地路径依赖**: `requirements.txt` 中的以下包使用本地文件路径：
   - `apache-tvm-ffi @ file:///.../3rdparty/tvm-ffi`
   - `tvm @ file:///.../python`
   
   这意味着这些包只能在此系统/环境中工作，无法直接在其他系统复现。

2. **CUDA版本依赖**: 系统中的TVM和TensorRT都是针对CUDA 13.0编译的。
   如果CUDA版本改变，需要重新编译TVM。

3. **源代码改动**: 如果修改了TVM或CNPY的源代码，需要重新编译。
