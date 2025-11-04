# 系统环境版本记录
# 记录当前系统中安装的系统级依赖和开发工具的实际版本
# 更新日期: 2025-11-04

## 操作系统
OS: Ubuntu 24.04 LTS
Kernel: Linux 6.6.87.2-microsoft-standard-WSL2 (WSL2)
Architecture: x86_64

## NVIDIA CUDA & 相关库
CUDA Toolkit: 13.0 (Build V13.0.88)
CUDA_HOME: /usr/local/cuda-13.0
NVIDIA Driver: 显示为支持CUDA 13.0

### CUDA相关库（通过requirements.txt中的nvidia-*包）
- nvidia-cuda-runtime-cu13==0.0.0a0
- nvidia-cuda-nvrtc-cu13
- nvidia-cudnn-cu12==9.10.2.21
- nvidia-nccl-cu12==2.27.5
- 其他CUDA库详见requirements.txt

## 编译工具链
GCC: 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04)
CMake: 3.28.3
Make: GNU Make (通过build-essential安装)

## LLVM / Clang
LLVM: 20.1.2
用途: TVM的CPU代码生成后端

## Python
Python: 3.12.3 (虚拟环境)
虚拟环境路径: /home/zzwtx/fl/edge/.venv

## TensorRT
TensorRT: 10.13.3.9
安装路径: /usr/local/TensorRT-10.13.3.9
工具: /usr/local/TensorRT-10.13.3.9/bin/trtexec

## Apache TVM
版本: 0.22.0
源代码路径: /home/zzwtx/fl/edge/apache-tvm-src-v0.22.0
编译选项: USE_CUDA=ON, USE_LLVM=ON, USE_CUTLASS=ON (预期)
Python绑定: 通过本地路径安装 (file://.../python)

## 深度学习框架
PyTorch: 2.9.0
TorchVision: 0.24.0
ONNX: 1.19.1
ONNX Runtime: 1.23.2 (CPU), 1.20.0 (GPU)

## 其他关键组件
CNPY: C++ NumPy库（本地编译）
位置: /home/zzwtx/fl/edge/cnpy
