cmake_minimum_required(VERSION 3.18)
project(EdgeAIProject LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 设置 TensorRT 根目录 ---
# 直接指向我们手动安装的 TensorRT SDK 目录
set(TENSORRT_ROOT /usr/local/TensorRT-10.13.3.9)
message(STATUS "Using TensorRT from: ${TENSORRT_ROOT}")

# 设置 TensorRT 包含目录
# 注意：新版 TensorRT 的头文件在 include/
include_directories(${TENSORRT_ROOT}/include)

# 设置 TensorRT 库目录
# 注意：新版 TensorRT 的库文件在 lib/
link_directories(${TENSORRT_ROOT}/lib)

# 定义 TensorRT 库
set(TENSORRT_LIBRARIES
    nvinfer
    nvonnxparser
    nvinfer_plugin
)

# 查找 CUDA (TensorRT 依赖)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 显式添加 CUDA 库目录到链接器搜索路径
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

# 查找 ZLIB 库 (cnpy 的依赖)
find_package(ZLIB REQUIRED)

# 添加 cnpy 子目录
add_subdirectory(cnpy)

# --- 定义可执行文件 ---

# 1. TensorRT Runner
add_executable(tensorrt_runner TensorRT.cpp)

# 显式地为 tensorrt_runner 添加 cnpy 的头文件目录
target_include_directories(tensorrt_runner PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)

target_link_libraries(tensorrt_runner
    PRIVATE
    cnpy
    ${ZLIB_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    cudart # 明确链接 cuda runtime
)
# 确保 cnpy 在 tensorrt_runner 之前构建
add_dependencies(tensorrt_runner cnpy)

# 2. TensorRT Profiling Runner (用于性能分析)
add_executable(tensorrt_profiling TensorRT_profiling.cpp)
target_include_directories(tensorrt_profiling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
target_link_libraries(tensorrt_profiling
    PRIVATE
    ${TENSORRT_LIBRARIES}
    ${CUDA_LIBRARIES}
    cudart
)

# 3. CNPY Reader Example (如果需要)
add_executable(read_npy cpp_read_npy_cnpy.cpp)
target_link_libraries(read_npy
    PRIVATE
    cnpy
    ${ZLIB_LIBRARIES}
)
add_dependencies(read_npy cnpy)

message(STATUS "Project configuration complete. You can now build with 'make'.")
message(STATUS "Targets are: tensorrt_runner, tensorrt_profiling, read_npy")

# 显式地为 read_npy 添加 cnpy 的头文件目录
target_include_directories(read_npy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cnpy)
