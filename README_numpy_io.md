快速说明：将 NumPy 导出并在 C++ 中读取

准备：
- Python: numpy
- C++: 可选 cnpy (读取 .npy), nlohmann/json (解析元数据 JSON)

Python 导出示例：

    python export_numpy.py

这会生成：data.npy, data.npz, data.bin, data_meta.json, labels.bin, labels_meta.json

C++ 读取示例：
- 使用 cnpy 读取 .npy：参考 `cpp_read_npy_cnpy.cpp`。链接 `cnpy` 与 `zlib`。
- 使用二进制+元数据读取：参考 `cpp_read_bin.cpp`，需要 `nlohmann/json.hpp` 在 include 路径。

编译示例（最小）：

    g++ cpp_read_bin.cpp -o read_bin -std=c++17 -I/path/to/nlohmann

若要使用 cnpy：

    g++ cpp_read_npy_cnpy.cpp -o read_npy -lcnpy -lz -std=c++17

注意：
- 确保字节序和 dtype 一致（本示例使用 float32, C-order）。
- 对大型数据建议使用 mmap 或 Arrow。
