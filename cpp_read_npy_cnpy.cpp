// 示例：使用 cnpy 库读取 .npy 文件
// 需要安装 cnpy（https://github.com/rogersce/cnpy）
// 编译示例： g++ cpp_read_npy_cnpy.cpp -o read_npy -lcnpy -lz -std=c++17

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include "cnpy.h"

int main() {
    // 定义 .npz 文件的路径。因为可执行文件在 build/ 目录下运行，
    std::string npz_path = "../data.npz";
    try {
        // 所以我们需要使用相对路径 ../ 来访问上级目录中的 data.npz 文件。
        std::cout << "Attempting to load .npz file: " << npz_path << std::endl;

        // 加载 .npz 文件。cnpy::npz_load 返回一个 map，
        // key 是数组名 (如 "images", "labels")，value 是 NpyArray 对象。
        cnpy::npz_t my_npz = cnpy::npz_load(npz_path);

        // --- 1. 加载并检查 "images" 数组 ---
        // 检查 "images" 键是否存在
        if (my_npz.find("images") == my_npz.end()) {
            throw std::runtime_error("Failed to find 'images' array in the .npz file.");
        }
        cnpy::NpyArray images_arr = my_npz["images"];
        
        // 获取数据指针和形状
        float* images_data = images_arr.data<float>();
        std::vector<size_t> images_shape = images_arr.shape;

        std::cout << "\nSuccessfully loaded 'images' array." << std::endl;
        // cnpy::NpyArray does not provide a type_name() method; we requested the data as float above.
        std::cout << "Data type: float" << std::endl;
        std::cout << "Shape: (";
        for(size_t i = 0; i < images_shape.size(); ++i) {
            std::cout << images_shape[i] << (i == images_shape.size() - 1 ? "" : ", ");
        }
        std::cout << ")" << std::endl;
        std::cout << "Total number of elements: " << images_arr.num_vals << std::endl;
        std::cout << "Total number of elements: " << images_arr.num_vals << std::endl;


        // --- 2. 加载并检查 "labels" 数组 ---
        if (my_npz.find("labels") == my_npz.end()) {
            throw std::runtime_error("Failed to find 'labels' array in the .npz file.");
        }
        cnpy::NpyArray labels_arr = my_npz["labels"];
        long* labels_data = labels_arr.data<long>(); // Python 中 int64 对应 C++ 的 long
        
        std::cout << "\nSuccessfully loaded 'labels' array." << std::endl;
        std::cout << "Shape: (" << labels_arr.shape[0] << ")" << std::endl;
        std::cout << "First 5 labels: ";
        for(size_t i = 0; i < 5 && i < labels_arr.num_vals; ++i) {
            std::cout << labels_data[i] << " ";
        }
        std::cout << std::endl;

        // --- 清理 ---
        // my_npz 会在 main 函数结束时自动销毁其内容
        std::cout << "\nCleanup and exit." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nAn error occurred: " << e.what() << std::endl;
        // 检查文件是否存在
        std::ifstream f(npz_path.c_str());
        if (!f.good()) {
            std::cerr << "Error: The file '" << npz_path << "' does not exist or cannot be read." << std::endl;
        }
        return 1;
    }

    return 0;
}
