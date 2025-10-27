"""
示例：将 NumPy 数组导出为多种文件，便于 C++ 程序读取
生成文件：
 - data.npy        : 单个 NumPy 数组（标准 .npy）
 - data.npz        : 多个数组压缩包（.npz）
 - data.bin        : 原始二进制（C-order, float32）
 - data_meta.json  : 原始二进制对应的元数据（dtype, shape）

用法:
    python export_numpy.py

可修改脚本生成你自己的数组或从 Hugging Face 数据集中读取并保存。
"""

import numpy as np
import json

def main():
    # 示例：生成 10 张随机图像 (N, C, H, W)
    N, C, H, W = 10, 3, 224, 224
    images = (np.random.rand(N, C, H, W).astype(np.float32) * 255.0).astype(np.float32)
    labels = np.arange(N, dtype=np.int32)

    # 1) 保存单个 .npy
    np.save("data.npy", images)
    print("Wrote data.npy -> shape", images.shape, "dtype", images.dtype)

    # 2) 保存 .npz（压缩包，包含多个数组）
    np.savez_compressed("data.npz", images=images, labels=labels)
    print("Wrote data.npz (images, labels)")

    # 3) 保存原始二进制文件（float32 C 行主序），并写元数据
    images.astype(np.float32).tofile("data.bin")
    meta = {
        "dtype": "float32",
        "shape": list(images.shape),
        "order": "C"
    }
    with open("data_meta.json", "w") as f:
        json.dump(meta, f)
    print("Wrote data.bin and data_meta.json")

    # 4) 也保存 labels 为独立的二进制（可选）
    labels.astype(np.int32).tofile("labels.bin")
    with open("labels_meta.json", "w") as f:
        json.dump({"dtype":"int32","shape":list(labels.shape)}, f)
    print("Wrote labels.bin and labels_meta.json")

if __name__ == '__main__':
    main()
