import numpy as np
from datasets import load_dataset
from itertools import islice
from transformers import AutoImageProcessor

# --- 1b. 加载数据并预处理 ---
print("\nLoading and preprocessing 100 images from ImageNet-1k...")
# 使用 streaming=True 避免下载整个数据集
dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)

images = []
labels = []
# 使用 islice 高效地只获取前100个样本
for sample in islice(dataset, 100):
    # 确保所有图片都是三通道的 RGB 格式
    images.append(sample["image"].convert("RGB"))
    labels.append(sample["label"])

print(f"Loaded {len(images)} images and labels.")

# 加载与模型匹配的预处理器
# 我们从本地加载，确保与转换模型时使用的预处理器完全一致
processor = AutoImageProcessor.from_pretrained("/home/zzwtx/fl/edge/model_files")

# 批量预处理所有图片，返回 NumPy 数组
pixel_values = processor(images=images, return_tensors="np")["pixel_values"]
print(f"Images preprocessed into tensor with shape: {pixel_values.shape}")

np.savez_compressed("data.npz", images=pixel_values, labels=np.array(labels))
print("Wrote data.npz (images, labels)")

# --- Crash Workaround ---
# Explicitly delete large objects and run garbage collection
import os
import gc
del images
del labels
del pixel_values
del processor
del dataset
gc.collect()

# Force exit to avoid crash during interpreter shutdown
print("Script finished, exiting forcefully to prevent potential crash.")
os._exit(0)
