import numpy as np
import torch
import onnxruntime
from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset
import itertools

def verify_onnx_model():
    """
    Compares the output of the original PyTorch model with the ONNX model
    to verify the conversion correctness.
    """
    # --- 1. Load Models ---
    print("Loading models...")
    # Load original PyTorch model from local files
    local_dir = "/home/zzwtx/fl/edge/model_files"
    pytorch_model = AutoModelForImageClassification.from_pretrained(local_dir)
    processor = AutoImageProcessor.from_pretrained(local_dir)
    pytorch_model.eval()

    # Load ONNX model and create an inference session
    onnx_path = "mobilenetv2.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    # --- 2. Create a common dummy input ---
    print("Creating dummy input...")
    # Determine input size from the processor configuration
    if hasattr(processor, "image_size") and isinstance(processor.image_size, int):
        height = width = processor.image_size
    else:
        height = width = 224 # Fallback to a default size

    # Create a numpy array for ONNX runtime and a torch tensor for PyTorch
    dummy_input_np = np.random.randn(1, 3, height, width).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    # --- 3. Run Inference ---
    print("Running inference...")
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input_torch).logits.numpy()

    # ONNX inference
    ort_output = ort_session.run(None, {input_name: dummy_input_np})[0]

    # --- 4. Compare Outputs ---
    print("Comparing outputs...")
    # Use np.allclose for robust floating-point comparison
    are_close = np.allclose(pytorch_output, ort_output, rtol=5e-2, atol=5e-2)

    if are_close:
        print("\n✅ Verification successful: The outputs of the PyTorch and ONNX models are close.")
    else:
        print("\n❌ Verification failed: The outputs of the models differ significantly.")

    # Print absolute difference for detailed analysis
    abs_diff = np.abs(pytorch_output - ort_output)
    print(f"   - Maximum absolute difference: {np.max(abs_diff)}")
    print(f"   - Mean absolute difference: {np.mean(abs_diff)}")


def verify_onnx_model_with_dataset(num_samples=10):
    """
    使用来自 ImageNet 数据集的真实图像，比较原始 PyTorch 模型和 ONNX 模型的输出。
    这个函数是验证模型转换在真实数据上是否正确的关键步骤。
    """
    # --- 第 1 步：加载模型和预处理器 ---
    # 这一步负责将所有需要用到的组件加载到内存中。
    # 这些对象（特别是模型）会占用大量的 CPU 和 GPU 内存。
    print("Loading models and processor...")
    local_dir = "/home/zzwtx/fl/edge/model_files"
    
    # 加载原始的 PyTorch 模型。
    # from_pretrained 会读取配置文件和权重文件，并在内存中构建模型对象。
    pytorch_model = AutoModelForImageClassification.from_pretrained(local_dir)
    
    # 加载与模型匹配的图像预处理器。
    # 预处理器知道如何将输入的图像转换成模型期望的格式（例如，尺寸、归一化）。
    processor = AutoImageProcessor.from_pretrained(local_dir)
    
    # 将 PyTorch 模型设置为评估模式（evaluation mode）。
    # 这会关闭 Dropout 和 BatchNorm 的训练行为，确保推理结果是确定性的。
    pytorch_model.eval()

    # 加载 ONNX 模型并创建一个 ONNX Runtime 推理会Session。
    # 这是与 ONNX 模型交互的核心对象。
    # `providers` 参数告诉 ONNX Runtime 优先尝试使用 CUDA (GPU)，如果失败则回退到 CPU。
    onnx_path = "mobilenetv2.onnx"
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # 获取 ONNX 模型的输入节点的名称。在运行推理时需要用这个名称来传递输入数据。
    input_name = ort_session.get_inputs()[0].name

    # --- 第 2 步：加载数据集 ---
    # 为了进行真实的验证，我们从 Hugging Face Hub 加载 ImageNet-1k 数据集。
    print(f"Loading {num_samples} samples from ImageNet-1k dataset (streaming)...")
    
    # 使用 `streaming=True` 是一个关键的性能优化。
    # 它使得我们不必下载整个庞大的数据集（几百GB），而是像流媒体一样逐个样本地获取数据。
    dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", streaming=True)
    
    successful_verifications = 0
    
    # --- 第 3 步：迭代、推理和验证 ---
    # 这是函数的核心循环。我们从数据流中取出指定数量的样本进行逐一验证。
    for i, sample in enumerate(itertools.islice(dataset, num_samples)):
        # 从样本中获取图像，并确保它是 RGB 格式。
        image = sample["image"].convert("RGB")
        print(f"\n--- Verifying sample {i+1}/{num_samples} ---")

        # 使用预处理器处理图像。
        # `return_tensors="pt"` 表示我们希望得到 PyTorch 张量（Tensor）。
        inputs = processor(images=image, return_tensors="pt")
        
        # 将 PyTorch 张量分别转换为 PyTorch 和 NumPy 格式，以供两个模型使用。
        pixel_values_torch = inputs['pixel_values']
        pixel_values_np = pixel_values_torch.numpy()

        # **PyTorch 推理**
        # `with torch.no_grad():` 是一个重要的优化。
        # 它关闭了梯度计算，减少了内存使用，并加快了推理速度。
        # 因为在推理时我们不需要反向传播，所以可以安全地关闭梯度。
        with torch.no_grad():
            pytorch_output = pytorch_model(pixel_values_torch).logits.numpy()

        # **ONNX 推理**
        # ONNX 推理的输入是 NumPy 数组，输出也是 NumPy 数组。
        # 这使得我们可以直接将数据传递给 ONNX Runtime 进行推理。
        ort_output = ort_session.run(None, {input_name: pixel_values_np})[0]

        # **比较输出**
        # 我们使用 np.allclose 函数来比较两个浮点数组是否在可接受的误差范围内相等。
        # 这里的相对误差和绝对误差容限都设置为 5e-2，意味着 1% 的相对误差或 0.05 的绝对误差都被认为是可以接受的。
        are_close = np.allclose(pytorch_output, ort_output, rtol=5e-2, atol=5e-2)

        if are_close:
            print(f"✅ Sample {i+1} verification successful.")
            successful_verifications += 1
        else:
            print(f"❌ Sample {i+1} verification failed.")
            abs_diff = np.abs(pytorch_output - ort_output)
            print(f"   - Max absolute difference: {np.max(abs_diff)}")
            print(f"   - Mean absolute difference: {np.mean(abs_diff)}")

    # --- 4. Final Report ---
    print("\n--- Verification Summary ---")
    print(f"Total samples tested: {num_samples}")
    print(f"Successful verifications: {successful_verifications}")
    if successful_verifications == num_samples:
        print("✅ All samples passed verification!")
    else:
        print(f"❌ {num_samples - successful_verifications} sample(s) failed verification.")

    # Explicitly delete large objects to help with cleanup
    print("\nCleaning up resources...")
    del ort_session
    del pytorch_model
    del dataset
    del processor
    
import os
import sys

def safe_exit(code=0):
    """安全退出函数"""
    if sys.platform != 'win32':
        os._exit(code)  # 立即退出，跳过清理过程
    else:
        sys.exit(code)

if __name__ == "__main__":
    try:
        # Ensure you have onnxruntime, datasets, and are logged in to huggingface-cli
        verify_onnx_model_with_dataset()
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        safe_exit(0)