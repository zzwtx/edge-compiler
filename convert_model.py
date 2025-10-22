import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

def convert_to_onnx():
    """
    Downloads a MobileNetV2 model from Hugging Face and converts it to ONNX format.
    """
    # 1. Load the model and feature extractor from Hugging Face
    model_name = "google/mobilenet_v2_1.0_224"
    print(f"Loading model: {model_name}...")
    local_dir = "/home/zzwtx/fl/edge/model_files"
    model = AutoModelForImageClassification.from_pretrained(local_dir)
    processor = AutoImageProcessor.from_pretrained(local_dir)

    # 2. Define a dummy input with the correct dimensions
    # 兼容 image_size、height/width 或默认值
    if hasattr(processor, "image_size"):
        size = processor.image_size
        if isinstance(size, dict):
            height = size.get("height", 224)
            width = size.get("width", 224)
        else:
            height = width = size
    elif hasattr(processor, "size") and "height" in processor.size and "width" in processor.size:
        height = processor.size["height"]
        width = processor.size["width"]
    else:
        height = width = 224  # 默认值
    dummy_input = torch.randn(1, 3, height, width)

    # 3. Set the model to evaluation mode
    model.eval()

    # 4. Define input and output names for the ONNX graph
    input_names = ["input"]
    output_names = ["output"]
    onnx_path = "mobilenetv2.onnx"

    # 5. Export the model to ONNX
    print(f"Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=11,  # A commonly supported opset version
        dynamic_axes={
            'input': {0: 'batch_size'},  # Allow for dynamic batch size
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    print("Model conversion successful!")

if __name__ == "__main__":
    convert_to_onnx()
