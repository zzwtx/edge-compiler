import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path):
    # 1. 创建 Builder
    builder = trt.Builder(TRT_LOGGER)
    
    # 2. 创建 Network Definition
    # EXPLICIT_BATCH 标志是现代 TensorRT 的标准做法
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 3. 创建 ONNX Parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 4. 解析 ONNX 模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    print("ONNX model parsed successfully.")
    
    # 5. 创建 Builder Config
    config = builder.create_builder_config()
    # 设置最大工作空间大小（例如 1GB）
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 
    
    # 如果需要 FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 mode enabled.")

    # 6. 构建并序列化引擎
    print("Building TensorRT engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return None
        
    # 7. 保存引擎到文件
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT engine saved to {engine_path}")

if __name__ == "__main__":
    build_engine("mobilenetv2.onnx", "mobilenetv2_fp16.engine")