import tensorrt as trt
import numpy as np
import os
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA

# 定义一个简单的日志记录器
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class Calibrator(trt.IInt8EntropyCalibrator2):
    """
    用于 TensorRT INT8 量化的校准器。
    """
    def __init__(self, calibration_data, cache_file="calibration.cache", batch_size=32):
        """
        初始化校准器。

        Args:
            calibration_data (np.ndarray): 用于校准的 NumPy 数组 (NCHW 格式)。
            cache_file (str): 校准缓存文件的路径。
            batch_size (int): 校准期间使用的批量大小。
        """
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.data = np.ascontiguousarray(calibration_data) # 确保数据是连续的
        self.current_index = 0
        
        # 分配 GPU 内存
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        """
        获取一个批次的校准数据。
        TensorRT 会调用此方法来获取数据。
        """
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None  # 所有数据都已使用

        # 复制当前批次的数据到 GPU
        batch = self.data[self.current_index : self.current_index + self.batch_size]
        cuda.memcpy_htod(self.device_input, batch)
        
        self.current_index += self.batch_size
        print(f"Calibrating with batch {self.current_index // self.batch_size}...")
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """
        读取校准缓存。
        如果缓存文件存在，则直接加载，跳过校准过程。
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using existing calibration cache")
                return f.read()

    def write_calibration_cache(self, cache):
        """
        写入校准缓存。
        将生成的校准表保存到文件，以备将来使用。
        """
        with open(self.cache_file, "wb") as f:
            print("Saving calibration cache")
            f.write(cache)
            
    def free(self):
        """
        释放 GPU 内存。
        """
        self.device_input.free()
        
def build_int8_engine(onnx_model_path, engine_path, calibration_data):
    """构建INT8 TensorRT引擎（适用于TensorRT 8.5+）"""
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    
    # 创建网络 - 使用 EXPLICIT_BATCH 标志
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    # 创建配置
    config = builder.create_builder_config()
    
    # 设置INT8模式（新API）
    if hasattr(config, 'set_flag'):
        config.set_flag(trt.BuilderFlag.INT8)
    else:
        # 兼容旧版本
        config.flags = 1 << int(trt.BuilderFlag.INT8)
    
    # 解析ONNX模型
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("Failed to parse the ONNX model.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 获取输入信息
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    print(f"Input name: {input_name}, shape: {input_tensor.shape}")
    
    # 设置优化配置文件（处理动态批次）
    profile = builder.create_optimization_profile()
    
    # 设置动态形状范围
    batch_size = 1
    min_shape = (batch_size, 3, 224, 224)
    opt_shape = (batch_size, 3, 224, 224)
    max_shape = (batch_size * 2, 3, 224, 224)  # 允许最大批次为2
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    # 新API添加优化配置文件
    if hasattr(config, 'add_optimization_profile'):
        config.add_optimization_profile(profile)
    else:
        # 旧版本兼容
        config.add_optimization_profile(profile)
    
    # 设置工作空间（新API）
    if hasattr(config, 'set_memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    elif hasattr(config, 'max_workspace_size'):
        # 旧版本
        config.max_workspace_size = 1 << 30
    else:
        print("Warning: Could not set workspace size")
    
    # 设置INT8校准器（新API）
    calibrator = Calibrator(calibration_data)
    
    # 检查设置校准器的方法
    if hasattr(config, 'set_calibration_profile'):
        # TensorRT 10.0+ 方式
        config.set_calibration_profile(profile)
        config.int8_calibrator = calibrator
    elif hasattr(config, 'int8_calibrator'):
        # TensorRT 8.x-9.x 方式
        config.int8_calibrator = calibrator
    else:
        print("Warning: Could not set INT8 calibrator")
    
    # 构建引擎
    print("Building INT8 engine...")
    
    # 新API构建方式
    if hasattr(builder, 'build_serialized_network'):
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("Failed to build serialized engine")
            return None
        
        # 保存引擎
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        # 从序列化数据创建引擎
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    else:
        # 旧API构建方式
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to build engine")
            return None
        
        # 保存引擎
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
    
    print(f"INT8 engine saved to: {engine_path}")
    return engine

if __name__ == "__main__":
    ONNX_MODEL_PATH = "mobilenetv2.onnx"
    CALIBRATION_DATA_PATH = "data.npz"
    ENGINE_PATH = "mobilenetv2_int8_calibrated.engine"

    # 加载校准数据
    print(f"Loading calibration data from {CALIBRATION_DATA_PATH}...")
    calibration_data = np.load(CALIBRATION_DATA_PATH)['images']
    print(f"Calibration data shape: {calibration_data.shape}")

    # 构建引擎
    build_int8_engine(ONNX_MODEL_PATH, ENGINE_PATH, calibration_data)
