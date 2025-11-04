"""
TensorRT 量化方法快速对比工具
支持：FP32, FP16, INT8, 混合精度等
"""

import tensorrt as trt
import numpy as np
import time
from pathlib import Path
from typing import Dict, Tuple
import json


class QuantizationBenchmark:
    """量化方法性能对比工具"""
    
    def __init__(self, onnx_model_path: str, logger_level=trt.Logger.WARNING):
        self.onnx_model = onnx_model_path
        self.logger = trt.Logger(logger_level)
        self.results = {}
    
    def _create_builder_and_network(self):
        """创建 Builder 和 Network"""
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)
        
        with open(self.onnx_model, 'rb') as f:
            if not parser.parse(f.read()):
                raise RuntimeError("Failed to parse ONNX model")
        
        return builder, network
    
    def build_fp32_engine(self, output_path: str = None) -> str:
        """构建 FP32 引擎"""
        print("\n" + "="*70)
        print("📊 构建 FP32 引擎（基准）")
        print("="*70)
        
        builder, network = self._create_builder_and_network()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        start_time = time.time()
        engine = builder.build_serialized_network(network, config)
        build_time = time.time() - start_time
        
        if output_path is None:
            output_path = "model_fp32.engine"
        
        with open(output_path, 'wb') as f:
            f.write(engine)
        
        engine_size = Path(output_path).stat().st_size / (1024**2)
        
        print(f"✓ 引擎大小: {engine_size:.2f} MB")
        print(f"✓ 构建时间: {build_time:.2f}s")
        
        self.results['FP32'] = {
            'path': output_path,
            'size_mb': engine_size,
            'build_time': build_time
        }
        
        return output_path
    
    def build_fp16_engine(self, output_path: str = None) -> str:
        """构建 FP16 引擎"""
        print("\n" + "="*70)
        print("📊 构建 FP16 引擎（半精度）")
        print("="*70)
        
        builder, network = self._create_builder_and_network()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        if not builder.platform_has_fast_fp16:
            print("⚠ 警告：硬件不支持快速 FP16")
        else:
            print("✓ 硬件支持快速 FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        
        start_time = time.time()
        engine = builder.build_serialized_network(network, config)
        build_time = time.time() - start_time
        
        if output_path is None:
            output_path = "model_fp16.engine"
        
        with open(output_path, 'wb') as f:
            f.write(engine)
        
        engine_size = Path(output_path).stat().st_size / (1024**2)
        
        print(f"✓ 引擎大小: {engine_size:.2f} MB")
        print(f"✓ 构建时间: {build_time:.2f}s")
        
        if 'FP32' in self.results:
            size_reduction = (
                (self.results['FP32']['size_mb'] - engine_size) /
                self.results['FP32']['size_mb'] * 100
            )
            print(f"✓ 大小减小: {size_reduction:.1f}%")
        
        self.results['FP16'] = {
            'path': output_path,
            'size_mb': engine_size,
            'build_time': build_time
        }
        
        return output_path
    
    def build_int8_engine(self, calibration_data: np.ndarray = None,
                         output_path: str = None) -> str:
        """构建 INT8 量化引擎"""
        print("\n" + "="*70)
        print("📊 构建 INT8 量化引擎")
        print("="*70)
        
        if calibration_data is None:
            print("⚠ 未提供校准数据，使用随机数据")
            calibration_data = np.random.randn(100, 3, 224, 224).astype(np.float32)
        
        builder, network = self._create_builder_and_network()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        
        if not builder.platform_has_fast_int8:
            print("⚠ 警告：硬件不支持快速 INT8")
        else:
            print("✓ 硬件支持快速 INT8")
        
        # 创建简单的校准器（使用随机数据）
        class SimpleCalibrator(trt.IInt8MinMaxCalibrator):
            def __init__(self, data):
                super().__init__()
                self.data = data
                self.batch_idx = 0
                
                import pycuda.driver as cuda
                self.device_input = cuda.mem_alloc(data[0].nbytes)
            
            def get_batch_size(self):
                return self.data.shape[0]
            
            def get_batch(self, names, p_input_data):
                if self.batch_idx >= 1:
                    return None
                
                import pycuda.driver as cuda
                cuda.memcpy_htod(self.device_input, self.data[0:1])
                cuda.c_void_p(int(self.device_input))
                self.batch_idx += 1
                
                return [int(self.device_input)]
            
            def read_calibration_cache(self):
                return None
            
            def write_calibration_cache(self, cache):
                pass
        
        try:
            import pycuda.autoinit
            
            calibrator = SimpleCalibrator(calibration_data)
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = calibrator
            config.quantization_flags = trt.QuantizationFlag.CALIBRATE_BEFORE_ACTIVATION
            
            print("✓ INT8 校准器已设置")
            
        except Exception as e:
            print(f"⚠ 校准器设置失败: {e}")
            print("  将使用动态范围量化")
            config.set_flag(trt.BuilderFlag.INT8)
        
        start_time = time.time()
        engine = builder.build_serialized_network(network, config)
        build_time = time.time() - start_time
        
        if output_path is None:
            output_path = "model_int8.engine"
        
        with open(output_path, 'wb') as f:
            f.write(engine)
        
        engine_size = Path(output_path).stat().st_size / (1024**2)
        
        print(f"✓ 引擎大小: {engine_size:.2f} MB")
        print(f"✓ 构建时间: {build_time:.2f}s")
        
        if 'FP32' in self.results:
            size_reduction = (
                (self.results['FP32']['size_mb'] - engine_size) /
                self.results['FP32']['size_mb'] * 100
            )
            print(f"✓ 大小减小: {size_reduction:.1f}%")
        
        self.results['INT8'] = {
            'path': output_path,
            'size_mb': engine_size,
            'build_time': build_time
        }
        
        return output_path
    
    def print_summary(self):
        """打印对比总结"""
        print("\n" + "="*70)
        print("📈 量化方法对比总结")
        print("="*70)
        
        print("\n{:<15} {:<15} {:<15} {:<15}".format(
            "量化方法", "模型大小(MB)", "构建时间(s)", "相对 FP32"
        ))
        print("-"*70)
        
        fp32_size = self.results.get('FP32', {}).get('size_mb', 1)
        
        for method in ['FP32', 'FP16', 'INT8', 'INT4', 'Mixed']:
            if method in self.results:
                result = self.results[method]
                size = result['size_mb']
                build_time = result['build_time']
                ratio = f"{size/fp32_size:.2f}x" if fp32_size > 0 else "N/A"
                
                print("{:<15} {:<15.2f} {:<15.2f} {:<15}".format(
                    method, size, build_time, ratio
                ))
        
        print("\n" + "="*70)
    
    def generate_report(self, output_file: str = "quantization_report.json"):
        """生成报告"""
        report = {
            'onnx_model': self.onnx_model,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': self.results,
            'summary': {
                'total_methods': len(self.results),
                'fp32_baseline': self.results.get('FP32', {}).get('size_mb')
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ 报告已保存: {output_file}")


def create_sample_calibration_data(batch_size: int = 100,
                                  image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """创建样本校准数据"""
    return np.random.randn(batch_size, 3, image_size[0], image_size[1]).astype(np.float32)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    onnx_model = "mobilenetv2.onnx"
    
    # 检查模型是否存在
    if not Path(onnx_model).exists():
        print(f"❌ 模型不存在: {onnx_model}")
        print("请先准备 ONNX 模型文件")
    else:
        # 创建 Benchmark 工具
        benchmark = QuantizationBenchmark(onnx_model)
        
        # 准备校准数据
        print("\n🔄 准备校准数据...")
        calib_data = create_sample_calibration_data(batch_size=100)
        print(f"✓ 校准数据形状: {calib_data.shape}")
        
        # 构建各种量化引擎
        print("\n🔨 开始构建量化引擎...")
        
        try:
            benchmark.build_fp32_engine("mobilenetv2_fp32.engine")
        except Exception as e:
            print(f"❌ FP32 构建失败: {e}")
        
        try:
            benchmark.build_fp16_engine("mobilenetv2_fp16.engine")
        except Exception as e:
            print(f"❌ FP16 构建失败: {e}")
        
        try:
            benchmark.build_int8_engine(calib_data, "mobilenetv2_int8.engine")
        except Exception as e:
            print(f"❌ INT8 构建失败: {e}")
        
        # 打印总结
        benchmark.print_summary()
        
        # 生成报告
        benchmark.generate_report("tensorrt_quantization_report.json")
        
        print("\n✅ 量化对比完成！")
        print("\n💡 建议：")
        print("1. FP16 适合大多数生产环境（精度/性能平衡）")
        print("2. INT8 用于极端性能要求的场景（需要精度验证）")
        print("3. 始终验证量化模型的精度")
        print("4. 使用代表性的真实校准数据")
