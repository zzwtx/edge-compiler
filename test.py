import sys
import os

print("=== TVM 诊断信息 ===")
print("Python版本:", sys.version)
print("Python路径:")
for p in sys.path:
    print(" ", p)

print("\n=== 环境变量 ===")
print("PYTHONPATH:", os.environ.get('PYTHONPATH', '未设置'))
print("TVM_HOME:", os.environ.get('TVM_HOME', '未设置'))

print("\n=== TVM 导入测试 ===")
try:
    import tvm
    print("✓ 成功导入 tvm")
    print("TVM路径:", tvm.__file__)
    print("TVM版本:", tvm.__version__)
    
    print("\n=== TVM 模块内容 ===")
    tvm_modules = [m for m in dir(tvm) if not m.startswith('_')]
    print("可用模块:", tvm_modules)
    
    print("\n=== Relay 导入测试 ===")
    try:
        from tvm import relay
        print("✓ 成功导入 relay")
        print("Relay路径:", relay.__file__)
    except ImportError as e:
        print("✗ 导入 relay 失败:", e)
        
except ImportError as e:
    print("✗ 导入 tvm 失败:", e)

print("\n=== 文件检查 ===")
tvm_paths = []
for p in sys.path:
    potential_tvm = os.path.join(p, 'tvm')
    if os.path.exists(potential_tvm):
        tvm_paths.append(potential_tvm)
        print("找到 tvm 目录:", potential_tvm)
        relay_init = os.path.join(potential_tvm, 'relay', '__init__.py')
        print("  relay/__init__.py 存在:", os.path.exists(relay_init))