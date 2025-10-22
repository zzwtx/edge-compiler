try:
    import tvm
    print('tvm imported successfully, version:', tvm.__version__)
except ImportError as e:
    print('Failed to import tvm:', e)
