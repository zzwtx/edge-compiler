import onnx
import numpy as np

def main():
    print('onnx version:', onnx.__version__)
    print('numpy version:', np.__version__)
    arr = np.array([1, 2, 3])
    print('numpy array:', arr)

if __name__ == '__main__':
    main()
