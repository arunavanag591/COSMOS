import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    print("GPU with CUDA is available.")
else:
    print("GPU with CUDA is not available.")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

import pycuda.driver as cuda

cuda.init()
num_devices = cuda.Device.count()

for i in range(num_devices):
    device = cuda.Device(i)
    print(f"GPU {i}: {device.name()}")
    print(f"  Compute capability: {device.compute_capability()}")
    