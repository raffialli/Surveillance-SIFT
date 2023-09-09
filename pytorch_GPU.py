import torch

if torch.cuda.is_available():
    print("CUDA is available! PyTorch can use your GPU.")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. PyTorch will use the CPU.")
