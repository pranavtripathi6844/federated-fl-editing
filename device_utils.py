import torch

def get_device():
    """Get the best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU as fallback)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_autocast_device_type():
    """Get the autocast device type for mixed precision training"""
    if torch.backends.mps.is_available():
        return 'cpu'  # MPS doesn't support autocast yet, fallback to CPU
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
