import torch

def to_device(tensor, device):
    if torch.cuda.is_available():
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
