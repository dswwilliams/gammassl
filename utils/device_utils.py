import torch

def to_device(tensor, device):
    if torch.cuda.is_available():
        tensor = tensor.to(device=device, non_blocking=True)
    return tensor

def init_device(gpu_no="0", use_cpu=False):
    # if available (and not overwridden by opt.use_cpu) use GPU, else use CPU
    device_id = "cuda:" + gpu_no if torch.cuda.is_available() and not use_cpu else "cpu"        
    print("Device: ", device_id)
    return torch.device(device_id)
    
