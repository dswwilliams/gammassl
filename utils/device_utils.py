import torch

def to_device(tensor, device):
    """ Move tensor to device, with non_blocking=True. """
    return tensor.to(device=device, non_blocking=True)

def init_device(gpu_no="0", use_cpu=False):
    """ Returns the device to be used for training/testing. """
    # if available (and not overwridden by opt.use_cpu) use GPU, else use CPU
    device_id = "cuda:" + gpu_no if torch.cuda.is_available() and not use_cpu else "cpu"        
    print("Device: ", device_id)
    return torch.device(device_id)
    
