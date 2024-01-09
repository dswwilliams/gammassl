import torch
import pickle
import io



class CPU_Unpickler(pickle.Unpickler):
    """
    - deals with issue of pickling device on cuda
    """
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def read_from_pickle(path):
    try:
        while True:
            yield CPU_Unpickler(open(path, 'rb')).load()
    except EOFError:
        pass



proto_save_path = "/Volumes/scratchdata/dw/gamma_star_gbranch_t/prototypes_6.pkl"


def load_prototypes_from_pkl(proto_save_path, device):
    prototypes = next(read_from_pickle(proto_save_path))
    if device is not None:
        prototypes = prototypes.to(device)
    return prototypes