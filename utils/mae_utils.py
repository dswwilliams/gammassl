
import torch
from collections import OrderedDict

def convert_beit(ckpt):
    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('blocks'):
            new_key = k.replace('blocks', 'layers')
            if 'norm' in new_key:
                new_key = new_key.replace('norm', 'ln')
            elif 'mlp.fc1' in new_key:
                new_key = new_key.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in new_key:
                new_key = new_key.replace('mlp.fc2', 'ffn.layers.1')
            new_ckpt[new_key] = v
        elif k.startswith('patch_embed'):
            new_key = k.replace('patch_embed.proj', 'patch_embed.projection')
            new_ckpt[new_key] = v
        else:
            new_key = k
            new_ckpt[new_key] = v

    return new_ckpt

