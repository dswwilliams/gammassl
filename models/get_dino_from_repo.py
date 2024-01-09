import torch

import sys
sys.path.append("/Users/dw/code/pytorch/dinov2")
import math
import torch.nn as nn


def interpolate_pos_encoding(pos_embed, x, w, h, patch_size):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = pos_embed.shape[1] - 1
    if npatch == N and w == h:
        return pos_embed
    pos_embed = pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


def get_repo_dino(dino_path="/Users/dw/networks/dinov2/dinov2.pth", dino_repo_path="/Users/dw/code/pytorch/dinov2", lora_rank=None, vit_size="small"):
    sys.path.append(dino_repo_path)

    if vit_size == "small":
        from dinov2.models.vision_transformer import vit_small
        dino_repo = vit_small(
                    patch_size=14,
                    init_values=0.25,
                    lora_rank=lora_rank,
                    )
    elif vit_size == "base":
        from dinov2.models.vision_transformer import vit_base
        dino_repo = vit_base(
                    patch_size=14,
                    init_values=0.25,
                    lora_rank=lora_rank,
                    )


    hub_state_dict = torch.load(dino_path, map_location="cpu")


    # interpolating pos_embed for 224x224, patch_size=14
    pos_embed = hub_state_dict["pos_embed"]
    pos_embed = interpolate_pos_encoding(pos_embed, torch.randn(1, 257, dino_repo.embed_dim), 224, 224, 14)
    hub_state_dict["pos_embed"] = pos_embed


    # refining keys
    new_hub_state_dict = {}
    for key in hub_state_dict:
        if "blocks" in key:
            orig_key = key
            key = key.split(".")
            key.insert(1, "0")
            key = ".".join(key)
            new_hub_state_dict[key] = hub_state_dict[orig_key]
        else:
            new_hub_state_dict[key] = hub_state_dict[key]


    # loading hub state dict to repo vit
    a = dino_repo.load_state_dict(new_hub_state_dict, strict=False)
    return dino_repo




if __name__ == "__main__":
    dino_repo = get_repo_dino()
    x = torch.randn(1, 3, 224, 224)
    y1 = dino_repo.forward_features(x)
