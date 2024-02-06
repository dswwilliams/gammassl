import torch
import torch.nn.functional as F


def semantic_inference(mask_cls, mask_pred):
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
    return semseg