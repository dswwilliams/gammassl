import torch
import torch.nn.functional as F


def semantic_inference(mask_cls, mask_pred):
    """
    Get semantic segmentation from mask_cls and mask_pred.

    Args:
        mask_cls (torch.Tensor): shape: [bs, num_classes], where mask_cls[i,j] is the class probability
            of the j-th class in image i.
        mask_pred (torch.Tensor): shape: [bs, num_classes, H, W], where mask_pred[i,j] is the predicted
            mask for the j-th class in image i.

    Returns:
        semseg (torch.Tensor): shape: [bs, num_classes, H, W], where semseg[i,j] is the predicted
            segmentation mask for the j-th class in image i.
    """
    mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
    mask_pred = mask_pred.sigmoid()
    seg_mask = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)
    return seg_mask