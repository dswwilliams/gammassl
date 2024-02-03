import torch


def get_gamma_seg_masks(seg_masks, gamma):
    """
    Calculate segmentation masks where the last class channel is gamma.

    Args:
        seg_masks (torch.Tensor): Segmentation masks [bs, K, h, w]
        gamma (float): Confidence threshold
    Returns:
        gamma_seg_masks (torch.Tensor): Segmentation masks with gamma concatenated [bs, K+1, h, w]
    """
    bs, _, h, w = seg_masks.shape
    device = seg_masks.device
    gammas = gamma * torch.ones(bs, 1, h, w).to(device)      # shape: [bs, 1, h, w]
    gamma_seg_masks = torch.cat((seg_masks, gammas), dim=1)             # shape: [bs, K+1, h, w]
    return gamma_seg_masks


def get_gamma_masks(seg_masks, gamma):
    """
    Calculates gamma masks, which are 1 where gamma is greater than the max class confidence, else 0.
    i.e. gamma_masks are binary uncertainty masks.

    Args:
        seg_masks (torch.Tensor): Segmentation masks [bs, K, h, w]
        gamma (float): Confidence threshold
    Returns:
        gamma_masks (torch.Tensor): Masks, where mask is 1
    """
    num_known_classes = seg_masks.shape[1]
    gamma = gamma.detach()
    gamma_seg_masks = get_gamma_seg_masks(seg_masks, gamma)
    gamma_masks = torch.eq(torch.argmax(gamma_seg_masks, dim=1), num_known_classes).float()
    return gamma_masks
