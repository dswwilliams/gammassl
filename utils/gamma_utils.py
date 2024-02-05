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
    Calculates gamma_masks, where gamma_masks = 1, when max(seg_masks) < gamma, else 0.
    and gamma_masks = 0 when max(seg_masks) >= gamma,
    i.e. it calculates binary uncertainty masks, where 1 indicates uncertain and 0 indicates certain.

    Args:
        seg_masks (torch.Tensor): Segmentation masks [bs, K, h, w]
        gamma (float): Confidence threshold
    Returns:
        gamma_masks (torch.Tensor): Binary uncertainty masks
    """
    gamma = gamma.detach()
    gamma_masks = torch.lt(torch.max(seg_masks, dim=1).values, gamma).float()
    return gamma_masks

def calculate_threshold(x, num_rejects=None):
    """
    Calculates the threshold such that num_rejects elements of x are < threshold.
    Args:
        x (torch.Tensor): Input tensor
        num_rejects (int): Number of elements to reject
    Returns:
        threshold (float): Threshold value
    """

    x = x.flatten()
    x = torch.sort(x, descending=False).values
    threshold = x[num_rejects]
    return threshold



if __name__ == "__main__":
    x = torch.randn(4, 4)  # Replace with your tensor
    print(x.flatten().sort(descending=False).values)
    threshold = calculate_threshold(x, num_rejects=10)
    print("Threshold:", threshold)

