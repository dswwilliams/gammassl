import numpy as np
import torch
from kornia.geometry.transform import crop_by_boxes


def get_random_crop_boxes(input_size, min_crop_ratio, max_crop_ratio):
    """
    Outputs tensor that defines the corners of a box, used to crop an image.
    Size of the crop box is randomly chosen in the bounds defined by min_crop_ratio and max_crop_ratio.

    Args:
        input_size: tuple of (h,w) of the input image
        min_crop_ratio: minimum ratio of the input_size to crop
        max_crop_ratio: maximum ratio of the input_size to crop

    Returns:    
        box: tensor of shape (4,2), where each row is a corner of the box
    """
    h,w = input_size
    min_input_size = np.minimum(h,w)
    if (max_crop_ratio == 1):
        box = torch.tensor([[0,0], [min_input_size-1, 0], [min_input_size-1, min_input_size-1], [0, min_input_size-1]])
        return box
    else:
        crop_size = np.random.randint(low=min_input_size//max_crop_ratio, high=min_input_size//min_crop_ratio)

        start_x = np.random.randint(low=0, high=(w - crop_size))
        end_x = start_x + crop_size
        start_y = np.random.randint(low=0, high=(h - crop_size))
        end_y = start_y + crop_size

        box = torch.tensor([[start_x,start_y], [end_x-1, start_y], [end_x-1, end_y-1], [start_x, end_y-1]])
        return box


def crop_by_box_and_resize(imgs, crop_boxes, mode="bilinear"):
    """
    Crop images using crop_boxes and resize them to the same size as crop_boxes.

    Args:
        imgs: tensor of shape (bs, 3, H, W)
        crop_boxes: tensor of shape (bs, 4, 2)
        mode: interpolation mode for resizing

    Returns:
        crops: tensor of shape (bs, 3, H, W)
    """
    bs,_, H, W = imgs.shape
    # crop is resized to same size as imgs
    dst_boxes = torch.tensor([[0,0], [W-1, 0], [W-1, H-1], [0, H-1]]).unsqueeze(0).expand(bs,-1,-1).to(imgs.device)

    crops = crop_by_boxes(imgs, src_box=crop_boxes, dst_box=dst_boxes, align_corners=True, mode=mode)
    return crops
