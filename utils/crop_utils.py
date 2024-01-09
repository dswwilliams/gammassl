import numpy as np
import torch
from kornia.geometry.transform import crop_by_boxes


def get_random_crop(img, label=None, crop_size=256, start_x=None, start_y=None):
    if start_x is None:
        start_x = np.random.randint(low=0, high=(img.shape[1] - crop_size))
    end_x = start_x + crop_size
    if start_y is None:
        start_y = np.random.randint(low=0, high=(img.shape[0] - crop_size))
    end_y = start_y + crop_size

    crop = img[start_y:end_y, start_x:end_x] # shape: (crop_size, crop_size, 3)
    
    if label is not None:
        label = label[start_y:end_y, start_x:end_x] # shape: (crop_size, crop_size)
        return crop, label
    else:
        return crop


def get_random_crop_boxes(input_size, min_crop_ratio, max_crop_ratio):
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
    bs,_, H, W = imgs.shape
    # crop is resized to same size as imgs
    dst_boxes = torch.tensor([[0,0], [W-1, 0], [W-1, H-1], [0, H-1]]).unsqueeze(0).expand(bs,-1,-1).to(imgs.device)

    crops = crop_by_boxes(imgs, src_box=crop_boxes, dst_box=dst_boxes, align_corners=True, mode=mode)
    return crops
