import cv2
import numpy as np
import torch
import torch.nn.functional as F
import random
from functools import partial
import math

import numpy as np
from scipy.ndimage import label

def compute_areas_for_each_value(tensor):
    """
    tensor.shape = [h, w]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.numpy()
    unique_values = np.unique(tensor)    
    result = {}
    for val in unique_values:
        binary_mask = (tensor == val).astype(int)
        
        # Label each connected component (object) for the current value
        labeled, num_features = label(binary_mask)
        
        # Compute area of each labeled region
        areas = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        result[val] = areas

    return result




def calculate_average_blob_area(segs, average_type='mean'):
    """
    segs is numpy array or tensor of shape [bs, h, w]
    """

    device = segs.device

    batch_average_areas = torch.zeros(segs.shape[0]).to(device)
    
    for batch_no, seg in enumerate(segs):
        unique_values = np.unique(seg)    
        areas_log = []
        for val in unique_values:
            binary_mask = (seg.detach().cpu().numpy() == val).astype(int)
            
            # Label each connected component (object) for the current value
            labeled, num_features = label(binary_mask)
            
            # Compute area of each labeled region
            areas = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            areas_log.append(areas)

        areas_log = np.array(areas_log)

        if average_type == "mean":
            average_area = np.mean(areas_log)
        elif average_type == "median":
            average_area =  np.median(areas_log)
        
        batch_average_areas[batch_no] = average_area

        return batch_average_areas
    

"""
^^^ 
|||

except we want to do this in a sliding window fashion, i.e. calculate a map of average blob areas for each segmentation in the batch
"""

        
    
    




