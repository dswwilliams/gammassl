import numpy as np
import torch
import torch.nn.functional as F
from utils.device_utils import to_device


def calculate_miou(segmentations, labels, num_classes):
    total_iou = 0
    n_active_classes = 0
    for k in range(num_classes):
        class_seg_mask = (segmentations == k).float()
        class_label_mask = (labels == k).float()

        intersection = (class_seg_mask * class_label_mask).sum()
        union = torch.max(class_seg_mask, class_label_mask).sum()

        if not (union == 0):
            iou = intersection/union
            total_iou += iou
            n_active_classes += 1

    if n_active_classes == 0:
        miou = 0
    else:
        miou = total_iou/n_active_classes
    return miou

def init_val_ue_metrics(num_thresholds):
    ue_metrics_totals = {}
    ue_metrics_counts = {}

    ue_metrics_totals["acc_md"] = torch.zeros(num_thresholds)
    ue_metrics_counts["acc_md"] = torch.zeros(num_thresholds)
    ue_metrics_totals["fhalf"] = torch.zeros(num_thresholds)
    ue_metrics_counts["fhalf"] = torch.zeros(num_thresholds)
    ue_metrics_totals["p_certain"] = torch.zeros(num_thresholds)
    ue_metrics_counts["p_certain"] = torch.zeros(num_thresholds)
    ue_metrics_totals["p_accurate"] = torch.zeros(num_thresholds)
    ue_metrics_counts["p_accurate"] = torch.zeros(num_thresholds)


    return ue_metrics_totals, ue_metrics_counts