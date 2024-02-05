import torch
"""
TODO
- either add more here, or move to a more appropriate location
"""

def calculate_p_certain_per_class(known_class_mask, segs_q, segs_t, num_known_classes):
    p_certain_per_class_q = torch.zeros(num_known_classes).cpu()
    p_certain_per_class_t = torch.zeros(num_known_classes).cpu()
    for k in range(num_known_classes):
        p_certain_per_class_q[k]= torch.eq((known_class_mask.cpu() * (segs_q.cpu() + 1) - 1), k).float().sum() / known_class_mask.cpu().sum()
        p_certain_per_class_t[k]= torch.eq((known_class_mask.cpu() * (segs_t.cpu() + 1) - 1), k).float().sum() / known_class_mask.cpu().sum()
    return p_certain_per_class_q, p_certain_per_class_t


def calculate_consistency2certainty_prob_metrics(confidence_masks, accuracy_masks):
    """
    - if denominator is 0 for a batch element, it shouldnt be considered in the mean across that batch
    -> leave as NaN and perform nanmean()

    confidence_masks = certainty masks
    accuracy_masks = consistency masks    
    """
    accuracy_masks = accuracy_masks.float()
    confidence_masks = confidence_masks.float()

    ################################################################################################################
    ### compute coverage ###
    coverage = confidence_masks.mean((1,2))      # shape: [bs,]
    pixel_accuracy = accuracy_masks.mean((1,2))
    ################################################################################################################

    ################################################################################################################
    ### compute p_accurate_given_certain ###
    n_accurate_and_certain = (accuracy_masks * confidence_masks).sum((1,2))
    n_certain = (confidence_masks).sum((1,2))
    p_accurate_given_certain = n_accurate_and_certain / n_certain   
    # returning NaN for batch element if n_certain is 0
    ################################################################################################################

    ################################################################################################################
    ### compute p_uncertain_given_inaccurate ###
    n_uncertain_and_inaccurate = ((1-accuracy_masks) * (1-confidence_masks)).sum((1,2))
    n_inaccurate = ((1-accuracy_masks)).sum((1,2))
    n_inaccurate_or_accurate = torch.ones_like(accuracy_masks).sum((1,2))
    p_uncertain_given_inaccurate = n_uncertain_and_inaccurate / n_inaccurate
    # returning NaN for batch element if n_inaccurate is 0
    ################################################################################################################

    ################################################################################################################
    ### compute combined metric ###
    combined_prob_metric = (n_uncertain_and_inaccurate + n_accurate_and_certain) / n_inaccurate_or_accurate
    ################################################################################################################


    val_certainty_metrics = {}
    val_certainty_metrics["p_accurate_given_certain"] = p_accurate_given_certain
    val_certainty_metrics["p_uncertain_given_inaccurate"] = p_uncertain_given_inaccurate
    val_certainty_metrics["combined_prob_metric"] = combined_prob_metric
    val_certainty_metrics["p_certain"] = coverage
    # val_certainty_metrics["p_accurate"] = pixel_accuracy
    return val_certainty_metrics