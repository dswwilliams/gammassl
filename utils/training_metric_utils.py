import torch
from ue_testing.test_utils import calculate_fbeta_score, calculate_accuracy, calculate_miou

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_states():
    """
    Define certainty as the prediction, and consistency as the pseudo ground truth.
    Therefore false positives are uncertain and consistent, false negatives are certain and inconsistent.
    Table:
    -------------------------------------------------
    | Predicted | 'Ground Truth'   | State            |
    |-----------|----------------|------------------|
    | Certain   | Consistent     | True Positive   |
    | Certain   | Inconsistent   | False Positive  |
    | Uncertain | Consistent     | False Negative  |
    | Uncertain | Inconsistent   | True Negative   |
    -------------------------------------------------
    """
    states = {}
    states["tp"] = "n_consistent_and_certain"
    states["tn"] = "n_inconsistent_and_uncertain"
    states["fp"] = "n_inconsistent_and_certain"
    states["fn"] = "n_consistent_and_uncertain"
    return states

def calculate_p_certain_per_class(known_class_mask, segs_q, segs_t, num_known_classes):
    device = segs_q.device
    p_certain_per_class_q = torch.zeros(num_known_classes).to(device)
    p_certain_per_class_t = torch.zeros(num_known_classes).to(device)
    for k in range(num_known_classes):
        p_certain_per_class_q[k]= torch.eq((known_class_mask * (segs_q + 1) - 1), k).float().sum() / known_class_mask.sum()
        p_certain_per_class_t[k]= torch.eq((known_class_mask * (segs_t + 1) - 1), k).float().sum() / known_class_mask.sum()
    return p_certain_per_class_q.cpu(), p_certain_per_class_t.cpu()

def calculate_p_consistent_per_class(consistency_masks, segs_t, num_known_classes):
    """
    Calculate the proportion of consistent pixels that belong to each class.
    The class is defined by the target segmentation.
    """
    device = segs_t.device
    p_consistent_per_class = torch.zeros(num_known_classes).to(device)
    for k in range(num_known_classes):
        p_consistent_per_class[k]= torch.eq((consistency_masks * (segs_t + 1) - 1), k).float().sum() / consistency_masks.sum()
    return p_consistent_per_class.cpu()


@torch.no_grad()
def calculate_ue_training_metrics(consistency_masks, certainty_masks):
    consistency_masks = consistency_masks.bool()
    certainty_masks = certainty_masks.bool()

    inconsistency_masks = ~consistency_masks
    uncertainty_masks = ~certainty_masks

    metric_totals = {}
    metric_totals["n_consistent_and_certain"] = torch.bitwise_and(consistency_masks, certainty_masks).sum((1,2))
    metric_totals["n_consistent_and_uncertain"] = torch.bitwise_and(consistency_masks, uncertainty_masks).sum((1,2))
    metric_totals["n_inconsistent_and_certain"] = torch.bitwise_and(inconsistency_masks, certainty_masks).sum((1,2))
    metric_totals["n_inconsistent_and_uncertain"] = torch.bitwise_and(inconsistency_masks, uncertainty_masks).sum((1,2))

    states = get_states()

    f1 = calculate_fbeta_score(metric_totals, states, beta=1).cpu()
    accuracy = calculate_accuracy(metric_totals, states).cpu()
    return f1, accuracy



@torch.no_grad()
def get_consistency_metrics(p_y_given_x_t, p_y_given_x_q, certainty_masks, detailed_metrics=False):
    """
    Compute and return consistency training metrics.

    Args:
        p_y_given_x_t (torch.Tensor): Pixel-wise categorical distribution from the target branch.
        p_y_given_x_q (torch.Tensor): Pixel-wise categorical distribution from the query branch.
        certainty_masks (torch.Tensor): Mask where 1 indicates certain pixels, 0 indicates uncertain pixels.
        detailed_metrics (bool): If True, compute and return a comprehensive set of metrics.
                                  If False, compute and return only key metrics.
    
    Returns:
        A dictionary of computed metrics.
    """
    # remove logits from graph
    p_y_given_x_t, p_y_given_x_q = p_y_given_x_t.detach(), p_y_given_x_q.detach()

    metrics = {}
    num_known_classes = p_y_given_x_t.shape[1]

    # calculate mean max softmax, i.e. mean certainty
    metrics["mean_max_softmax_t"] = torch.max(p_y_given_x_t, dim=1).values.mean().cpu()
    metrics["mean_max_softmax_q"] = torch.max(p_y_given_x_q, dim=1).values.mean().cpu()

    # calculate proportion of certain pixels
    metrics["p_certain_q"] = certainty_masks.mean().cpu()

    # calculate segmentation from seg masks
    segs_t = torch.argmax(p_y_given_x_t, dim=1)
    segs_q = torch.argmax(p_y_given_x_q, dim=1)

    consistency_masks = torch.eq(segs_q, segs_t).float()
    metrics["p_consistent"] = consistency_masks.float().mean().cpu()

    if detailed_metrics:
        metrics["p_certain_per_class_q"], metrics["p_certain_per_class_t"] = calculate_p_certain_per_class(
                                                                                                        certainty_masks, 
                                                                                                        segs_q, 
                                                                                                        segs_t, 
                                                                                                        num_known_classes,
                                                                                                        )

        # of the pixels that are consistent, calculate which classes they belong to
        metrics["p_consistent_per_class"] = calculate_p_consistent_per_class(
                                                consistency_masks, 
                                                segs_t, 
                                                num_known_classes,
                                                )

        # calculate uncertainty estimation metrics using consistency as pseudo ground truth
        metrics["f1_score"], metrics["acc_md"] = calculate_ue_training_metrics(consistency_masks, certainty_masks)

    return metrics



def calculate_supervised_metrics(seg_masks, labels):
    # remove from graph
    seg_masks = seg_masks.detach()
    
    metrics = {}
    num_known_classes = seg_masks.shape[1]
    segs = torch.argmax(seg_masks, dim=1)

    metrics["labelled_miou"] = calculate_miou(segs, labels, num_classes=num_known_classes)
    metrics["labelled_accuracy"] = torch.eq(segs, labels).float().mean()
    metrics["labelled_mean_max_softmax"] = torch.max(torch.softmax(seg_masks, dim=1), dim=1).values.mean().cpu()

    return metrics